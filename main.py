import asyncio
import base64
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from google import genai
from groq import AsyncGroq, GroqError
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT_SECONDS = 25

logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent


class MathQuery(BaseModel):
    problem: str
    image_base64: Optional[str] = None  # Base64 string if user uploads a photo


@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503, detail="Groq API Key is not configured."
        )
    return AsyncGroq(api_key=api_key)


@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503, detail="Gemini API Key is not configured."
        )
    return genai.Client(api_key=api_key)


async def solve_with_gemini(
    prompt: str, image_bytes: bytes, mime_type: str = "image/jpeg"
) -> str:
    """Uses Google Gemini API to extract math from images and solve it step-by-step."""
    client = get_gemini_client()

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=GEMINI_MODEL,
        contents=[
            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text


async def solve_with_groq(prompt: str) -> str:
    """Uses Groq API for sub-second text-only math solving."""
    client = get_groq_client()
    completion = await asyncio.wait_for(
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
    {
        "role": "system",
        "content": "You are KingMath AI, a professional, elite math tutoring assistant. You solve mathematics problems and explain them step-by-step. If anyone asks who you are, you must state that you are KingMath AI. Never mention OpenAI, ChatGPT, or Groq."
    },
    {
        "role": "user",
        "content": prompt
    }
],

        ),
        timeout=15,
    )
    return completion.choices[0].message.content


@app.post("/solve")
async def solve_math(query: MathQuery):
    try:
        system_prompt = """
        You are an expert math tutor. Solve the following math problem step-by-step.
        Format the response clearly with numbered steps and a final answer.
        Wrap inline mathematics in \\( ... \\) and standalone equations in \\[ ... \\].
        Do not leave LaTeX commands such as \\frac or math expressions without delimiters.
        """

        # ROUTE 1: If an image is provided, use Gemini
        if query.image_base64:
            logger.info("Image detected: Routing to Gemini Vision API")
            image_data = base64.b64decode(query.image_base64.split(",")[-1])
            prompt = f"{system_prompt}\nAnalyze the math problem in this image. Additional context from user: {query.problem}"

            solution_text = await asyncio.wait_for(
                solve_with_gemini(prompt, image_data),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return {"solution": solution_text, "engine": "Gemini Vision"}

        # ROUTE 2: If text-only, use Groq for maximum speed
        logger.info("Text-only query: Routing to Groq LPU API")
        prompt = f"{system_prompt}\nProblem: {query.problem}"
        solution_text = await solve_with_groq(prompt)

        return {"solution": solution_text }

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail="KingMath AI took too long to respond."
        )
    except Exception as exc:
        logger.exception("Processing error")
        raise HTTPException(
            status_code=502, detail=f"Request failed: {str(exc)}"
        )


@app.get("/")
async def home():
    return FileResponse(APP_DIR / "index.html")
