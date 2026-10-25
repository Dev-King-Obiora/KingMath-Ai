import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq, GroqError

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated to Groq's active high-performance models
GROQ_PRIMARY_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT_SECONDS = 20

logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Groq is not configured. Add GROQ_API_KEY in the project Secrets.",
        )
    return AsyncGroq(api_key=api_key)


class MathQuery(BaseModel):
    problem: str


async def generate_with_fallback(client: AsyncGroq, prompt: str):
    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=GROQ_PRIMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=12,
        )
        return completion.choices[0].message.content, GROQ_PRIMARY_MODEL
    except Exception as exc:
        logger.warning("Primary model %s failed: %s. Trying fallback...", GROQ_PRIMARY_MODEL, exc)

    completion = await asyncio.wait_for(
        client.chat.completions.create(
            model=GROQ_FALLBACK_MODEL,
            messages=[{"role": "user", "content": prompt}],
        ),
        timeout=12,
    )
    return completion.choices[0].message.content, GROQ_FALLBACK_MODEL


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/solve")
async def solve_math(query: MathQuery):
    try:
        prompt = f"""
        You are an expert math tutor. Solve the following math problem step-by-step.
        Format the response clearly with numbered steps and a final answer.
        Wrap inline mathematics in \\( ... \\) and standalone equations in \\[ ... \\].
        Do not leave LaTeX commands such as \\frac or math expressions without delimiters.
        
        Problem: {query.problem}
        """
        
        client = get_groq_client()
        solution_text, model_used = await asyncio.wait_for(
            generate_with_fallback(client, prompt),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        
        if not solution_text:
            raise HTTPException(status_code=502, detail="KingMath-Ai returned an empty response.")
        logger.info("Math solution generated with %s", model_used)
        return {"solution": solution_text}

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="KingMath-Ai took too long to respond. Please try again.",
        )
    except GroqError as exc:
        logger.error("Groq API error: %s", exc)
        status_code = getattr(exc, "status_code", 500)
        if status_code == 429:
            raise HTTPException(
                status_code=503,
                detail="KingMath-Ai is temporarily busy. Please wait a few seconds and try again.",
            )
        raise HTTPException(
            status_code=502,
            detail=f"KingMath-Ai request failed (HTTP {status_code}).",
        )
    except Exception as exc:
        logger.exception("Unexpected error during processing")
        raise HTTPException(
            status_code=502,
            detail="KingMath-Ai request failed. Please try again.",
        )


@app.get("/")
async def home():
    return FileResponse(APP_DIR / "index.html")
