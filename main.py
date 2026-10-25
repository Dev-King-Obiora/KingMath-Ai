import asyncio
import logging
import os
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from google import genai
from google.genai import types
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.6-flash"
GEMINI_HTTP_TIMEOUT_MS = 30_000
GEMINI_ATTEMPT_TIMEOUT_SECONDS = 12
GEMINI_REQUEST_TIMEOUT_SECONDS =24
TRANSIENT_GEMINI_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured. Add GEMINI_API_KEY in the project Secrets.",
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_HTTP_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

class MathQuery(BaseModel):
    problem: str


async def generate_with_fallback(client, prompt):
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            ),
            timeout=GEMINI_ATTEMPT_TIMEOUT_SECONDS,
        )
        return response, GEMINI_MODEL
    except (asyncio.TimeoutError, httpx.TransportError) as exc:
        logger.warning(
            "Primary Gemini model %s timed out or had a transport error; trying %s: %s",
            GEMINI_MODEL,
            GEMINI_FALLBACK_MODEL,
            exc,
        )
    except genai.errors.APIError as exc:
        status_code = getattr(exc, "code", None)
        if status_code not in TRANSIENT_GEMINI_STATUS_CODES:
            raise
        logger.warning(
            "Primary Gemini model %s returned transient HTTP %s; trying %s",
            GEMINI_MODEL,
            status_code,
            GEMINI_FALLBACK_MODEL,
        )

    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=GEMINI_FALLBACK_MODEL,
            contents=prompt,
        ),
        timeout=GEMINI_ATTEMPT_TIMEOUT_SECONDS,
    )
    return response, GEMINI_FALLBACK_MODEL


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
        
        response, model_used = await asyncio.wait_for(
            generate_with_fallback(get_gemini_client(), prompt),
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        )
        
        if not response.text:
            raise HTTPException(status_code=502, detail="Gemini returned an empty response.")
        logger.info("Math solution generated with %s", model_used)
        return {"solution": response.text}
    except HTTPException:
        raise
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        logger.warning("Gemini request timed out: %s", exc)
        raise HTTPException(
            status_code=504,
            detail="Gemini took too long to respond. Please try again.",
        ) from exc
    except httpx.TransportError as exc:
        logger.warning("Gemini request failed with a transport error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Could not reach Gemini right now. Please try again shortly.",
        ) from exc
    except genai.errors.APIError as exc:
        logger.error(
            "Gemini API request failed: code=%s message=%s",
            getattr(exc, "code", None),
            getattr(exc, "message", str(exc)),
        )
        status_code = getattr(exc, "code", None)
        if status_code in (408, 504):
            raise HTTPException(
                status_code=504,
                detail="Gemini timed out before completing the request. Please try again.",
            ) from exc
        if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily unavailable. Please try again shortly.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"Gemini rejected the request (HTTP {status_code or 'unknown'}).",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while requesting Gemini")
        raise HTTPException(
            status_code=502,
            detail="Gemini request failed. Please try again.",
        ) from exc

@app.get("/")
async def home():
    return FileResponse(APP_DIR / "index.html")
