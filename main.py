import asyncio
import logging
import os
from functools import lru_cache

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI()

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_HTTP_TIMEOUT_MS = 15_000
GEMINI_REQUEST_TIMEOUT_SECONDS = 18
logger = logging.getLogger(__name__)


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
        
        response = await asyncio.wait_for(
            get_gemini_client().aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            ),
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        )
        
        if not response.text:
            raise HTTPException(status_code=502, detail="Gemini returned an empty response.")
        return {"solution": response.text}
    except HTTPException:
        raise
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        logger.warning("Gemini request timed out: %s", exc)
        raise HTTPException(
            status_code=504,
            detail="Gemini took too long to respond. Please try again.",
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

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Math Solver</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
        <style>
            body { font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; max-width: 600px; margin: auto; }
            h2 { color: #38bdf8; }
            textarea { width: 100%; height: 100px; padding: 10px; background: #1e293b; color: #fff; border: 1px solid #334155; border-radius: 8px; margin-bottom: 10px; box-sizing: border-box; }
            button { background: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%; }
            button:hover { background: #0369a1; }
            .result { margin-top: 20px; background: #1e293b; padding: 15px; border-radius: 8px; white-space: pre-wrap; border: 1px solid #334155; }
            .katex-display { overflow-x: auto; overflow-y: hidden; }
        </style>
    </head>
    <body>
        <h2>AI Math Tutor & Solver</h2>
        <p>Type your math problem below:</p>
        <textarea id="problem" placeholder="e.g., Solve 3x + 5 = 20 or find the derivative of x^2 + 3x"></textarea>
        <br>
        <button id="solve-button" onclick="solveMath()">Solve Problem</button>
        <div class="result" id="output">Your solution will appear here...</div>

        <script src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>
        <script>
            async function solveMath() {
                const problem = document.getElementById('problem').value;
                const output = document.getElementById('output');
                const solveButton = document.getElementById('solve-button');
                if (!problem) return alert('Please enter a math problem!');
                
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 25000);
                solveButton.disabled = true;
                output.textContent = 'Solving... Please wait...';
                
                try {
                    const res = await fetch('/solve', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ problem }),
                        signal: controller.signal
                    });
                    const data = await res.json();
                    if(res.ok) {
                        output.textContent = data.solution;
                        if (typeof renderMathInElement === 'function') {
                            renderMathInElement(output, {
                                delimiters: [
                                    { left: '$$', right: '$$', display: true },
                                    { left: '\\\\[', right: '\\\\]', display: true },
                                    { left: '\\\\(', right: '\\\\)', display: false },
                                    { left: '$', right: '$', display: false }
                                ],
                                throwOnError: false
                            });
                        }
                    } else {
                        output.textContent = 'Error: ' + (data.detail || 'The solution could not be generated.');
                    }
                } catch (err) {
                    output.textContent = err.name === 'AbortError'
                        ? 'The request timed out. Please try again.'
                        : 'Network error occurred.';
                } finally {
                    clearTimeout(timeoutId);
                    solveButton.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
