from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from google import genai
import os

app = FastAPI()

GEMINI_MODEL = "gemini-3.8-flash"


def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured. Add GEMINI_API_KEY in the project Secrets.",
        )
    return genai.Client(api_key=api_key)

class MathQuery(BaseModel):
    problem: str

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/solve")
def solve_math(query: MathQuery):
    try:
        prompt = f"""
        You are an expert math tutor. Solve the following math problem step-by-step.
        Format the response clearly with numbered steps and a final answer.
        
        Problem: {query.problem}
        """
        
        response = get_gemini_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        
        if not response.text:
            raise HTTPException(status_code=502, detail="Gemini returned an empty response.")
        return {"solution": response.text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Math Solver</title>
        <style>
            body { font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; max-width: 600px; margin: auto; }
            h2 { color: #38bdf8; }
            textarea { width: 100%; height: 100px; padding: 10px; background: #1e293b; color: #fff; border: 1px solid #334155; border-radius: 8px; margin-bottom: 10px; box-sizing: border-box; }
            button { background: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%; }
            button:hover { background: #0369a1; }
            .result { margin-top: 20px; background: #1e293b; padding: 15px; border-radius: 8px; white-space: pre-wrap; border: 1px solid #334155; }
        </style>
    </head>
    <body>
        <h2>AI Math Tutor & Solver</h2>
        <p>Type your math problem below:</p>
        <textarea id="problem" placeholder="e.g., Solve 3x + 5 = 20 or find the derivative of x^2 + 3x"></textarea>
        <br>
        <button onclick="solveMath()">Solve Problem</button>
        <div class="result" id="output">Your solution will appear here...</div>

        <script>
            async function solveMath() {
                const problem = document.getElementById('problem').value;
                const output = document.getElementById('output');
                if (!problem) return alert('Please enter a math problem!');
                
                output.innerText = 'Solving... Please wait...';
                
                try {
                    const res = await fetch('/solve', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ problem })
                    });
                    const data = await res.json();
                    if(res.ok) {
                        output.innerText = data.solution;
                    } else {
                        output.innerText = 'Error: ' + data.detail;
                    }
                } catch (err) {
                    output.innerText = 'Network error occurred.';
                }
            }
        </script>
    </body>
    </html>
    """
