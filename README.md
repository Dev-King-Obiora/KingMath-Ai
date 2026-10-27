# KingMath-AI

A compact FastAPI + single-page frontend starter for an AI mathematics/science tutor and dynamically generated CBT practice exams.

## Run locally or in Replit

1. Install Python 3.10+ dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` for local use and set `GROQ_API_KEY` and `GEMINI_API_KEY`. In Replit, add these as Secrets instead of committing a `.env` file.
3. Start the server: `uvicorn main:app --host 0.0.0.0 --port 8000`
4. Open the app at the URL provided by Replit or `http://127.0.0.1:8000` locally.

Text solving and CBT generation use the configured Groq-compatible chat API. Image solving uses the configured Gemini API. Keys are read only by the server.

## Important implementation notes

- The UI is contained in `index.html`; API routes and session logic are in `main.py`.
- CBT questions are generated on demand. The answer key and explanations are held in server memory and are not sent to the browser until the exam is submitted.
- Active CBT sessions are in memory. A process restart invalidates them; users are asked to start a new exam.
- The in-memory per-IP rate limiter is a basic starter safeguard, not a substitute for a production gateway/WAF. Add persistent storage and stronger identity/session protections before public high-traffic deployment.
- Automated syntax/startup checks can be run with `python -m py_compile main.py` and `uvicorn main:app --host 127.0.0.1 --port 8000`.
