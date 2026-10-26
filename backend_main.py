import os
import json
import hmac
import hashlib
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from supabase import create_client, Client
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from groq import Groq
import google.generativeai as genai

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================

app = FastAPI(
    title="KingMath AI Backend",
    description="Production API for KingMath AI: AI Problem Solving, CBT Engine, and Paystack Billing",
    version="1.0.0"
)

# Setup Rate Limiter (by IP address)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS for Vercel Frontend
ALLOWED_ORIGINS = [
    "https://king-math-ai.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "kingmath-admin-secret-2026")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Clients
supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# 2. PYDANTIC SCHEMAS
# ==========================================

class SolveRequest(BaseModel):
    user_id: Optional[str] = None
    prompt: str
    image_base64: Optional[str] = None

class PaystackInitRequest(BaseModel):
    email: str
    amount: Optional[int] = 3500

class CBTSubmitRequest(BaseModel):
    user_id: str
    exam_type: str
    subject: str
    answers: dict

# ==========================================
# 3. BASE & HEALTH ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "KingMath AI Backend API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "supabase": supabase is not None,
        "groq": groq_client is not None,
        "gemini": GEMINI_API_KEY is not None,
        "paystack": PAYSTACK_SECRET_KEY is not None
    }

# ==========================================
# 4. AI PROBLEM SOLVER ENDPOINT
# ==========================================

@app.post("/api/solve")
@limiter.limit("20/minute")
def solve_math_problem(request: Request, body: SolveRequest):
    if not body.prompt and not body.image_base64:
        raise HTTPException(status_code=400, detail="Please provide a text prompt or an image.")

    system_prompt = (
        "You are KingMath AI, an elite Nigerian mathematics tutor specializing in WAEC, JAMB UTME, "
        "Further Mathematics, and secondary school mathematics.\n"
        "Instructions:\n"
        "1. Provide a step-by-step worked solution.\n"
        "2. Format all mathematical equations using standard LaTeX syntax enclosed in $...$ for inline "
        "   and $$...$$ for block equations.\n"
        "3. Do NOT use raw markdown formatting like '--- ### Step 1:'. Use clear text headers like '### Step 1:'.\n"
        "4. Include final answer clearly formatted."
    )

    try:
        if body.image_base64:
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=500, detail="Gemini API Key is not configured for Vision requests.")
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            image_data = {"mime_type": "image/jpeg", "data": body.image_base64}
            response = model.generate_content([system_prompt, body.prompt or "Solve this math problem step-by-step.", image_data])
            return {"solution": response.text, "model": "Gemini 1.5 Flash (Vision)"}
        else:
            if not groq_client:
                raise HTTPException(status_code=500, detail="Groq API Key is not configured for Text requests.")
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": body.prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=1500,
            )
            return {"solution": chat_completion.choices[0].message.content, "model": "Groq Llama-3.3 70B"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Solver Error: {str(e)}")

# ==========================================
# 5. CBT QUESTION BANK & GRADING ENDPOINTS
# ==========================================

@app.get("/api/cbt/questions")
def get_cbt_questions(exam_type: str = "JAMB UTME", subject: str = "Mathematics", limit: int = 10):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client is not initialized.")
    
    try:
        res = supabase.table("cbt_questions") \
            .select("id, exam_type, subject, topic, question, option_a, option_b, option_c, option_d") \
            .eq("exam_type", exam_type) \
            .eq("subject", subject) \
            .limit(limit) \
            .execute()
        return {"questions": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch CBT questions: {str(e)}")

@app.post("/api/cbt/submit")
def submit_cbt_exam(body: CBTSubmitRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client is not initialized.")

    try:
        q_ids = list(body.answers.keys())
        res = supabase.table("cbt_questions").select("id, correct_option, explanation").in_("id", q_ids).execute()
        
        correct_map = {str(item["id"]): item for item in res.data}
        total_questions = len(q_ids)
        score =
