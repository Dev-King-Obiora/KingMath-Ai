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
    "https://kingmath-ai.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to ALLOWED_ORIGINS in production if strict CORS is preferred
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
    answers: dict  # e.g., {"q_id_1": "A", "q_id_2": "C"}

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
# 4. AI PROBLEM SOLVER ENDPOINT (TEXT & VISION)
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
        # Vision Request (Gemini)
        if body.image_base64:
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=500, detail="Gemini API Key is not configured for Vision requests.")
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            image_data = {"mime_type": "image/jpeg", "data": body.image_base64}
            response = model.generate_content([system_prompt, body.prompt or "Solve this math problem step-by-step.", image_data])
            return {"solution": response.text, "model": "Gemini 1.5 Flash (Vision)"}

        # Text Request (Groq)
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
        # Fetch correct answers for questions
        q_ids = list(body.answers.keys())
        res = supabase.table("cbt_questions").select("id, correct_option, explanation").in_("id", q_ids).execute()
        
        correct_map = {str(item["id"]): item for item in res.data}
        total_questions = len(q_ids)
        score = 0
        breakdown = []

        for q_id, user_ans in body.answers.items():
            q_info = correct_map.get(str(q_id), {})
            is_correct = (q_info.get("correct_option") == user_ans)
            if is_correct:
                score += 1
            breakdown.append({
                "question_id": q_id,
                "user_answer": user_ans,
                "correct_answer": q_info.get("correct_option"),
                "is_correct": is_correct,
                "explanation": q_info.get("explanation", "")
            })

        percentage = round((score / total_questions * 100), 2) if total_questions > 0 else 0

        # Store test results in Supabase
        supabase.table("cbt_results").insert({
            "user_id": body.user_id,
            "exam_type": body.exam_type,
            "subject": body.subject,
            "score": score,
            "total": total_questions,
            "percentage": percentage
        }).execute()

        return {
            "score": score,
            "total": total_questions,
            "percentage": percentage,
            "breakdown": breakdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit CBT exam: {str(e)}")

# ==========================================
# 6. PAYSTACK PAYMENTS & WEBHOOKS
# ==========================================

@app.post("/api/paystack/initialize")
def initialize_paystack_payment(data: PaystackInitRequest):
    if not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Paystack secret key is not configured on server.")

    amount_kobo = int(data.amount * 100)  # Convert Naira to Kobo
    url = "https://api.paystack.co/transaction/initialize"
    headers = {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "email": data.email,
        "amount": amount_kobo,
        "callback_url": "https://king-math-ai.vercel.app"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        res_data = response.json()
        if response.status_code != 200 or not res_data.get("status"):
            raise HTTPException(status_code=400, detail=res_data.get("message", "Paystack initialization failed"))
        
        return {
            "authorization_url": res_data["data"]["authorization_url"],
            "access_code": res_data["data"]["access_code"],
            "reference": res_data["data"]["reference"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Paystack Error: {str(e)}")

@app.post("/api/paystack-webhook")
async def paystack_webhook(request: Request, x_paystack_signature: Optional[str] = Header(None)):
    if not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Webhook Secret Not Configured")
    
    body_bytes = await request.body()
    
    # Verify Paystack HMAC SHA512 Signature
    computed_hmac = hmac.new(
        PAYSTACK_SECRET_KEY.encode('utf-8'),
        body_bytes,
        hashlib.sha512
    ).hexdigest()

    if x_paystack_signature != computed_hmac:
        raise HTTPException(status_code=400, detail="Invalid Paystack Webhook Signature")

    payload = json.loads(body_bytes.decode('utf-8'))
    event = payload.get("event")

    if event == "charge.success":
        data = payload.get("data", {})
        customer_email = data.get("customer", {}).get("email")

        if customer_email and supabase:
            # Upgrade user account in Supabase
            supabase.table("profiles").update({
                "tier": "pro",
                "daily_credits": 999999
            }).eq("email", customer_email).execute()

    return {"status": "success"}

# ==========================================
# 7. PROTECTED ADMIN MONITORING ENDPOINT
# ==========================================

@app.get("/api/admin/stats")
def get_admin_stats(x_admin_key: Optional[str] = Header(None)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized access: Invalid Admin Key")

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    try:
        users_count = supabase.table("profiles").select("id", count="exact").execute().count
        pro_users_count = supabase.table("profiles").select("id", count="exact").eq("tier", "pro").execute().count
        cbt_tests_taken = supabase.table("cbt_results").select("id", count="exact").execute().count

        return {
            "total_registered_users": users_count or 0,
            "active_pro_subscriptions": pro_users_count or 0,
            "total_cbt_exams_completed": cbt_tests_taken or 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin Query Error: {str(e)}")
