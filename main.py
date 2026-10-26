import asyncio
import base64
import hmac
import hashlib
import logging
import os
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, Request, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from groq import AsyncGroq, GroqError
from pydantic import BaseModel, Field
import httpx
from supabase import create_client, Client

# SlowAPI Rate Limiting imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# -----------------------------------------------------------------------------
# 1. APPLICATION & MIDDLEWARE SETUP
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
APP_DIR = Path(__file__).resolve().parent

# Initialize Rate Limiter based on IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="KingMath AI API",
    description="Backend engine for KingMath AI: Math Solver, CBT, Learning Hub, and Subscription System",
    version="2.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS - Restrict allowed origins in production for maximum security
ALLOWED_ORIGINS = [
    "https://king-math-ai.vercel.app",
    "https://kingmath-ai.vercel.app",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 2. CONFIGURATION & CLIENT LAZY LOADING
# -----------------------------------------------------------------------------
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT_SECONDS = 25
FREE_DAILY_CREDIT_LIMIT = 10

@lru_cache(maxsize=1)
def get_groq_client() -> AsyncGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Groq API Key is not configured.")
    return AsyncGroq(api_key=api_key)

@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Gemini API Key is not configured.")
    return genai.Client(api_key=api_key)

@lru_cache(maxsize=1)
def get_supabase_admin() -> Client:
    """Returns a Supabase client with admin privileges (using service role key)."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase configuration missing.")
    return create_client(url, key)

# -----------------------------------------------------------------------------
# 3. DATA MODELS (PYDANTIC)
# -----------------------------------------------------------------------------
class MathQuery(BaseModel):
    problem: str
    image_base64: Optional[str] = None

class CBTStartRequest(BaseModel):
    exam_type: str = Field(..., example="WAEC")  # WAEC, JAMB, SS1, SS2, SS3
    subject: str = Field(..., example="Mathematics")
    topic: Optional[str] = None
    question_count: int = Field(default=10, ge=1, le=50)

class CBTAnswerItem(BaseModel):
    question_id: str
    selected_option: str  # "A", "B", "C", "D"

class CBTSubmitRequest(BaseModel):
    exam_type: str
    subject: str
    topic: Optional[str] = None
    answers: List[CBTAnswerItem]

class SubscriptionInitRequest(BaseModel):
    user_email: str
    user_id: str

# -----------------------------------------------------------------------------
# 4. AUTHENTICATION & SECURITY HELPER FUNCTIONS
# -----------------------------------------------------------------------------
async def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Decodes Supabase JWT token if present. Returns None for guest users."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    try:
        supabase = get_supabase_admin()
        user_response = supabase.auth.get_user(token)
        if user_response and user_response.user:
            return {
                "id": user_response.user.id,
                "email": user_response.user.email
            }
    except Exception as err:
        logger.warning(f"Invalid token provided: {err}")
    return None

def verify_and_deduct_credits(user_id: str) -> Dict[str, Any]:
    """Checks user subscription tier and daily credit allowance."""
    supabase = get_supabase_admin()
    res = supabase.table("profiles").select("*").eq("id", user_id).execute()
    
    if not res.data:
        # Auto-create profile if missing
        new_profile = {
            "id": user_id,
            "tier": "free",
            "daily_credits_used": 0,
            "last_credit_reset": date.today().isoformat()
        }
        supabase.table("profiles").insert(new_profile).execute()
        profile = new_profile
    else:
        profile = res.data[0]

    # Pro users have unlimited access
    if profile.get("tier") == "pro":
        return {"tier": "pro", "credits_remaining": 999999}

    today_str = date.today().isoformat()
    last_reset = profile.get("last_credit_reset")
    credits_used = profile.get("daily_credits_used", 0)

    # Reset daily quota if new day
    if last_reset != today_str:
        credits_used = 0
        supabase.table("profiles").update({
            "daily_credits_used": 0,
            "last_credit_reset": today_str
        }).eq("id", user_id).execute()

    if credits_used >= FREE_DAILY_CREDIT_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"You have reached your daily limit of {FREE_DAILY_CREDIT_LIMIT} free AI solves. Upgrade to KingMath AI Pro (₦3,500/month) for unlimited solves!"
        )

    # Deduct 1 credit
    supabase.table("profiles").update({
        "daily_credits_used": credits_used + 1
    }).eq("id", user_id).execute()

    return {
        "tier": "free",
        "credits_remaining": FREE_DAILY_CREDIT_LIMIT - (credits_used + 1)
    }

# -----------------------------------------------------------------------------
# 5. AI SOLVER CORE ENGINES
# -----------------------------------------------------------------------------
async def solve_with_gemini(prompt: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Uses Google Gemini API to solve math from image uploads."""
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
    """Uses Groq LPU API for ultra-fast text-only math solutions."""
    client = get_groq_client()
    completion = await asyncio.wait_for(
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are KingMath AI, an elite math tutoring assistant for Nigerian students (SS1-SS3, WAEC, JAMB). You solve math problems step-by-step with clear LaTeX equations. If asked who you are, state that you are KingMath AI. Never mention Groq, OpenAI, or ChatGPT."
                },
                {"role": "user", "content": prompt}
            ],
        ),
        timeout=15,
    )
    return completion.choices[0].message.content

# -----------------------------------------------------------------------------
# 6. ROUTE ENDPOINTS
# -----------------------------------------------------------------------------

# --- A. AI MATH SOLVER ---
@app.post("/solve")
@limiter.limit("20/minute")
async def solve_math(request: Request, query: MathQuery, current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    """Solves a math problem via text or image after verifying user quotas."""
    quota_status = {"tier": "free", "credits_remaining": FREE_DAILY_CREDIT_LIMIT}
    
    if current_user:
        quota_status = verify_and_deduct_credits(current_user["id"])

    try:
        system_prompt = """
        You are KingMath AI, an expert math tutor. Solve the following problem step-by-step.
        Format response clearly with numbered steps and final answer.
        Wrap inline math in \\( ... \\) and standalone equations in \\[ ... \\].
        Never leave unformatted LaTeX commands like \\frac without delimiters.
        """

        # Gemini Vision Route (for photo uploads)
        if query.image_base64:
            logger.info("Image problem received: Routing to Gemini Vision API")
            image_data = base64.b64decode(query.image_base64.split(",")[-1])
            prompt = f"{system_prompt}\nAnalyze the math problem in this image. Additional user notes: {query.problem}"

            solution_text = await asyncio.wait_for(
                solve_with_gemini(prompt, image_data),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return {
                "solution": solution_text,
                "engine": "Gemini Vision",
                "quota": quota_status
            }

        # Groq LPU Route (for text queries)
        logger.info("Text problem received: Routing to Groq LPU API")
        prompt = f"{system_prompt}\nProblem: {query.problem}"
        solution_text = await solve_with_groq(prompt)

        return {
            "solution": solution_text,
            "engine": "Groq LPU",
            "quota": quota_status
        }

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="KingMath AI took too long to compute a solution.")
    except Exception as exc:
        logger.exception("Processing error inside solver")
        raise HTTPException(status_code=502, detail=f"Request failed: {str(exc)}")


# --- B. CBT PRACTICE SYSTEM ---
@app.post("/api/cbt/start")
@limiter.limit("30/minute")
async def start_cbt_session(request: Request, cbt_req: CBTStartRequest, current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    """Fetches CBT questions for WAEC, JAMB, or SS1-SS3 exams without revealing correct answers."""
    supabase = get_supabase_admin()
    
    db_query = supabase.table("cbt_questions").select("id, question, option_a, option_b, option_c, option_d, topic")\
        .eq("exam_type", cbt_req.exam_type)\
        .eq("subject", cbt_req.subject)
        
    if cbt_req.topic:
        db_query = db_query.eq("topic", cbt_req.topic)
        
    res = db_query.limit(cbt_req.question_count).execute()
    
    # Fallback default questions if database table is currently empty
    questions = res.data if res.data else [
        {
            "id": "q1",
            "question": "Solve for x: 2x + 5 = 15",
            "option_a": "x = 3",
            "option_b": "x = 5",
            "option_c": "x = 10",
            "option_d": "x = 7",
            "topic": "Algebra"
        },
        {
            "id": "q2",
            "question": "Evaluate sin(30°)",
            "option_a": "0.5",
            "option_b": "1",
            "option_c": "0.866",
            "option_d": "0",
            "topic": "Trigonometry"
        }
    ]

    return {"exam_type": cbt_req.exam_type, "subject": cbt_req.subject, "questions": questions}

@app.post("/api/cbt/submit")
@limiter.limit("15/minute")
async def submit_cbt_test(request: Request, submit_req: CBTSubmitRequest, current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    """Server-side grades CBT test submissions to prevent cheating."""
    supabase = get_supabase_admin()
    
    # Get question IDs
    question_ids = [a.question_id for a in submit_req.answers]
    
    # Query database for correct answers
    db_res = supabase.table("cbt_questions").select("id, correct_option, explanation, topic").in_("id", question_ids).execute()
    
    answer_key = {q["id"]: q for q in db_res.data} if db_res.data else {
        "q1": {"correct_option": "B", "explanation": "2x = 15 - 5 => 2x = 10 => x = 5", "topic": "Algebra"},
        "q2": {"correct_option": "A", "explanation": "sin(30°) = 1/2 = 0.5", "topic": "Trigonometry"}
    }

    correct_count = 0
    total_questions = len(submit_req.answers)
    topic_performance: Dict[str, Dict[str, int]] = {}

    for user_ans in submit_req.answers:
        q_data = answer_key.get(user_ans.question_id, {})
        correct_op = q_data.get("correct_option")
        topic = q_data.get("topic", "General")

        if topic not in topic_performance:
            topic_performance[topic] = {"correct": 0, "total": 0}
        topic_performance[topic]["total"] += 1

        if user_ans.selected_option.upper() == str(correct_op).upper():
            correct_count += 1
            topic_performance[topic]["correct"] += 1

    score_percentage = round((correct_count / max(total_questions, 1)) * 100, 1)

    # If user is logged in, save result to Supabase progress tracker
    if current_user:
        supabase.table("cbt_results").insert({
            "user_id": current_user["id"],
            "exam_type": submit_req.exam_type,
            "subject": submit_req.subject,
            "score": score_percentage,
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()

    return {
        "score_percentage": score_percentage,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "topic_breakdown": topic_performance
    }


# --- C. PAYSTACK SUBSCRIPTION & WEBHOOK INTEGRATION ---
@app.post("/api/subscribe/initialize")
@limiter.limit("5/minute")
async def initialize_paystack_subscription(request: Request, sub_req: SubscriptionInitRequest):
    """Initializes ₦3,500/month subscription on Paystack and returns checkout URL."""
    paystack_secret = os.environ.get("PAYSTACK_SECRET_KEY")
    plan_code = os.environ.get("PAYSTACK_PLAN_CODE")

    if not paystack_secret or not plan_code:
        raise HTTPException(status_code=500, detail="Paystack integration is not configured on server.")

    headers = {
        "Authorization": f"Bearer {paystack_secret}",
        "Content-Type": "application/json"
    }

    payload = {
        "email": sub_req.user_email,
        "amount": "350000",  # ₦3,500 in kobo
        "plan": plan_code,
        "callback_url": "https://king-math-ai.vercel.app/?payment=success",
        "metadata": {
            "user_id": sub_req.user_id,
            "email": sub_req.user_email
        }
    }

    async with httpx.AsyncClient() as client:
        res = await client.post("https://api.paystack.co/transaction/initialize", json=payload, headers=headers)
        if res.status_code != 200:
            raise HTTPException(status_code=400, detail="Paystack failed to initialize payment.")
        data = res.json()
        return {"checkout_url": data["data"]["authorization_url"]}

@app.post("/api/paystack-webhook")
async def paystack_webhook(request: Request):
    """Securely verifies HMAC-SHA512 Paystack webhook signatures to upgrade users to Pro."""
    paystack_secret = os.environ.get("PAYSTACK_SECRET_KEY")
    if not paystack_secret:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Secret missing"})

    signature = request.headers.get("x-paystack-signature")
    body_bytes = await request.body()

    # Verify HMAC-SHA512 Signature
    computed_sig = hmac.new(
        paystack_secret.encode('utf-8'),
        body_bytes,
        hashlib.sha512
    ).hexdigest()

    if signature != computed_sig:
        logger.warning("Paystack webhook signature verification failed!")
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_data = await request.json()
    event_type = event_data.get("event")

    supabase = get_supabase_admin()

    if event_type in ["charge.success", "subscription.create"]:
        data = event_data.get("data", {})
        metadata = data.get("metadata", {})
        user_id = metadata.get("user_id")
        customer_code = data.get("customer", {}).get("customer_code")
        sub_code = data.get("subscription_code")

        if user_id:
            supabase.table("profiles").update({
                "tier": "pro",
                "paystack_customer_code": customer_code,
                "paystack_subscription_code": sub_code
            }).eq("id", user_id).execute()
            logger.info(f"Successfully upgraded user {user_id} to KingMath AI Pro!")

    elif event_type in ["subscription.disable", "invoice.payment_failed"]:
        customer_code = event_data.get("data", {}).get("customer", {}).get("customer_code")
        if customer_code:
            supabase.table("profiles").update({"tier": "free"}).eq("paystack_customer_code", customer_code).execute()
            logger.info(f"Downgraded customer {customer_code} to Free tier due to failed payment or cancellation.")

    return {"status": "success"}


# --- D. ADMIN MONITORING & REVENUE DASHBOARD ---
@app.get("/api/admin/metrics")
async def get_admin_metrics(admin_key: str):
    """Admin-only endpoint to track registered users, Pro subscriptions, and total revenue."""
    expected_key = os.environ.get("ADMIN_SECRET_KEY", "kingmath_admin_2026")
    if admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized admin access.")

    supabase = get_supabase_admin()

    # Query profiles
    profiles_res = supabase.table("profiles").select("id, tier, created_at").execute()
    profiles = profiles_res.data if profiles_res.data else []

    total_users = len(profiles)
    pro_users = sum(1 for p in profiles if p.get("tier") == "pro")
    free_users = total_users - pro_users
    monthly_recurring_revenue = pro_users * 3500  # ₦3,500 per Pro user

    return {
        "platform": "KingMath AI",
        "total_registered_users": total_users,
        "free_tier_users": free_users,
        "active_pro_subscribers": pro_users,
        "monthly_recurring_revenue_ngn": f"₦{monthly_recurring_revenue:,}",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# --- E. SERVE FRONTEND ---
@app.get("/")
async def home():
    """Serves the main application HTML file."""
    return FileResponse(APP_DIR / "index.html")
