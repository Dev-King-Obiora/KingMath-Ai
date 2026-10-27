"""KingMath-AI: FastAPI backend. Secrets stay server-side; CBT answer keys never leave the server before submission."""
from __future__ import annotations
import os, re, json, time, uuid, base64, logging, asyncio
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional
import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kingmath")
BASE = Path(__file__).resolve().parent
app = FastAPI(title="KingMath-AI", docs_url=None if os.getenv("HIDE_API_DOCS", "true").lower()=="true" else "/docs")
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("APP_ORIGIN", "*")], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Session-ID"])

# In-memory for the starter version. Restarting the process invalidates active attempts gracefully.
EXAMS: dict[str, dict[str, Any]] = {}
RATE: dict[str, deque] = defaultdict(deque)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
MAX_TEXT = 6000
MAX_IMAGE = 7 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

class SolveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_TEXT)
    level: str = Field(default="Secondary school", max_length=80)
    conversation: list[dict[str, str]] = Field(default_factory=list, max_length=8)
    image_note: Optional[str] = Field(default=None, max_length=1000)
class ActionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_TEXT)
    solution: dict[str, Any]
    action: str = Field(pattern="^(clearer|simpler|similar|verify)$")
    level: str = Field(default="Secondary school", max_length=80)
class TopicRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    difficulty: str = Field(default="Intermediate", pattern="^(Beginner|Intermediate|Advanced)$")
class ExamGenerateRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    difficulty: str = Field(default="Intermediate", pattern="^(Beginner|Intermediate|Advanced)$")
    duration_minutes: int = Field(ge=1, le=180)
    count: int = Field(default=20, ge=20, le=20)
    level: str = Field(default="Secondary school", max_length=80)
class AnswerRequest(BaseModel):
    question_index: int = Field(ge=0, le=19)
    answer: Optional[str] = Field(default=None, max_length=5)
    review: bool = False
class SubmitRequest(BaseModel):
    force: bool = False

@app.middleware("http")
async def protect_rate(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        key = (request.client.host if request.client else "unknown")
        now = time.time(); bucket = RATE[key]
        while bucket and now - bucket[0] > 60: bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            return JSONResponse({"detail":"You are sending requests too quickly. Please wait a moment and try again."}, status_code=429)
        bucket.append(now)
    return await call_next(request)

async def groq_chat(messages: list[dict[str, Any]], *, json_mode: bool=False, max_tokens: int=3500) -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key: raise HTTPException(503, "AI tutoring is not configured yet. Add the required server key in your environment settings.")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.25, "max_tokens": max_tokens}
    if json_mode: payload["response_format"] = {"type":"json_object"}
    try:
        async with httpx.AsyncClient(timeout=70) as client:
            r = await client.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}, json=payload)
        if r.status_code >= 400:
            log.warning("Text AI request failed status=%s", r.status_code)
            raise HTTPException(503, "KingMath-AI is temporarily unable to complete that request. Please try again.")
        return r.json()["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        raise HTTPException(504, "That request took too long. Please try again.")
    except HTTPException: raise
    except Exception:
        log.exception("Text AI request error")
        raise HTTPException(503, "KingMath-AI is temporarily unavailable. Please try again.")

async def gemini_image(image_bytes: bytes, mime: str, prompt: str) -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key: raise HTTPException(503, "Image solving is not configured yet. Add the image AI key in server environment settings.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents":[{"parts":[{"text":prompt},{"inline_data":{"mime_type":mime,"data":base64.b64encode(image_bytes).decode("ascii")}}]}],"generationConfig":{"temperature":0.2,"responseMimeType":"application/json"}}
    try:
        async with httpx.AsyncClient(timeout=75) as client: r = await client.post(url, json=payload)
        if r.status_code >= 400:
            log.warning("Image AI request failed status=%s", r.status_code)
            raise HTTPException(503, "KingMath-AI could not process that image right now. Please retry or type the question.")
        data = r.json(); return data["candidates"][0]["content"]["parts"][0]["text"]
    except httpx.TimeoutException: raise HTTPException(504, "Image processing took too long. Please try a clearer or smaller image.")
    except HTTPException: raise
    except Exception:
        log.exception("Image AI request error")
        raise HTTPException(503, "The image could not be processed. Please try again.")

SYSTEM = """You are KingMath-AI, a careful mathematics and science tutor. Never mention AI providers. Be accurate, show useful steps, preserve uncertainty if a diagram/image is unclear, and never invent missing labels or values. Return only valid JSON matching: {\"title\":string,\"understanding\":string,\"principle\":string,\"steps\":[{\"title\":string,\"working\":string,\"explanation\":string}],\"final_answer\":string,\"verification\":string,\"notes\":string}. Use Unicode math notation where clear; avoid raw LaTeX. For a question asking who you are, final_answer should be: I am KingMath-AI, your personal AI mathematics and science tutor. I help you understand concepts, solve problems step by step, practise different topics, and improve your problem-solving skills."""
def extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try: obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match: raise HTTPException(502, "The generated response was not in a usable format. Please retry.")
        try: obj = json.loads(match.group(0))
        except Exception: raise HTTPException(502, "The generated response was incomplete. Please retry.")
    if not isinstance(obj, dict): raise HTTPException(502, "The generated response was not usable. Please retry.")
    return obj

def clean_solution(obj: dict[str, Any]) -> dict[str, Any]:
    steps = obj.get("steps", [])
    if not isinstance(steps, list): steps = []
    normalized=[]
    for i, s in enumerate(steps[:16]):
        if isinstance(s, dict): normalized.append({"title":str(s.get("title",f"Step {i+1}"))[:180],"working":str(s.get("working",""))[:2000],"explanation":str(s.get("explanation",""))[:2000]})
    return {"title":str(obj.get("title","Worked solution"))[:180],"understanding":str(obj.get("understanding",""))[:2500],"principle":str(obj.get("principle",""))[:1500],"steps":normalized,"final_answer":str(obj.get("final_answer","Please retry this question."))[:2500],"verification":str(obj.get("verification",""))[:2000],"notes":str(obj.get("notes",""))[:1500]}

@app.get("/")
def home(): return FileResponse(BASE / "index.html")
@app.get("/health")
def health(): return {"status":"ok","app":"KingMath-AI","text_ai_configured":bool(os.getenv("GROQ_API_KEY")),"image_ai_configured":bool(os.getenv("GEMINI_API_KEY"))}

@app.post("/api/solve")
async def solve(req: SolveRequest):
    q=req.question.strip()
    if not q: raise HTTPException(400,"Enter a question first.")
    messages=[{"role":"system","content":SYSTEM+f"\nAdapt to learner level: {req.level}. Keep the response pedagogical and correct."}]
    for turn in req.conversation[-6:]:
        if turn.get("role") in ("user","assistant") and isinstance(turn.get("content"),str): messages.append({"role":turn["role"],"content":turn["content"][:2000]})
    messages.append({"role":"user","content":q})
    return clean_solution(extract_json(await groq_chat(messages,json_mode=True)))

@app.post("/api/solve-image")
async def solve_image(file: UploadFile=File(...), question: str=Form(default=""), level: str=Form(default="Secondary school")):
    mime=(file.content_type or "").lower()
    if mime not in ALLOWED_MIME: raise HTTPException(415,"Upload a PNG, JPG, or WebP image.")
    raw=await file.read(MAX_IMAGE+1)
    if not raw or len(raw)>MAX_IMAGE: raise HTTPException(413,"Images must be smaller than 7 MB.")
    # Check actual file signatures rather than trusting MIME alone.
    signatures={"image/jpeg":raw.startswith(b"\xff\xd8\xff"),"image/png":raw.startswith(b"\x89PNG\r\n\x1a\n"),"image/webp":raw[:4]==b"RIFF" and raw[8:12]==b"WEBP"}
    if not signatures.get(mime,False): raise HTTPException(415,"The file contents do not match a supported image type.")
    prompt=SYSTEM+f"\nLearner level: {level}. Carefully transcribe and solve the question in the image. If any important text, number, or diagram detail is unreadable, say so explicitly in notes/final_answer and ask for clarification; do not guess. Student's additional note: {question[:1000]}. Return the required JSON structure."
    return clean_solution(extract_json(await gemini_image(raw,mime,prompt)))

@app.post("/api/action")
async def action(req: ActionRequest):
    action_prompts={"clearer":"Explain the existing solution more clearly, keeping the same correct result and breaking difficult reasoning into clearer steps.","simpler":"Explain this in simpler language for a learner. Define unfamiliar terms and split complex steps into smaller ones. Preserve the correct result.","verify":"Independently re-check the original question and solution. Recalculate important steps, identify any errors, and correct them. Do not assume the earlier answer is correct. For simple arithmetic/algebra, show a direct substitution or calculation.","similar":"Create one new question testing the same concept and similar difficulty. Do not solve it. Return JSON {title, understanding, principle, steps:[], final_answer: 'New question: ...', verification:'Try it yourself before asking for a hint.', notes:'similar_question'}"}
    prompt=f"Original question: {req.question}\nExisting solution JSON: {json.dumps(req.solution,ensure_ascii=False)[:10000]}\nTask: {action_prompts[req.action]}\nReturn valid JSON with title, understanding, principle, steps, final_answer, verification, notes. Learner level: {req.level}."
    result=clean_solution(extract_json(await groq_chat([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],json_mode=True)))
    return result

@app.post("/api/topic-question")
async def topic_question(req: TopicRequest):
    prompt=f"Generate one original, clear practice question about {req.topic!r} at {req.difficulty} difficulty. Return JSON with topic, difficulty, question, hint. Do not include the answer or solution."
    obj=extract_json(await groq_chat([{"role":"system","content":"You are KingMath-AI, an expert maths and science educator. Return valid JSON only."},{"role":"user","content":prompt}],json_mode=True,max_tokens=800))
    return {"topic":str(obj.get("topic",req.topic))[:200],"difficulty":req.difficulty,"question":str(obj.get("question",""))[:1500],"hint":str(obj.get("hint",""))[:500]}

@app.post("/api/exam/generate")
async def generate_exam(req: ExamGenerateRequest, request: Request):
    prompt=f"Generate exactly 20 fresh multiple-choice questions for a KingMath-AI CBT practice examination. Topic/subject: {req.topic}. Difficulty: {req.difficulty}. Learner level: {req.level}. Duration: {req.duration_minutes} minutes. Each item must be on-topic, unambiguous, and have exactly four distinct options labelled A, B, C, D, one correct_answer letter, a concise explanation, topic, difficulty, and optional pitfall category selected from Sign Errors, Formula Misapplication, Algebraic Manipulation, Calculation Errors, Conceptual Misunderstanding, Unit Conversion Errors, Graph Interpretation, Scientific Principle Misunderstanding. Return JSON exactly shaped {{\"title\":string,\"subject\":string,\"questions\":[{{\"question\":string,\"options\":{{\"A\":string,\"B\":string,\"C\":string,\"D\":string}},\"correct_answer\":\"A|B|C|D\",\"explanation\":string,\"topic\":string,\"difficulty\":\"Beginner|Intermediate|Advanced\",\"pitfall\":string}}]}}. Make all 20 questions distinct. Ensure the key is mathematically correct. No markdown."
    obj=None
    for attempt in range(2):
        try:
            obj=extract_json(await groq_chat([{"role":"system","content":"You are KingMath-AI, an expert exam setter. Return valid JSON only and never mention providers."},{"role":"user","content":prompt}],json_mode=True,max_tokens=7000))
            qs=obj.get("questions")
            if not isinstance(qs,list) or len(qs)!=20: raise ValueError("question count")
            normalized=[]
            for q in qs:
                if not isinstance(q,dict): raise ValueError("question object")
                options=q.get("options")
                if not isinstance(options,dict) or any(not str(options.get(k,"")).strip() for k in "ABCD"): raise ValueError("options")
                vals=[str(options[k]).strip() for k in "ABCD"]
                if len({v.casefold() for v in vals})<4: raise ValueError("duplicate options")
                key=str(q.get("correct_answer","")).upper()
                if key not in "ABCD" or len(key)!=1: raise ValueError("answer key")
                if not str(q.get("question","")).strip(): raise ValueError("question text")
                normalized.append({"question":str(q["question"])[:2500],"options":{k:vals[i][:1000] for i,k in enumerate("ABCD")},"correct_answer":key,"explanation":str(q.get("explanation","Explanation unavailable."))[:2500],"topic":str(q.get("topic",req.topic))[:200],"difficulty":str(q.get("difficulty",req.difficulty))[:30],"pitfall":str(q.get("pitfall",""))[:100]})
            obj={"title":str(obj.get("title",req.topic+" Practice Exam"))[:200],"subject":str(obj.get("subject",req.topic))[:200],"questions":normalized}
            break
        except HTTPException: raise
        except Exception as e:
            log.warning("Exam validation attempt %s failed: %s",attempt+1,type(e).__name__)
            obj=None
    if not obj: raise HTTPException(502,"KingMath-AI could not generate a valid 20-question examination. Please retry with a more specific topic.")
    exam_id=uuid.uuid4().hex
    EXAMS[exam_id]={"id":exam_id,"owner":request.headers.get("x-session-id", "anonymous")[:100],"title":obj["title"],"subject":obj["subject"],"topic":req.topic,"difficulty":req.difficulty,"duration":req.duration_minutes*60,"questions":obj["questions"],"answers":{},"review":set(),"started_at":None,"deadline":None,"status":"ready","created_at":time.time(),"result":None}
    # Client receives questions only, never the answer key/explanations.
    return {"exam_id":exam_id,"title":obj["title"],"subject":obj["subject"],"topic":req.topic,"difficulty":req.difficulty,"duration_seconds":req.duration_minutes*60,"question_count":20,"questions":[{"question":q["question"],"options":q["options"],"topic":q["topic"],"difficulty":q["difficulty"]} for q in obj["questions"]]}

def get_exam(exam_id: str, request: Request):
    ex=EXAMS.get(exam_id)
    if not ex: raise HTTPException(404,"This examination session has expired or the server restarted. Please generate a new examination.")
    owner=request.headers.get("x-session-id","anonymous")[:100]
    if ex["owner"] != owner: raise HTTPException(403,"This examination belongs to a different session.")
    if ex["status"] == "active" and time.time() >= ex["deadline"]: finalize(ex, timed_out=True)
    return ex

def public_exam(ex):
    return {"exam_id":ex["id"],"title":ex["title"],"subject":ex["subject"],"topic":ex["topic"],"difficulty":ex["difficulty"],"duration_seconds":ex["duration"],"deadline":ex["deadline"],"status":ex["status"],"answers":ex["answers"],"review":[*ex["review"]],"questions":[{"question":q["question"],"options":q["options"],"topic":q["topic"],"difficulty":q["difficulty"]} for q in ex["questions"]]}
@app.post("/api/exam/{exam_id}/start")
async def start_exam(exam_id: str, request: Request):
    ex=get_exam(exam_id,request)
    if ex["status"]!="ready": raise HTTPException(409,"This examination has already started or finished.")
    ex["started_at"]=time.time(); ex["deadline"]=ex["started_at"]+ex["duration"]; ex["status"]="active"
    return public_exam(ex)
@app.get("/api/exam/{exam_id}")
async def exam_state(exam_id: str, request: Request): return public_exam(get_exam(exam_id,request))
@app.post("/api/exam/{exam_id}/answer")
async def save_answer(exam_id: str, req: AnswerRequest, request: Request):
    ex=get_exam(exam_id,request)
    if ex["status"]!="active": raise HTTPException(409,"This examination is no longer accepting answers.")
    if req.answer not in (None,"A","B","C","D"): raise HTTPException(400,"Choose one of the available answer options.")
    if req.answer is None: ex["answers"].pop(str(req.question_index),None)
    else: ex["answers"][str(req.question_index)]=req.answer
    if req.review: ex["review"].add(req.question_index)
    else: ex["review"].discard(req.question_index)
    return {"saved":True,"answered_count":len(ex["answers"]),"remaining_seconds":max(0,int(ex["deadline"]-time.time()))}

def finalize(ex, timed_out=False):
    if ex["status"]=="completed": return ex["result"]
    qs=ex["questions"]; answers=ex["answers"]
    correct=[]; incorrect=[]; unanswered=[]; by_topic={}; pitfalls=defaultdict(list)
    for i,q in enumerate(qs):
        selected=answers.get(str(i)); ok=selected==q["correct_answer"]
        row={"index":i,"question":q["question"],"selected":selected,"correct_answer":q["correct_answer"],"options":q["options"],"explanation":q["explanation"],"topic":q["topic"],"difficulty":q["difficulty"],"correct":ok,"pitfall":q.get("pitfall","")}
        stat=by_topic.setdefault(q["topic"],{"correct":0,"total":0}); stat["total"]+=1
        if ok: correct.append(row); stat["correct"]+=1
        elif selected is None: unanswered.append(row)
        else:
            incorrect.append(row)
            p=q.get("pitfall","")
            if p and p in {"Sign Errors","Formula Misapplication","Algebraic Manipulation","Calculation Errors","Conceptual Misunderstanding","Unit Conversion Errors","Graph Interpretation","Scientific Principle Misunderstanding"}: pitfalls[p].append(row)
    elapsed=max(0,int((ex["deadline"] if timed_out else time.time())-(ex["started_at"] or time.time())))
    # Time used cannot exceed duration; on manual submit it is actual elapsed time.
    if not timed_out: elapsed=max(0,int(time.time()-(ex["started_at"] or time.time())))
    elapsed=min(ex["duration"],elapsed); score=len(correct)
    ex["status"]="completed"; ex["result"]={"exam_id":ex["id"],"title":ex["title"],"subject":ex["subject"],"topic":ex["topic"],"score":score,"total":20,"percentage":round(score/20*100),"correct_count":len(correct),"incorrect_count":len(incorrect),"unanswered_count":len(unanswered),"time_used_seconds":elapsed,"timed_out":timed_out,"by_topic":by_topic,"review":{"correct":correct,"incorrect":incorrect,"unanswered":unanswered,"all":sorted(correct+incorrect+unanswered,key=lambda x:x["index"])},"pitfalls":[{"badge":name,"count":len(rows),"items":[{"index":r["index"],"question":r["question"],"selected":r["selected"],"correct_answer":r["correct_answer"],"options":r["options"],"explanation":r["explanation"],"topic":r["topic"]} for r in rows]} for name,rows in pitfalls.items()]}
    return ex["result"]
@app.post("/api/exam/{exam_id}/submit")
async def submit_exam(exam_id: str, req: SubmitRequest, request: Request):
    ex=get_exam(exam_id,request)
    if ex["status"]=="completed": return ex["result"]
    if ex["status"]!="active": raise HTTPException(409,"Start the examination before submitting it.")
    if time.time()<ex["deadline"] and not req.force: raise HTTPException(409,"The examination time has not expired. Confirm submission to finish early.")
    return finalize(ex,timed_out=time.time()>=ex["deadline"])
@app.get("/api/exam/{exam_id}/results")
async def exam_results(exam_id: str, request: Request):
    ex=get_exam(exam_id,request)
    if ex["status"]!="completed": raise HTTPException(409,"Submit the examination before viewing results.")
    return ex["result"]

@app.post("/api/exam/{exam_id}/analyse")
async def analyse_exam(exam_id: str, request: Request):
    ex=get_exam(exam_id,request)
    if ex["status"]!="completed": raise HTTPException(409,"Complete the examination before analysing answers.")
    wrong=ex["result"]["review"]["incorrect"]
    if not wrong: return {"summary":"Excellent work. There are no incorrect answers to analyse.","insights":[]}
    prompt="Review these actual incorrect responses and identify only evidence-supported learning gaps. Do not assume a misconception from a single ambiguous error. Return JSON {summary:string, insights:[{badge:string, evidence:string, question_number:number, why_incorrect:string, correct_reasoning:string, advice:string, confidence:'possible'|'strong'}]}. Data: "+json.dumps([{ "question_number":r["index"]+1,"question":r["question"],"options":r["options"],"student_answer":r["selected"],"correct_answer":r["correct_answer"],"explanation":r["explanation"],"topic":r["topic"],"pitfall":r["pitfall"]} for r in wrong],ensure_ascii=False)[:12000]
    obj=extract_json(await groq_chat([{"role":"system","content":"You are KingMath-AI, a careful learning analyst. Use cautious language and only cite evidence in the provided responses. Return JSON only."},{"role":"user","content":prompt}],json_mode=True,max_tokens=2500))
    return {"summary":str(obj.get("summary","Review the explanations for the missed questions."))[:1000],"insights":obj.get("insights",[])[:8] if isinstance(obj.get("insights",[]),list) else []}
