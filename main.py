import os
import uuid
import time
import traceback
import requests
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from dotenv import load_dotenv
from transformers import pipeline

# -------------------------------
# ENV SETUP
# -------------------------------
load_dotenv()
VLLM_API_URL = os.getenv("VLLM_API_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
MASTER_API_KEY = os.getenv("MASTER_API_KEY", "local-key")
DATABASE_URL = os.getenv("DATABASE_URL")

# -------------------------------
# DATABASE SETUP
# -------------------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    owner = Column(String(50), nullable=False)
    usage_count = Column(Integer, default=0)
    usage_limit = Column(Integer, default=100)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class QueryLog(Base):
    __tablename__ = "query_logs"
    id = Column(Integer, primary_key=True)
    api_key = Column(String(100), nullable=False)
    user_query = Column(Text, nullable=False)
    model_response = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

Base.metadata.create_all(bind=engine)

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI(title="vLLM Local FastAPI Server")

# -------------------------------
# LOAD LOCAL MODEL
# -------------------------------
print("🚀 Loading GPT-2 model locally (CPU fallback)...")
local_generator = pipeline("text-generation", model="gpt2", device=-1)
print("✅ Model ready on CPU")

# -------------------------------
# DB DEPENDENCY
# -------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# QUERY FUNCTION
# -------------------------------
def query_vllm(prompt: str) -> str:
    """Query vLLM backend or fallback to GPT2 locally."""
    if VLLM_API_URL:
        try:
            resp = requests.post(VLLM_API_URL, json={"prompt": prompt, "model": MODEL_NAME}, timeout=10)
            if resp.ok:
                data = resp.json()
                if "choices" in data and data["choices"]:
                    c = data["choices"][0]
                    return c.get("text") or c.get("message", {}).get("content", "")
        except Exception as e:
            print("[⚠️ vLLM fallback]:", e)
    try:
        result = local_generator(prompt, max_length=80, num_return_sequences=1)
        return result[0]["generated_text"]
    except Exception as e:
        print("[⚠️ GPT2 ERROR]:", e)
        return "Error generating text."

# -------------------------------
# Pydantic Schemas
# -------------------------------
class AskRequest(BaseModel):
    model: str | None = "gpt2"
    messages: list

class APIKeyRequest(BaseModel):
    owner: str

# -------------------------------
# API ROUTES
# -------------------------------
@app.post("/api/generate-key")
def generate_key(req: APIKeyRequest, db: Session = Depends(get_db)):
    key = str(uuid.uuid4())
    entry = APIKey(key=key, owner=req.owner)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return {"message": "✅ API key generated", "api_key": entry.key}

@app.post("/v1/chat/completions")
async def chat_completions(request: AskRequest, db: Session = Depends(get_db), x_api_key: str = Header(None)):
    # 🔑 Validate key
    key = db.query(APIKey).filter(APIKey.key == x_api_key).first()
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if key.usage_count >= key.usage_limit:
        raise HTTPException(status_code=403, detail="API key usage limit reached")

    # 🧠 Prepare prompt
    messages = request.messages or []
    if not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' list")
    prompt = "\n".join([m["content"] for m in messages if m["role"] == "user"])

    # 💬 Generate text
    answer = query_vllm(prompt)

    # 🗃️ Log
    log = QueryLog(api_key=x_api_key, user_query=prompt, model_response=answer)
    db.add(log)
    key.usage_count += 1
    db.commit()

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(prompt.split()) + len(answer.split())
        },
        "usage_status": f"{key.usage_count}/{key.usage_limit} used"
    }

# -------------------------------
# GLOBAL ERROR CATCHER
# -------------------------------
@app.exception_handler(Exception)
async def catch_all_exceptions(request: Request, exc: Exception):
    print("🔥 SERVER ERROR:\n", traceback.format_exc())
    return {"detail": str(exc)}

# -------------------------------
@app.get("/")
def root():
    return {"message": "✅ vLLM FastAPI server is live. Use /api/generate-key then /v1/chat/completions"}
