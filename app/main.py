import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
# from cachetools import TTLCache
from app.routes import chat
from app.routes import ingest
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

app = FastAPI()
app.include_router(chat.router)
app.include_router(ingest.router)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# rate_limit_cache = TTLCache(maxsize=1000, ttl=TIME_WINDOW)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later, restrict this to frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=api_key
) 

# health check route
@app.get("/health")
def read_health():
    return {"status": "ok"}

# request schema for chat
class ChatRequest(BaseModel):
    question: str

# response schema
class ChatResponse(BaseModel):
    answer: str

class IngestRequest(BaseModel):
    document_text: str

# def rate_limiter(request: Request):
#     client_ip = request.client.host
#     count = rate_limit_cache.get(client_ip, 0)

#     if count >= RATE_LIMIT:
#         raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

#     rate_limit_cache[client_ip] = count + 1