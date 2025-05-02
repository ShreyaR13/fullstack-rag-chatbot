from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later, we'll restrict this to your frontend URL.
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

# chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # for now just echo question
    # return {"answer": f"You asked: {request.question}"}
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant."
            },
            {
                "role": "user", 
                "content": request.question
            }
        ],
        max_tokens = 200
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer}