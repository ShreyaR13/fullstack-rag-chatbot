import os
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from app.models.schemas import ChatRequest, ChatResponse
from app.services.pinecone_utils import index
from app.limiter import limiter
from starlette.requests import Request
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key
) 

# chat endpoint
@router.post("/chat", response_model=ChatResponse)
@limiter.limit("5/minute")
def chat(request: Request, chat_request: ChatRequest):
    question = chat_request.question

    # step 1: embed users question
    try:
        embedding_response = client.embeddings.create(
            model = "text-embedding-3-small",
            input = question
        )
        question_embedding = embedding_response.data[0].embedding
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    # step 2: query pinecone for similar chunks
    try:
        search_results = index.query(
            vector = question_embedding,
            top_k=3,
            include_metadata = True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")
    
    # step 3: collect retrieved chunks
    retrieved_chunks = []
    for match in search_results.matches:
        if match.metadata and "text" in match.metadata:
            retrieved_chunks.append(match.metadata["text"])
    
    # if no match found, return empty string
    if not retrieved_chunks or all(match.score < 0.75 for match in search_results.matches):
        retrieved_chunks.append("No relevant information found in the document.")

    #### 4️⃣ Prepare context for GPT-4o
    context_text = "\n\n".join(retrieved_chunks)

    # send question + retrieved chunks to GPR 4o
    try:
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. ONLY answer the user's question using the context provided. If the context does not contain relevant information, reply: 'No relevant information found in the document.'"

                },
                {
                    "role": "user", 
                    "content": f"Context: {context_text}\n\nQuestion: {question}"
                }
            ],
            max_tokens = 300
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-4o chat failed: {str(e)}")
    return {"answer": answer}