import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from openai import OpenAI
from app.services.pinecone_utils import index
from app.limiter import limiter
from starlette.requests import Request
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
RATE_LIMIT = 5
TIME_WINDOW = 60  # seconds

client = OpenAI(
    api_key=api_key
) 

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@router.post("/ingest")
@limiter.limit("5/minute")
async def ingest(request: Request, file: UploadFile = File(...)):
    # for text upload api call
    # document_text = request.document_text

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 2 MB.")

    text = content.decode("utf-8")

    # chunk the document
    chunks = chunk_text(text)

    # generate embeddings (semantic vectors) for each chunk
    embeddings = []
    for chunk in chunks:
        embedding_response = client.embeddings.create(
            model = "text-embedding-3-small", #1536
            input = chunk
        )
        embedding = embedding_response.data[0].embedding #1536-d vector
        embeddings.append(embedding)
    
    # upsert to pinecone
    vectors = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"chunk-{idx}"
        vector = {
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk
            }
        }
        vectors.append(vector)
    
    # upsert all vectors to Pinecone index
    index.upsert(vectors=vectors)

    return {
        "filename": file.filename,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "num_embeddings" : len(embeddings),
        "first_vector_id": vectors[0]["id"],
        "first_embedding_dimension": len(embeddings[0]) if embeddings else 0,
        "first_embedding_sample": embeddings[0][:5] if embeddings else [],
        "first_vector_sample_values": vectors[0]["values"][:5],
        "first_vector_metadata": vectors[0]["metadata"] if vectors else {}
        }