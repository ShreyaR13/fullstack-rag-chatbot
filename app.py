from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

app = FastAPI()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

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

pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
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

# chat endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question

    # step 1: embed users question
    embedding_response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = question
    )
    question_embedding = embedding_response.data[0].embedding

    # step 2: query pinecone for similar chunks
    search_results = index.query(
        vector = question_embedding,
        top_k=2,
        include_metadata = True
    )

    # step 3: collect retrieved chunks
    retrieved_chunks = []
    for match in search_results.matches:
        if match.metadata and "text" in match.metadata:
            retrieved_chunks.append(match.metadata["text"])
    
    # if no match found, return empty string
    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."

    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Use the provided context to answer the question."
            },
            {
                "role": "user", 
                "content": f"Context: {context}\n\nQuestion: {question}"
            }
        ],
        max_tokens = 200
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer}

# Function to get or create index
def get_or_create_index(index_name, dimension):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)

INDEX_NAME = "quickstart" 
DIMENSION = 1536  # For OpenAI text-embedding-3-small

index = get_or_create_index(INDEX_NAME, DIMENSION)

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@app.post("/ingest")
def ingest(request: IngestRequest):
    document_text = request.document_text

    # chunk the document
    chunks = chunk_text(document_text)

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
        "num_chunks": len(chunks),
        "chunks": chunks,
        "num_embeddings" : len(embeddings),
        "first_vector_id": vectors[0]["id"],
        "first_embedding_dimension": len(embeddings[0]) if embeddings else 0,
        "first_embedding_sample": embeddings[0][:5] if embeddings else [],
        "first_vector_sample_values": vectors[0]["values"][:5],
        "first_vector_metadata": vectors[0]["metadata"]
        }