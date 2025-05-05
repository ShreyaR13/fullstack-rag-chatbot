import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

INDEX_NAME = "quickstart" 
DIMENSION = 1536  # For OpenAI text-embedding-3-small

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

index = get_or_create_index(INDEX_NAME, DIMENSION)