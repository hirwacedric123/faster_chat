import os
import dotenv
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

@lru_cache(maxsize=1)
def load_environment():
    """
    Load environment variables from .env file
    Returns the OpenAI API key, Pinecone API key, and Pinecone environment
    """
    # Load .env file if it exists
    load_dotenv()
    
    # Get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    
    # Get paths for embeddings and documents
    embeddings_file = os.getenv("EMBEDDINGS_FILE", str(Path(__file__).parent.parent / "data" / "embeddings.json"))
    documents_file = os.getenv("DOCUMENTS_FILE", str(Path(__file__).parent.parent / "data" / "documents.json"))
    
    return openai_api_key, pinecone_api_key, pinecone_environment, embeddings_file, documents_file 