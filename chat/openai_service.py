from typing import List, Dict, Any, Optional
import os
from django.conf import settings
from openai import OpenAI
from .embeddings_service import EmbeddingsService
import dotenv
from pathlib import Path

# Load environment variables directly
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
print(f"Loading .env file from: {dotenv_path} (exists: {dotenv_path.exists()})")
dotenv.load_dotenv(dotenv_path, override=True)  # Force override existing env vars

# Print the system OpenAI API key for debugging
system_api_key = os.environ.get('OPENAI_API_KEY', 'Not set')
print(f"System OPENAI_API_KEY: {system_api_key[:20]}... (length: {len(system_api_key) if system_api_key else 0})")

# Now reload from .env and override the system variable
if dotenv_path.exists():
    # Load the .env file content directly
    with open(dotenv_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                if key == 'OPENAI_API_KEY':
                    print(f"Explicitly setting OPENAI_API_KEY from .env file to: {value[:20]}...")
                    os.environ['OPENAI_API_KEY'] = value

class OpenAIService:
    """Service for interacting with OpenAI API"""
    
    def __init__(self):
        # Check what's in environment variables after our override
        print("DEBUG - Environment variables after override:")
        print(f"OPENAI_API_KEY in os.environ: {'OPENAI_API_KEY' in os.environ}")
        print(f"OPENAI_API_KEY from os.environ: {os.environ.get('OPENAI_API_KEY')[:20]}...")
        
        # Get API key directly from .env file
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Debug logging
        print(f"OpenAIService - Final API key format: {self.api_key[:20]}... (length: {len(self.api_key) if self.api_key else 0})")
        
        # Create client with explicit API key
        print(f"Creating OpenAI client with API key starting with: {self.api_key[:20]}...")
        self.client = OpenAI(api_key=self.api_key)
        print("OpenAI client created.")
        
        # Print client base URL to verify configuration
        print(f"OpenAI client base URL: {self.client.base_url}")
        
        self.embeddings_service = EmbeddingsService()
        self.model = "gpt-3.5-turbo"
    
    def generate_response(self, messages: List[Dict[str, str]], 
                          query: str = "", temperature: float = 0.7, 
                          max_tokens: int = 500) -> str:
        """
        Generate a response using OpenAI API
        """
        try:
            # Create a copy of messages to avoid modifying the original
            messages_copy = messages.copy()
            
            # If the query is not empty, augment the system message with document context
            if query:
                # Get relevant context from documents
                context = self.embeddings_service.get_relevant_context(query)
                
                # If context exists, add it to the system message
                if context:
                    system_message = {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant that answers questions based on provided documents. "
                            "Use the following information from documents to answer the question, and cite the source document. "
                            "If the information is not in the documents, say that you don't have information on this topic in "
                            "your documents and provide a general answer. Here are the relevant document sections:\n\n"
                            f"{context}"
                        )
                    }
                    
                    # Find and replace system message or insert at the beginning
                    system_index = next((i for i, msg in enumerate(messages_copy) if msg["role"] == "system"), None)
                    if system_index is not None:
                        messages_copy[system_index] = system_message
                    else:
                        messages_copy.insert(0, system_message)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_copy,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract and return the response content
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Handle errors
            error_message = f"Error generating response: {str(e)}"
            return error_message
    
    def is_answer_in_documents(self, query: str) -> bool:
        """Check if the answer to a query can be found in the documents"""
        # Get relevant context
        context = self.embeddings_service.get_relevant_context(query, max_chunks=2)
        
        # If no context, answer is not in documents
        if not context:
            return False
        
        # Ask OpenAI if the context answers the query
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI evaluator. Your job is to determine whether the provided context "
                    "contains enough information to answer the given question. Respond with 'YES' "
                    "if the context contains information to answer the question at least partially, "
                    "or 'NO' if the context does not contain relevant information to answer the question."
                )
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context}\n\nDoes the context contain information to answer the question? Answer with YES or NO."
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,  # Use low temperature for more deterministic response
            max_tokens=5
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer 