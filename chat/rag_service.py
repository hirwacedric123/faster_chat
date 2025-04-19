from typing import List, Dict, Any, Optional, Tuple
from django.conf import settings
import time
import datetime

from .openai_service import OpenAIService
from .embeddings_service import EmbeddingsService
from .models import Conversation, Message

# Global service cache
_openai_service = None
_embeddings_service = None

class RAGService:
    """Retrieval-Augmented Generation service that combines document search with AI generation"""
    
    def __init__(self):
        global _openai_service, _embeddings_service
        
        # Initialize services with caching
        if _openai_service is None:
            _openai_service = OpenAIService()
        self.openai_service = _openai_service
        
        if _embeddings_service is None:
            _embeddings_service = EmbeddingsService()
        self.embeddings_service = _embeddings_service
    
    def ask(self, conversation: Conversation, query: str) -> Tuple[str, bool]:
        """
        Process a query using RAG:
        1. Check if answer is in documents
        2. If not, use general AI response
        
        Returns the response and a flag indicating if docs were used
        """
        # Get messages from the conversation (for context)
        messages = self._get_conversation_messages(conversation)
        
        # Check if the answer is in the documents
        start_time = time.time()
        has_document_answer = self.openai_service.is_answer_in_documents(query)
        doc_check_time = time.time() - start_time
        if doc_check_time > 1.0:  # Only log if slow
            print(f"Document check took {doc_check_time:.2f}s - Result: {has_document_answer}")
        
        # Generate response
        start_time = time.time()
        response = self.openai_service.generate_response(
            messages=messages,
            query=query if has_document_answer else "",  # Only pass query for context if we found relevant docs
            temperature=0.5 if has_document_answer else 0.7,  # Lower temperature for document-based answers
            max_tokens=800  # Increased from 500 to 800 for more comprehensive answers
        )
        gen_time = time.time() - start_time
        if gen_time > 2.0:  # Only log if slow
            print(f"Response generation took {gen_time:.2f}s")
        
        return response, has_document_answer
    
    def _get_conversation_messages(self, conversation: Conversation) -> List[Dict[str, str]]:
        """Get conversation messages in OpenAI format"""
        # Get all messages from the conversation
        db_messages = Message.objects.filter(conversation=conversation).order_by('timestamp')
        
        # Convert to OpenAI format
        openai_messages = []
        
        # First, add a system message if it doesn't exist
        if not any(msg.role == 'system' for msg in db_messages):
            openai_messages.append({
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that provides accurate and concise information. "
                    "When using information from documents, always cite your sources. "
                    "Be helpful, harmless, and honest in your responses."
                )
            })
        
        # Then add the rest of the messages
        for msg in db_messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return openai_messages 