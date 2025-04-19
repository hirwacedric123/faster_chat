from typing import List, Dict, Any, Optional, Tuple
from django.conf import settings

from .openai_service import OpenAIService
from .embeddings_service import EmbeddingsService
from .models import Conversation, Message

class RAGService:
    """Retrieval-Augmented Generation service that combines document search with AI generation"""
    
    def __init__(self):
        self.openai_service = OpenAIService()
        self.embeddings_service = EmbeddingsService()
    
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
        has_document_answer = self.openai_service.is_answer_in_documents(query)
        
        # Generate response
        response = self.openai_service.generate_response(
            messages=messages,
            query=query if has_document_answer else "",  # Only pass query for context if we found relevant docs
            temperature=0.7 if has_document_answer else 0.9,  # Lower temperature for document-based answers
            max_tokens=500
        )
        
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
                "content": "You are a helpful AI assistant that provides accurate and concise information."
            })
        
        # Then add the rest of the messages
        for msg in db_messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return openai_messages 