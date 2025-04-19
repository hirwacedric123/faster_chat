from typing import List, Dict, Any, Optional, Tuple
from django.conf import settings
import time
import datetime

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
        # Start timing
        start_time = time.time()
        start_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"⌛ RAG START [{start_timestamp}] - Processing query")
        
        # Get messages from the conversation (for context)
        msg_start = time.time()
        messages = self._get_conversation_messages(conversation)
        msg_time = time.time() - msg_start
        print(f"⌛ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Got {len(messages)} messages in {msg_time:.2f}s")
        
        # Check if the answer is in the documents
        doc_check_start = time.time()
        print(f"⌛ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Checking if answer is in documents")
        has_document_answer = self.openai_service.is_answer_in_documents(query)
        doc_check_time = time.time() - doc_check_start
        print(f"⌛ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Document check completed in {doc_check_time:.2f}s - Result: {has_document_answer}")
        
        # Generate response
        gen_start = time.time()
        print(f"⌛ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Generating response with {'document context' if has_document_answer else 'general knowledge'}")
        response = self.openai_service.generate_response(
            messages=messages,
            query=query if has_document_answer else "",  # Only pass query for context if we found relevant docs
            temperature=0.5 if has_document_answer else 0.7,  # Lower temperature for document-based answers
            max_tokens=800  # Increased from 500 to 800 for more comprehensive answers
        )
        gen_time = time.time() - gen_start
        print(f"⌛ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Response generation completed in {gen_time:.2f}s")
        
        # End timing
        total_time = time.time() - start_time
        end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"⌛ RAG END [{end_timestamp}] - Total processing time: {total_time:.2f}s")
        print(f"⌛ TIMING SUMMARY: Total: {total_time:.2f}s | Doc Check: {doc_check_time:.2f}s | Generation: {gen_time:.2f}s | Used docs: {has_document_answer}")
        
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