import os
import time
from typing import List, Dict, Any, Optional, Tuple
import pinecone
from openai import OpenAI
from django.conf import settings
from langchain.schema import Document as LangchainDocument
from documents.models import DocumentChunk

class EmbeddingsService:
    """Service for handling text embeddings using OpenAI and Pinecone"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-large"  # Changed from ada-002 to 3-large
        self.embedding_dimensions = 3072  # text-embedding-3-large has 3072 dimensions (vs 1536 for ada-002)
        
        # Initialize Pinecone with new API
        self.pinecone = pinecone.Pinecone(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
        
        # Index name for document chunks
        self.index_name = "faster-chat-docs"
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
    
    def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists, create if it doesn't"""
        # List indexes
        indexes = [index.name for index in self.pinecone.list_indexes()]
        
        if self.index_name not in indexes:
            # Create a new index with dimensions for text-embedding-3-large
            self.pinecone.create_index(
                name=self.index_name,
                dimension=self.embedding_dimensions,
                metric="cosine"
            )
            # Wait for index to initialize
            time.sleep(1)
    
    def get_index(self):
        """Get the Pinecone index"""
        return self.pinecone.Index(self.index_name)
    
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding vector for a text using OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def store_document_chunk(self, chunk: DocumentChunk) -> str:
        """Store a document chunk in Pinecone and update the chunk with embedding ID"""
        # Generate a unique ID for the chunk
        chunk_id = f"doc_{chunk.document.id}_chunk_{chunk.chunk_number}"
        
        # Create embedding for the chunk text
        embedding = self.create_embedding(chunk.content)
        
        # Store in Pinecone
        index = self.get_index()
        index.upsert(
            vectors=[{
                'id': chunk_id,
                'values': embedding,
                'metadata': {
                    "document_id": str(chunk.document.id),
                    "chunk_number": chunk.chunk_number,
                    "document_title": chunk.document.title,
                    "document_type": chunk.document.file_type
                }
            }]
        )
        
        # Update the chunk with embedding ID
        chunk.embedding_id = chunk_id
        chunk.save(update_fields=['embedding_id'])
        
        return chunk_id
    
    def similarity_search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar document chunks based on query"""
        # Create embedding for the query
        query_embedding = self.create_embedding(query)
        
        # Search in Pinecone
        index = self.get_index()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Get document chunks
        chunks_with_scores = []
        for match in results.matches:
            try:
                document_id = int(match.metadata["document_id"])
                chunk_number = match.metadata["chunk_number"]
                
                chunk = DocumentChunk.objects.get(
                    document_id=document_id,
                    chunk_number=chunk_number
                )
                
                chunks_with_scores.append((chunk, match.score))
            except (DocumentChunk.DoesNotExist, KeyError, ValueError) as e:
                continue
        
        return chunks_with_scores
    
    def delete_document_embeddings(self, document_id: int) -> None:
        """Delete all embeddings for a document"""
        # Get chunks for the document
        chunks = DocumentChunk.objects.filter(document_id=document_id)
        
        # Get embedding IDs
        embedding_ids = [chunk.embedding_id for chunk in chunks if chunk.embedding_id]
        
        if embedding_ids:
            # Delete from Pinecone
            index = self.get_index()
            index.delete(ids=embedding_ids)
    
    def get_relevant_context(self, query: str, max_chunks: int = 3) -> str:
        """Get the most relevant context from documents for a query"""
        chunks_with_scores = self.similarity_search(query, top_k=max_chunks)
        
        if not chunks_with_scores:
            return ""
        
        # Sort by score (highest first) and extract content
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build context with document references
        context_parts = []
        for chunk, score in chunks_with_scores:
            document_title = chunk.document.title
            context_parts.append(f"--- From document: {document_title} ---\n{chunk.content}\n")
        
        return "\n".join(context_parts) 