import os
import time
from typing import List, Dict, Any, Optional, Tuple
import pinecone
from openai import OpenAI
from django.conf import settings
from langchain.schema import Document as LangchainDocument
from documents.models import DocumentChunk
import dotenv
from pathlib import Path

# No need to load .env here since openai_service.py already does it
# Just print debug info
print(f"EmbeddingsService - Using OPENAI_API_KEY from environment: {os.environ.get('OPENAI_API_KEY')[:20]}...")

class EmbeddingsService:
    """Service for handling text embeddings using OpenAI and Pinecone"""
    
    def __init__(self):
        # Get API keys directly from environment
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
        
        # Debug logging - will show in console during development
        print(f"Initializing EmbeddingsService with:")
        print(f"Embedding model: text-embedding-3-small")
        print(f"Pinecone environment: {self.pinecone_environment}")
        print(f"Pinecone API key present: {bool(self.pinecone_api_key)}")
        print(f"OpenAI API key: {self.openai_api_key[:20]}...")
        
        # Initialize OpenAI client with explicit API key parameter
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        print("OpenAI client created in EmbeddingsService")
        
        self.embedding_model = "text-embedding-3-small"  # Changed from 3-large to 3-small
        self.embedding_dimensions = 1536  # text-embedding-3-small has 1536 dimensions
        
        # Initialize Pinecone with new API
        self.pinecone = pinecone.Pinecone(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment
        )
        
        # Index name for document chunks
        self.index_name = "faster-chat-docs"
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
    
    def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists, create if it doesn't"""
        try:
            # List indexes
            indexes = [index.name for index in self.pinecone.list_indexes()]
            
            if self.index_name not in indexes:
                # Create a new index with dimensions for text-embedding-3-large
                # Updated to use the current Pinecone API which requires a 'spec' parameter
                from pinecone import ServerlessSpec
                
                print(f"Creating new Pinecone index: {self.index_name} with {self.embedding_dimensions} dimensions")
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait for index to initialize
                time.sleep(5)  # Increased wait time for index initialization
                print(f"Created index: {self.index_name}")
        except Exception as e:
            print(f"Error ensuring index exists: {str(e)}")
            raise
    
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
        import time
        import datetime
        
        search_start = time.time()
        print(f"⚡ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Starting similarity search for query")
        
        # Create embedding for the query
        embed_start = time.time()
        query_embedding = self.create_embedding(query)
        embed_time = time.time() - embed_start
        print(f"⚡ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Created embedding in {embed_time:.2f}s")
        
        # Search in Pinecone
        pinecone_start = time.time()
        index = self.get_index()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        pinecone_time = time.time() - pinecone_start
        print(f"⚡ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Pinecone query completed in {pinecone_time:.2f}s, returned {len(results.matches)} matches")
        
        # Get document chunks
        db_start = time.time()
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
        db_time = time.time() - db_start
        
        search_time = time.time() - search_start
        print(f"⚡ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Similarity search completed in {search_time:.2f}s - Embedding: {embed_time:.2f}s, Pinecone: {pinecone_time:.2f}s, DB: {db_time:.2f}s")
        print(f"⚡ Retrieved {len(chunks_with_scores)} chunks with scores")
        
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
        import time
        import datetime
        
        context_start = time.time()
        print(f"⚡ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Getting relevant context for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        # Get similar chunks
        chunks_with_scores = self.similarity_search(query, top_k=max_chunks)
        
        if not chunks_with_scores:
            print(f"⚡ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - No relevant chunks found")
            return ""
        
        # Sort by score (highest first) and extract content
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build context with document references
        format_start = time.time()
        context_parts = []
        for chunk, score in chunks_with_scores:
            document_title = chunk.document.title
            context_parts.append(f"--- From document: {document_title} ---\n{chunk.content}\n")
        
        context = "\n".join(context_parts)
        format_time = time.time() - format_start
        
        total_time = time.time() - context_start
        print(f"⚡ [{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] - Context retrieval completed in {total_time:.2f}s - Formatting: {format_time:.2f}s")
        print(f"⚡ Retrieved context length: {len(context)} characters from {len(chunks_with_scores)} chunks")
        
        return context 