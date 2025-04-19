import os
import time
from typing import List, Dict, Any, Optional, Tuple
import pinecone
from openai import OpenAI
from django.conf import settings
from langchain.schema import Document as LangchainDocument
from documents.models import DocumentChunk
from pathlib import Path
from .env_utils import load_environment
import json
import numpy as np
import pandas as pd

# Global cached clients
_openai_client = None
_pinecone_client = None
_pinecone_index = None
_vector_store = None
_documents = None
_embedding_cache = {}

class EmbeddingsService:
    """Service for handling text embeddings using OpenAI and Pinecone"""
    
    def __init__(self):
        global _openai_client, _vector_store, _documents
        
        # Get API keys from environment utils
        self.api_key, self.pinecone_api_key, self.pinecone_environment, self.embeddings_file, self.documents_file = load_environment()
        
        # Create client with explicit API key if not already created
        if _openai_client is None:
            _openai_client = OpenAI(api_key=self.api_key)
        
        self.client = _openai_client
        
        # Set embedding model and dimensions
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536  # text-embedding-3-small has 1536 dimensions
        
        # Load vector store if not already loaded
        if _vector_store is None:
            self._load_vector_store()
        else:
            self.vector_store = _vector_store
        
        # Load documents if not already loaded
        if _documents is None:
            self._load_documents()
        else:
            self.documents = _documents
        
        # Initialize Pinecone with new API
        global _pinecone_client
        if _pinecone_client is None:
            _pinecone_client = pinecone.Pinecone(
                api_key=self.pinecone_api_key,
                environment=self.pinecone_environment
            )
        self.pinecone = _pinecone_client
        
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
                
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pinecone.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait for index to initialize
                time.sleep(5)
        except Exception as e:
            print(f"Error ensuring index exists: {str(e)}")
            raise
    
    def get_index(self):
        """Get the Pinecone index"""
        global _pinecone_index
        if _pinecone_index is None:
            _pinecone_index = self.pinecone.Index(self.index_name)
        return _pinecone_index
    
    def _load_vector_store(self):
        """Load the vector store from disk"""
        try:
            start_time = time.time()
            
            # Load the embeddings file
            embeddings_path = Path(self.embeddings_file)
            if not embeddings_path.exists():
                print(f"Embeddings file {embeddings_path} does not exist")
                global _vector_store
                _vector_store = {}
                self.vector_store = {}
                return
                
            # Load the embeddings as a numpy array
            with open(embeddings_path, 'r') as f:
                data = json.load(f)
                
            # Convert to numpy array for faster similarity calculations
            _vector_store = {
                chunk_id: np.array(embedding) for chunk_id, embedding in data.items()
            }
            self.vector_store = _vector_store
            
            load_time = time.time() - start_time
            if load_time > 0.5:  # Only log if slow
                print(f"Vector store loaded in {load_time:.2f}s ({len(self.vector_store)} chunks)")
                
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            _vector_store = {}
            self.vector_store = {}
    
    def _load_documents(self):
        """Load the documents from disk"""
        try:
            start_time = time.time()
            
            # Load the documents file
            documents_path = Path(self.documents_file)
            if not documents_path.exists():
                print(f"Documents file {documents_path} does not exist")
                global _documents
                _documents = {}
                self.documents = {}
                return
                
            # Load the documents
            with open(documents_path, 'r') as f:
                _documents = json.load(f)
            
            self.documents = _documents
            
            load_time = time.time() - start_time
            if load_time > 0.5:  # Only log if slow
                print(f"Documents loaded in {load_time:.2f}s ({len(self.documents)} chunks)")
                
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            _documents = {}
            self.documents = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get an embedding for a text"""
        global _embedding_cache
        
        # Check if embedding is in cache
        if text in _embedding_cache:
            return _embedding_cache[text]
        
        try:
            # Get embedding from OpenAI
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            
            # Convert to numpy array
            embedding = np.array(response.data[0].embedding)
            
            # Cache the embedding (limit cache size to avoid memory issues)
            if len(_embedding_cache) > 1000:
                # Remove a random key if cache is too large
                _embedding_cache.pop(next(iter(_embedding_cache)))
            _embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return np.zeros(self.embedding_dimensions)
    
    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding vector for a text using OpenAI"""
        start_time = time.time()
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        embedding_time = time.time() - start_time
        if embedding_time > 0.5:  # Only log if slow
            print(f"Embedding creation took {embedding_time:.2f}s")
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
        query_embedding = self._get_embedding(query)
        
        # Search in Pinecone
        start_time = time.time()
        index = self.get_index()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        pinecone_time = time.time() - start_time
        if pinecone_time > 1.0:  # Only log if slow
            print(f"Pinecone query took {pinecone_time:.2f}s")
        
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
    
    def get_relevant_context(self, query: str, max_chunks: int = 3, similarity_threshold: float = 0.7) -> str:
        """
        Get context relevant to a query from documents
        """
        if not self.vector_store or not self.documents:
            return ""
        
        try:
            start_time = time.time()
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Calculate similarity with all chunks
            similarities = {}
            for chunk_id, chunk_embedding in self.vector_store.items():
                # Calculate cosine similarity (dot product of normalized vectors)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities[chunk_id] = similarity
            
            # Sort by similarity and get top chunks
            top_chunks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:max_chunks]
            
            # Filter by similarity threshold
            top_chunks = [(chunk_id, sim) for chunk_id, sim in top_chunks if sim >= similarity_threshold]
            
            # Format context
            context = ""
            for chunk_id, similarity in top_chunks:
                if chunk_id in self.documents:
                    doc = self.documents[chunk_id]
                    context += f"Document: {doc.get('source', 'Unknown')}\n"
                    context += f"Content: {doc.get('content', '')}\n\n"
            
            search_time = time.time() - start_time
            if search_time > 0.5 and context:  # Only log if slow and context was found
                print(f"Found {len(top_chunks)} relevant chunks in {search_time:.2f}s")
            
            return context
            
        except Exception as e:
            print(f"Error getting relevant context: {str(e)}")
            return "" 