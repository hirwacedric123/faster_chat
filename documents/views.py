from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from django.conf import settings
from django.utils import timezone
from django.db import transaction

from .models import Document, DocumentChunk
from .forms import DocumentUploadForm
from .document_processor import DocumentProcessor
from chat.embeddings_service import EmbeddingsService

def document_home(request):
    """Home page for document management"""
    form = DocumentUploadForm()
    documents = Document.objects.all().order_by('-uploaded_at')
    
    context = {
        'form': form,
        'documents': documents,
    }
    
    return render(request, 'documents/home.html', context)

def document_list(request):
    """View for listing all documents"""
    documents = Document.objects.all().order_by('-uploaded_at')
    
    context = {
        'documents': documents,
    }
    
    return render(request, 'documents/list.html', context)

def upload_document(request):
    """Handle document upload and processing"""
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            with transaction.atomic():
                # Save document with pending status
                document = form.save(commit=False)
                document.status = 'processing'
                document.save()
                
                try:
                    # Process document (extract text and create chunks)
                    processor = DocumentProcessor(document)
                    success = processor.process()
                    
                    if success:
                        # Create embeddings for each chunk
                        embeddings_service = EmbeddingsService()
                        chunks = DocumentChunk.objects.filter(document=document)
                        
                        for chunk in chunks:
                            embeddings_service.store_document_chunk(chunk)
                        
                        # Mark document as completed
                        document.status = 'completed'
                        document.processed_at = timezone.now()
                        document.save()
                        
                        messages.success(request, f"Document '{document.title}' uploaded and processed successfully!")
                    else:
                        # If process() returned False but didn't raise an exception
                        messages.error(request, f"Error processing document: {document.error_message or 'Unknown error'}")
                
                except Exception as e:
                    # Log the error and mark document as failed
                    document.status = 'failed'
                    document.error_message = str(e)
                    document.save()
                    messages.error(request, f"Error processing document: {str(e)}")
            
            return redirect('documents:list')
        else:
            messages.error(request, "Error uploading document. Please check the form.")
    else:
        form = DocumentUploadForm()
    
    context = {
        'form': form,
    }
    
    return render(request, 'documents/upload.html', context)

def delete_document(request, doc_id):
    """Delete a document"""
    document = get_object_or_404(Document, id=doc_id)
    
    if request.method == 'POST':
        try:
            # Delete embeddings from Pinecone
            embeddings_service = EmbeddingsService()
            embeddings_service.delete_document_embeddings(document.id)
            
            # Get document title before deletion
            document_title = document.title
            
            # Delete file and database entry
            document.file.delete(save=False)  # Delete the file from storage
            document.delete()  # Delete the database entry
            
            messages.success(request, f"Document '{document_title}' deleted successfully!")
        except Exception as e:
            messages.error(request, f"Error deleting document: {str(e)}")
        
        return redirect('documents:list')
    
    # If it's a GET request, show confirmation page
    context = {
        'document': document,
    }
    
    return render(request, 'documents/delete_confirm.html', context)
