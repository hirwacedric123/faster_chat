from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from django.conf import settings
from django.utils import timezone

from .models import Document, DocumentChunk
from .forms import DocumentUploadForm

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
            document = form.save()
            
            # TODO: Implement document processing in a background task
            # For now, we'll just mark it as completed
            document.status = 'completed'
            document.processed_at = timezone.now()
            document.save()
            
            messages.success(request, f"Document '{document.title}' uploaded successfully!")
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
        document_title = document.title
        document.file.delete(save=False)  # Delete the file from storage
        document.delete()  # Delete the database entry
        
        messages.success(request, f"Document '{document_title}' deleted successfully!")
        return redirect('documents:list')
    
    # If it's a GET request, show confirmation page
    context = {
        'document': document,
    }
    
    return render(request, 'documents/delete_confirm.html', context)
