from django.db import models
from django.utils import timezone
import os

def document_upload_path(instance, filename):
    """Generate a unique path for uploaded documents"""
    # Get the file extension
    ext = filename.split('.')[-1]
    # Create filename with timestamp
    timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
    return f'documents/{timestamp}_{filename}'

class Document(models.Model):
    """Model for uploaded documents"""
    PROCESSING_STATUS = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to=document_upload_path)
    file_type = models.CharField(max_length=20)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=PROCESSING_STATUS, default='pending')
    error_message = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.title
    
    @property
    def file_extension(self):
        return os.path.splitext(self.file.name)[1].lower()
    
    def save(self, *args, **kwargs):
        # Set file type based on file extension
        if not self.id:
            extension = os.path.splitext(self.file.name)[1].lower()
            if extension == '.pdf':
                self.file_type = 'pdf'
            elif extension in ['.doc', '.docx']:
                self.file_type = 'word'
            elif extension in ['.txt', '.md']:
                self.file_type = 'text'
            elif extension in ['.jpg', '.jpeg', '.png']:
                self.file_type = 'image'
            else:
                self.file_type = 'other'
        
        super().save(*args, **kwargs)

class DocumentChunk(models.Model):
    """Model for document chunks after processing"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField()
    chunk_number = models.IntegerField()
    embedding_id = models.CharField(max_length=255, blank=True, null=True)
    
    class Meta:
        ordering = ['chunk_number']
        unique_together = ['document', 'chunk_number']
    
    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_number}"
