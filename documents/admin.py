from django.contrib import admin
from .models import Document, DocumentChunk

class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    fields = ('chunk_number', 'short_content', 'embedding_id')
    readonly_fields = ('short_content',)
    ordering = ('chunk_number',)
    
    def short_content(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    short_content.short_description = 'Content'

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'file_type', 'status', 'uploaded_at', 'chunk_count')
    list_filter = ('file_type', 'status', 'uploaded_at')
    search_fields = ('title', 'description')
    readonly_fields = ('uploaded_at', 'processed_at', 'file_type')
    date_hierarchy = 'uploaded_at'
    inlines = [DocumentChunkInline]
    
    def chunk_count(self, obj):
        return obj.chunks.count()
    chunk_count.short_description = 'Chunks'

@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ('get_document_title', 'chunk_number', 'short_content', 'has_embedding')
    list_filter = ('document__file_type', 'document__status')
    search_fields = ('content', 'document__title')
    ordering = ('document', 'chunk_number')
    
    def get_document_title(self, obj):
        return obj.document.title
    get_document_title.short_description = 'Document'
    
    def short_content(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    short_content.short_description = 'Content'
    
    def has_embedding(self, obj):
        return bool(obj.embedding_id)
    has_embedding.boolean = True
    has_embedding.short_description = 'Has Embedding'
