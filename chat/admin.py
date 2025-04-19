from django.contrib import admin
from .models import Conversation, Message

class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ('timestamp',)
    fields = ('role', 'content', 'timestamp')
    ordering = ('timestamp',)

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at', 'message_count')
    search_fields = ('title',)
    list_filter = ('created_at',)
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)
    inlines = [MessageInline]
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('get_conversation_title', 'role', 'short_content', 'timestamp')
    list_filter = ('role', 'timestamp')
    search_fields = ('content', 'conversation__title')
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'
    
    def get_conversation_title(self, obj):
        return obj.conversation.title
    get_conversation_title.short_description = 'Conversation'
    
    def short_content(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    short_content.short_description = 'Content'
