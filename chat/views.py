from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import json

from .models import Conversation, Message
from .forms import MessageForm, ConversationForm
from documents.models import DocumentChunk

def chat_home(request):
    """Home page for the chat interface"""
    conversations = Conversation.objects.all().order_by('-created_at')
    
    # Create a new conversation if none exists
    active_conversation_id = request.session.get('active_conversation_id')
    active_conversation = None
    
    if active_conversation_id:
        try:
            active_conversation = Conversation.objects.get(id=active_conversation_id)
        except Conversation.DoesNotExist:
            # If the conversation was deleted, clear the session
            request.session.pop('active_conversation_id', None)
    
    if not active_conversation:
        # Either no active conversation or it doesn't exist
        if conversations.exists():
            active_conversation = conversations.first()
        else:
            active_conversation = Conversation.objects.create(title="New Conversation")
        
        request.session['active_conversation_id'] = active_conversation.id
    
    # Get messages for the active conversation
    messages = Message.objects.filter(conversation=active_conversation)
    
    context = {
        'conversations': conversations,
        'active_conversation': active_conversation,
        'messages': messages,
        'form': MessageForm(),
    }
    
    return render(request, 'chat/home.html', context)

@require_POST
def ask_question(request):
    """API endpoint to ask a question and get a response"""
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        conversation_id = data.get('conversation_id')
        
        if not question:
            return JsonResponse({'error': 'Question cannot be empty'}, status=400)
        
        # Get or create conversation
        if conversation_id:
            conversation = get_object_or_404(Conversation, id=conversation_id)
        else:
            conversation = Conversation.objects.create(title=question[:50])
            request.session['active_conversation_id'] = conversation.id
        
        # Save user message
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=question
        )
        
        # TODO: Implement actual RAG pipeline and AI response here
        # For now, just return a dummy response
        response = "This is a placeholder response. The actual AI integration will be implemented later."
        
        # Save assistant message
        assistant_message = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=response
        )
        
        return JsonResponse({
            'response': response,
            'conversation_id': conversation.id,
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
