from django import forms
from .models import Message, Conversation

class MessageForm(forms.ModelForm):
    """Form for chat messages"""
    class Meta:
        model = Message
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Type your message here...'}),
        }

class ConversationForm(forms.ModelForm):
    """Form for creating/editing conversations"""
    class Meta:
        model = Conversation
        fields = ['title'] 