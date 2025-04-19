from django import forms
from .models import Document

class DocumentUploadForm(forms.ModelForm):
    """Form for uploading documents"""
    class Meta:
        model = Document
        fields = ['title', 'description', 'file']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Optional description...'}),
        }
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            # Check file extension
            ext = file.name.split('.')[-1].lower()
            valid_extensions = ['pdf', 'doc', 'docx', 'txt', 'md']
            if ext not in valid_extensions:
                raise forms.ValidationError(
                    f"Unsupported file format. Please upload a file in one of these formats: {', '.join(valid_extensions)}"
                )
            
            # Check file size (10MB limit)
            if file.size > 10 * 1024 * 1024:  # 10MB in bytes
                raise forms.ValidationError("File size must be under 10MB.")
                
        return file 