# Faster Chat - AI-Powered Chatbot with Document Integration

Faster Chat is an intelligent chatbot built with Django that can answer questions based on provided documents. It uses Retrieval-Augmented Generation (RAG) to search for relevant information within uploaded materials before calling the AI model.

## Key Features

- **Document Processing**: Upload PDF, Word, and text documents to teach the AI about your specific content
- **Intelligent Responses**: The system first tries to find answers in your documents before calling the AI model
- **Cost-Effective**: By using existing documents, the system can respond to many queries without needing to call the API
- **Vector Search**: Uses Pinecone for semantic search and retrieval of document content

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd faster_chat
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following content:

```
# Django settings
DEBUG=True
SECRET_KEY=your-secret-key

# API Keys
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment

# Media settings
MEDIA_ROOT=media/
STATIC_ROOT=static/
```

Replace the placeholders with your actual API keys.

### 5. Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Create a Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 7. Run the Development Server

```bash
python manage.py runserver
```

Visit http://127.0.0.1:8000/ in your browser to access the app.

## Project Structure

- `/chat`: App for handling conversations and AI interactions
- `/documents`: App for managing document uploads and processing
- `/templates`: Contains the base templates for the application
- `/media`: Storage for uploaded documents
- `/static`: Static files (CSS, JS, images)

## Usage

1. **Upload Documents**: Go to the Documents section and upload files
2. **Ask Questions**: Use the Chat section to ask questions
3. **Get Answers**: The system will search your documents for answers or use AI if needed

## Technology Stack

- Django: Web framework
- OpenAI: GPT-3.5 Turbo API for generating responses
- Pinecone: Vector database for document embeddings
- LangChain: Utilities for structuring the RAG pipeline
- PyPDF/python-docx: Document parsing libraries 