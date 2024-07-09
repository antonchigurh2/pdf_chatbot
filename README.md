# PDF Chatbot using FastAPI

This application allows users to upload a PDF file, chat about its contents, and get responses using a Question Answering (QA) model. It leverages FastAPI for the backend, Sentence Transformers for text embeddings, and PyPDF2 for PDF content extraction.

## Installation
1. Clone the repository
   git clone https://github.com/antonchigurh2/pdf_chatbot.git 
   cd pdf-chatbot  
2. Install dependencies  
   pip install -r requirements.txt  
## USAGE
 1. Start the FastAPI server  
    uvicorn main:app --reload  
 2. Open your web browser and go to http://127.0.0.1:8000/  
 3. Upload a PDF file and start querying its content.   
