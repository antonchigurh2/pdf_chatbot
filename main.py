from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os
app = FastAPI()

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory chat history storage
chat_history = []

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_pdf(
    request: Request,
    pdf_file: UploadFile = File(None),
    query: str = Form(...)
):
    if pdf_file is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF file before submitting a query.")
    
    pdf_content = await pdf_file.read()
    pdf_name=pdf_file.filename
    pdf_name=pdf_name[:-4]
    # Using PyPDF2 to read PDF contents
    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Splitting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    
    #converting the chunks to its embeddings using sentence-transformer
    if os.path.exists(f"{pdf_name}.pkl"):
        with open(f"{pdf_name}.pkl", "rb") as f:
                embeddings = pickle.load(f)
    else:
        embeddings = model.encode(chunks, show_progress_bar=True)
        with open(f"{pdf_name}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

    if query:
        # Logging the user query
        chat_history.append({"role": "user", "message": query})
        
        # Process the bot response asynchronously

        #finding the embedding of user query
        query_embedding = model.encode(query)
        query_embedding_2d = np.reshape(query_embedding, (1, -1))

        #finding similarity of user_query with all the embeddings
        cosine_similarities = cosine_similarity(query_embedding_2d, embeddings).flatten()

        # Considering the top 3 scoring embeddings
        top_3_indices = np.argsort(cosine_similarities)[-3:][::-1]
        top_3_chunks = [chunks[i] for i in top_3_indices]
        combined_context = " ".join(top_3_chunks)

        #using the top3 chunks to send as a context to the model
        input = {"context": combined_context, "question": query}
        response = qa_pipeline(input)
        combined_response = response["answer"]
        # Loging the bot response
        chat_history.append({"role": "bot", "message": combined_response})
    
    return {"chat_history": chat_history, "new_message": combined_response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
