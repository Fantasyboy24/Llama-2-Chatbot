from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader, PyPDFLoader, SeleniumURLLoader, CSVLoader
from langchain.docstore.document import Document
import uvicorn
import os
import requests

# Initialize FastAPI app
app = FastAPI()

# Define SQLAlchemy models
Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    docid = Column(String)

# Create the database engine and session
DATABASE_URL = "postgresql+psycopg2://llama_user:zRn9ZZTS@llama-test-db.cmh2xpy9adng.af-south-1.rds.amazonaws.com:5432/llama_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Define FastAPI endpoint for file upload
@app.post("/upload/")
async def upload_file(
    user_id: str = Form(...),
    collection_name: str = Form(...),
    loader_type: str = Form(...),  # Add a parameter for loader_type
    file: UploadFile = File(...),
):
    try:
        # Read the uploaded file directly
        uploaded_file_content = await file.read()

        # Determine the loader based on loader_type
        if loader_type == "csv":
            loader = CSVLoader(file_path=f'csv/{file.filename}', encoding="utf-8")
        elif loader_type == "pdf":
            loader = PyPDFLoader(file_path=f'pdf/{file.filename}')

        elif loader_type == "text":  # For "text" loader, use the uploaded file content
            loader = TextLoader(f'text/{file.filename}', encoding="UTF-8")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            CONNECTION_STRING = (
                "postgresql+psycopg2://llama_user:zRn9ZZTS@llama-test-db.cmh2xpy9adng.af-south-1.rds.amazonaws.com:5432/llama_db"
            )

            db_vector = PGVector.from_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=collection_name,  # Use the provided collection_name
                connection_string=CONNECTION_STRING,
            )

            # Insert user_id, collection_name, and docid into the PostgreSQL database
            db = SessionLocal()
            doc = DocumentModel(user_id=user_id, docid=collection_name)
            db.add(doc)
            db.commit()
            db.refresh(doc)
            db.close()

            return {"message": "File uploaded successfully", "user_id": user_id, "docid": collection_name}
        else:
            raise HTTPException(status_code=400, detail="Invalid loader_type")

        documents = loader.load()


        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        CONNECTION_STRING = (
            "postgresql+psycopg2://llama_user:zRn9ZZTS@llama-test-db.cmh2xpy9adng.af-south-1.rds.amazonaws.com:5432/llama_db"
        )

        db_vector = PGVector.from_documents(
            embedding=embeddings,
            documents=documents,
            collection_name=collection_name,  # Use the provided collection_name
            connection_string=CONNECTION_STRING,
        )

        # Insert user_id, collection_name, and docid into the PostgreSQL database
        db = SessionLocal()
        doc = DocumentModel(user_id=user_id, docid=collection_name)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        db.close()

        return {"message": "File uploaded successfully", "user_id": user_id, "docid": collection_name}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# Define FastAPI endpoint for URL upload (without file upload)
@app.post("/upload/url/")
async def upload_url(
    user_id: str = Form(...),
    collection_name: str = Form(...),
    url: str = Form(...),  # Accept a single URL directly
):
    try:
        # Fetch content from the URL
        url_content = [f"{url}"]

        # Use the URL content with UnstructuredURLLoader
        loader = SeleniumURLLoader(urls=url_content)

        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        CONNECTION_STRING = (
            "postgresql+psycopg2://llama_user:zRn9ZZTS@llama-test-db.cmh2xpy9adng.af-south-1.rds.amazonaws.com:5432/llama_db"
        )

        db_vector = PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=collection_name,  # Use the provided collection_name
            connection_string=CONNECTION_STRING,
        )

        # Insert user_id, collection_name, and docid into the PostgreSQL database
        db = SessionLocal()
        doc = DocumentModel(user_id=user_id, docid=collection_name)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        db.close()

        return {"message": "URL content uploaded successfully", "user_id": user_id, "docid": collection_name}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
