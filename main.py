from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Response
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader, PyPDFLoader, SeleniumURLLoader, CSVLoader
from langchain.docstore.document import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from duckduckgo_search import DDGS
from sqlalchemy.orm import Session
import pandas as pd
import uvicorn
import os
import re
import json
import requests
from db import CollectionModel,EmbeddingModel
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv('.env')


class LlamaRequest(BaseModel):
    collection_name: str
    search_kwargs: dict
    query: str
    config: dict
    model_name: str

class UpdateDataRequest(BaseModel):
    user_id: int
    collection_name: str
    new_name: str

# Initialize FastAPI app
app = FastAPI()

# Define SQLAlchemy models
Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    docid = Column(String)
# Define a Pydantic model to receive the data for updating a collection
class UpdateCollection(BaseModel):
    new_name: str
    user_id: int

# Create the database engine and session
DATABASE_URL = os.getenv("DATABASE_URL")

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
            folder_name = "csv"
            # Create the folder if it doesn't exist
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, file.filename)
        elif loader_type == "pdf":
            folder_name = "pdf"
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, file.filename)
        elif loader_type == "text":
            folder_name = "text"
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, file.filename)
        else:
            raise HTTPException(status_code=400, detail="Invalid loader_type")

        # Save the uploaded file to the appropriate folder
        with open(file_path, "wb") as f:
            f.write(uploaded_file_content)

        # Determine the loader based on load
        # er_type
        if loader_type == "csv":
            loader = CSVLoader(file_path=f'csv/{file.filename}', encoding="utf-8")
        elif loader_type == "pdf":
            loader = PyPDFLoader(file_path=f'pdf/{file.filename}')

        elif loader_type == "text":  # For "text" loader, use the uploaded file content
            loader = TextLoader(f'text/{file.filename}', encoding="UTF-8")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            CONNECTION_STRING = (
                DATABASE_URL
            )

            PGVector.from_documents(
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
            os.remove(file_path)
            return {"message": "File uploaded successfully", "user_id": user_id, "docid": collection_name}
        else:
            raise HTTPException(status_code=400, detail="Invalid loader_type")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        CONNECTION_STRING = (
            DATABASE_URL
        )

        PGVector.from_documents(
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
        os.remove(file_path)
        return {"message": "File uploaded successfully", "user_id": user_id, "docid": collection_name}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
#Search the web for results and store them in the Vector Database
@app.post("/upload/search/")
async def upload_url(
    user_id: str = Form(...),
    max_results : int = Form(...),
    query: str = Form(...),  # Accept a single URL directly
):
    try:
        # Replace spaces with underscores and remove or replace invalid characters in the query
        query = re.sub(r'[\/:*?"<>|]', '_', query.replace(" ", "_"))

        # Perform the search and create a DataFrame
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
            data = pd.DataFrame(results)


        # Save the DataFrame to a CSV file with a dynamic filename and specify encoding
        csv_filename = f'{query}.csv'
        data.to_csv(f'csv/{csv_filename}', index=False, encoding='utf-8')

        # Read the CSV file back into a DataFrame and specify encoding
        from langchain.document_loaders.csv_loader import CSVLoader

        loader = CSVLoader(file_path=f'csv/{csv_filename}', encoding='utf-8',csv_args={
            'fieldnames' : ['titles','body','href']
        })


        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        CONNECTION_STRING = (
            DATABASE_URL
        )

        PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=query,  # Use the provided collection_name
            connection_string=CONNECTION_STRING,
        )

        # Insert user_id, collection_name, and docid into the PostgreSQL database
        db = SessionLocal()
        doc = DocumentModel(user_id=user_id, docid=query)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        db.close()
        # Delete the CSV file
        if os.path.exists(f'csv/{csv_filename}'):
            os.remove(f'csv/{csv_filename}')
        # Serialize the DataFrame to JSON
        data_json = data.to_json(orient="records")

        # Return the JSON response
        return Response(content=data_json, media_type="application/json")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#similarity search for results in the database
@app.post("/similarity_search/")
async def similarity_search(
    collection_name: str = Form(...),
    query: str = Form(...),  # Add a parameter for the search query
):
    try:
        CONNECTION_STRING  = os.getenv("DATABASE_URL")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = PGVector(
            collection_name=collection_name,
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
        )
        # Perform similarity search with your vector database
        docs_with_score = db.similarity_search_with_score(query)

        # Return the results as JSON
        results = []
        for doc, score in docs_with_score:
            results.append({"score": score, "page_content": doc.page_content})

        return {"results": results}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/view_docids_by_user_id/")
async def view_docids_by_user_id(user_id: int):
    try:
        # Create a database session
        db = SessionLocal()

        # Query the database for docids by user_id
        docids = db.query(DocumentModel.docid).filter(DocumentModel.user_id == user_id).all()
        docids = [item[0] for item in docids]  # Extract docids from the query result

        # Close the database session
        db.close()

        return {"docids": docids}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Modify the /delete_collection/ endpoint to perform the additional deletions
@app.delete("/delete_collection/")
async def delete_collection(collection_name: str):
    try:
        # Create a database session
        db = SessionLocal()

        # Get the UUID of the collection using the collection_name (docid)
        collection = db.query(CollectionModel).filter(CollectionModel.name == collection_name).first()

        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")

        # Delete records from langchain_pg_embedding associated with the collection
        db.query(EmbeddingModel).filter(EmbeddingModel.collection_id == collection.uuid).delete()

        # Delete the collection from langchain_pg_collection
        db.query(CollectionModel).filter(CollectionModel.uuid == collection.uuid).delete()

        # Delete the row in DocumentModel where collection_name matches docid
        db.query(DocumentModel).filter(DocumentModel.docid == collection_name).delete()

        db.commit()
        db.close()

        return {"message": f"Collection '{collection_name}' and its associated records deleted successfully."}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llama2_rag/")
async def llama2(request_data: LlamaRequest):

    try:
        # Extract data from request_data
        collection_name = request_data.collection_name
        search_kwargs = request_data.search_kwargs
        query = request_data.query
        config = request_data.config
        model_name = request_data.model_name

        from langchain import hub
        QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
        CONNECTION_STRING = os.getenv("DATABASE_URL")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = PGVector(
            collection_name=collection_name,
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
        )
        
        # Create the LLM chain using the provided config and model_name (no need to convert them again)
        llm = CTransformers(model=model_name, callbacks=[StreamingStdOutCallbackHandler()], model_type="llama", config=config)
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs=search_kwargs),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )

        # Perform the query
        result = qa_chain.run(query)
        
        return {"response": result}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True)
