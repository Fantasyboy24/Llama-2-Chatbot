from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import uuid
from dotenv import load_dotenv
import os
load_dotenv('.env')

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    docid = Column(String)

class CollectionModel(Base):
    __tablename__ = "langchain_pg_collection"
    uuid = Column(String, primary_key=True)
    name = Column(String)
    cmetadata = Column(String)
    

class EmbeddingModel(Base):
    __tablename__ = "langchain_pg_embedding"
    uuid = Column(Integer, primary_key=True)
    collection_id = Column(String, ForeignKey("langchain_pg_collection.uuid"))
    embedding = Column(String)
    document = Column(String)
    cmetadata = Column(String)
    custom_id = Column(String)

if __name__ == "__main__":
    # Create the tables in the database
    Base.metadata.create_all(bind=engine)
