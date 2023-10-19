from dotenv import load_dotenv
load_dotenv('.env')
import os
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import SentenceTransformerEmbeddings

query = "To say within thine own deep sunken eyes, Were an all-eating shame, and thriftless praise"
DATABASE_URL = os.getenv("DATABASE_URL")

COLLECTION_NAME = "shakespeare sonnets 1"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = PGVector(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=DATABASE_URL
)

docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)