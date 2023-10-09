from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document


loader = TextLoader("state_of_the_union.txt",encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

CONNECTION_STRING = "postgresql+psycopg2://llama_user:zRn9ZZTS@llama-test-db.cmh2xpy9adng.af-south-1.rds.amazonaws.com:5432/llama_db"

# The PGVector Module will try to create a table with the name of the collection.
# So, make sure that the collection name is unique and the user has the permission to create a table.

COLLECTION_NAME = "state_of_the_union_test"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)
