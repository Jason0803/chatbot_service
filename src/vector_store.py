from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(documents, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def save_vector_store(vector_store, path):
    vector_store.save_local(path)

def load_vector_store(path, embeddings):
    return FAISS.load_local(path, embeddings)
