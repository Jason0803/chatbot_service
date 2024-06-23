from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from utils.config import load_config

def create_retriever(docs, top_k=3, chunk_size=300, chunk_overlap=150):
    config = load_config()
    embeddings = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEY'])
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": top_k})
