from langchain.document_loaders import WebBaseLoader
from src.retriever.retriever_template import create_retriever

def load_web_content(urls):
    loader = WebBaseLoader(web_paths=urls)
    docs = loader.load()
    return docs

def recsys_retriever():
    urls = [
        "https://mellerikat.com/ko/user_guide/data_scientist_guide/ai_contents/tcr/",
        "https://mellerikat.com/ko/user_guide/data_scientist_guide/ai_contents/gcr/"
    ]
    docs = load_web_content(urls)
    return create_retriever(docs)
