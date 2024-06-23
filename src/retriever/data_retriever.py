import os
import pandas as pd
from langchain.schema import Document
from src.retriever.retriever_template import create_retriever

def load_csv_content(file_path):
    return pd.read_csv(file_path).to_string()

def load_parquet_content(file_path):
    return pd.read_parquet(file_path).to_string()

def load_data_files(directory):
    docs = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.csv'):
            csv_content = load_csv_content(file_path)
            docs.append(Document(page_content=csv_content, metadata={"source": filename}))
        elif filename.endswith('.parquet'):
            parquet_content = load_parquet_content(file_path)
            docs.append(Document(page_content=parquet_content, metadata={"source": filename}))
    return docs

def data_retriever():
    directory = 'utils/data'  # data 디렉토리 경로
    docs = load_data_files(directory)
    return create_retriever(docs)
