import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader, TextLoader


import os
import pandas as pd

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            continue
        documents.extend(loader.load_and_split())
    return documents

def load_csv_files(directory):
    csv_data = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if file_path.endswith('.csv'):
            csv_data[file_path] = pd.read_csv(file_path)
    return csv_data
