from src.retriever.doc_retriever import doc_retriever
from src.retriever.data_retriever import data_retriever
from src.retriever.web_retriever import recsys_retriever

def answer_from_doc_files(query):
    retriever = doc_retriever()
    documents = retriever.retrieve(query)
    return "\n".join([doc.page_content for doc in documents])

def recommend_csv_files(query):
    retriever = data_retriever()
    documents = retriever.retrieve(query)
    return "\n".join([doc.page_content for doc in documents])

def recommend_preprocessing_and_features(csv_file, csv_data):
    if csv_file not in csv_data:
        return "CSV file not found."

    data = csv_data[csv_file]
    preprocessing_steps = f"Preprocessing steps for {csv_file}:\n - Remove null values\n - Normalize data\n"
    features = f"Features from {csv_file}:\n - {', '.join(data.columns)}"
    return preprocessing_steps + features

def answer_from_web(query):
    retriever = recsys_retriever()
    documents = retriever.retrieve(query)
    return "\n".join([doc.page_content for doc in documents])
