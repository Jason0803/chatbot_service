import os
import PyPDF2
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from src.retriever.retriever_template import create_retriever

def load_pdf_content(file_path):
    pdf_content = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            pdf_content += page.extract_text()
    return pdf_content

def load_txt_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.pdf'):
            pdf_content = load_pdf_content(file_path)
            docs.append(Document(page_content=pdf_content, metadata={"source": filename}))
        elif filename.endswith('.txt'):
            txt_content = load_txt_content(file_path)
            docs.append(Document(page_content=txt_content, metadata={"source": filename}))
    return docs

def doc_retriever():
    directory = 'utils/documents'  # documents 디렉토리 경로
    docs = load_documents(directory)
    return create_retriever(docs)
