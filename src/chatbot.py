from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import ChatMessage, HumanMessage, SystemMessage, Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnableLambda
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from src.data_loader import load_documents, load_csv_files
from src.vector_store import create_vector_store, save_vector_store, load_vector_store
from src.query_handlers import answer_from_doc_files, recommend_csv_files, recommend_preprocessing_and_features, answer_from_web
from utils.config import load_config

def initialize_chatbot():
    # Load configuration
    config = load_config()

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEY'])

    # Load and prepare data
    document_directory = 'utils/documents'
    docs = load_documents(document_directory)  # Load documents here
    vector_store = create_vector_store(docs, embeddings)
    save_vector_store(vector_store, 'faiss_index')

    csv_directory = 'utils/data'
    csv_data = load_csv_files(csv_directory)
    csv_info = {file: data.columns.tolist() for file, data in csv_data.items()}

    # Define LLM
    llm = OpenAI(temperature=0.5, openai_api_key=config['OPENAI_API_KEY'])

    # Define system message for the assistant
    system_message = SystemMessage(
        content=(
            "너는 기본적으로 'my플랫폼' 에 들어온 사용자의 질문에 대답하는 상담사야. "
            "매너있는 답변을 유지해. "
            "모르는 것은 모른다고 하고 chltndud308@gmail.com 으로 연락하라는 응답을 해."
        )
    )

    # Define prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{question}")
    ])

    # Define runnable sequences
    doc_search_sequence = RunnableSequence(RunnableLambda(lambda inputs: {"text": answer_from_doc_files(inputs["question"])}), StrOutputParser())
    recommend_content_sequence = RunnableSequence(chat_prompt, llm, StrOutputParser())
    execute_request_sequence = RunnableSequence(chat_prompt, llm, StrOutputParser())
    data_recommend_sequence = RunnableSequence(RunnableLambda(lambda inputs: {"text": recommend_csv_files(inputs["question"])}), StrOutputParser())
    general_qa_sequence = RunnableSequence(chat_prompt, llm, StrOutputParser())
    
    # Web-based retriever sequence
    web_retriever_sequence = RunnableSequence(RunnableLambda(lambda inputs: {"text": answer_from_web(inputs["question"])}), StrOutputParser())

    # Setup RunnableBranch
    def setup_branch():
        branch = RunnableBranch(
            (lambda x: "고객지수란" in x["question"].lower(), doc_search_sequence),
            (lambda x: "고객지수 개발 방법은" in x["question"].lower(), doc_search_sequence),
            (lambda x: "데이터 사용할 수 있을까" in x["question"].lower(), recommend_content_sequence),
            (lambda x: "전처리 하고" in x["question"].lower(), execute_request_sequence),
            (lambda x: "~에 적합한 데이터 추천해 줘" in x["question"].lower(), data_recommend_sequence),
            (lambda x: "추천 시스템" in x["question"].lower(), web_retriever_sequence),
            general_qa_sequence  # Set the default chain directly here
        )
        return branch

    return setup_branch()

runnable = initialize_chatbot()
