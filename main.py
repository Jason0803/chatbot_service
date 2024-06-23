import streamlit as st
from langchain.schema import ChatMessage, HumanMessage
from src.chatbot import runnable

# Streamlit interface
st.title("Customer Index Chatbot")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat input
question = st.chat_input("Enter your question:")

if question:
    # Store the user question
    st.session_state.chat_history.append(HumanMessage(content=question))
    response = runnable.invoke({"question": question, "history": st.session_state.chat_history})  # Use invoke method
    
    # Assuming the response is a string containing the answer
    answer = response if isinstance(response, str) else "No answer available."
    
    # Store the bot response
    st.session_state.chat_history.append(ChatMessage(role="assistant", content=answer))

# Display chat history
avatars = {"user": "human", "assistant": "ai"}

def show_chat_history():
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, ChatMessage):
            role = "assistant"
        else:
            continue
        st.chat_message(avatars[role]).write(msg.content)

show_chat_history()
