import streamlit as st
from core_function import load_and_prepare_docs, answer_question
import os

# Set page config
st.set_page_config(page_title="RAG-based Q&A Chatbot", page_icon="📄", layout="centered")
st.sidebar.header("API Configuration")

# API key input
user_api_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key",
    type="password",
    help="Your API key is used only for this session. Not stored."
)
if not user_api_key:
    st.warning("Please enter your Google API key to use the chatbot.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = user_api_key

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = None

# UI layout
st.title("RAG-based Q&A Chatbot")
st.markdown("Upload a PDF to start chatting with it. Make sure to enter short and precise prompts for accurate answers.")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if pdf_file and st.session_state.vectorstore is None:
    with st.spinner("Processing document..."):
        st.session_state.vectorstore, st.session_state.bm25, st.session_state.documents = load_and_prepare_docs(pdf_file)
    st.success("PDF processed and ready for Q&A!")

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.vectorstore = None
    st.session_state.bm25 = None
    st.session_state.documents = None
    st.session_state.last_question = None

# Chat interface
if st.session_state.vectorstore:
    question = st.chat_input("Ask a question about the document...")

    if question:
        # Get answer from Gemini using hybrid retrieval
        response = answer_question(
            question,
            st.session_state.vectorstore,
            st.session_state.bm25,
            st.session_state.documents
        )

        # Save to chat history
        st.session_state.chat_history.append((question, response))
        st.session_state.last_question = question

    # Show chat bubbles
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
