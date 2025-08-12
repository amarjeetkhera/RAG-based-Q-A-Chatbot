import streamlit as st
from core_function import load_and_prepare_docs, answer_question
import os

# Set page config
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")

# Load Gemini API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Session state to keep track of chat history and vectorstore
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# UI layout
st.title("📄 Ask Questions About Your PDF")
st.markdown("Upload a PDF and start chatting with it!")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if pdf_file and st.session_state.vectorstore is None:
    with st.spinner("Processing document..."):
        st.session_state.vectorstore = load_and_prepare_docs(pdf_file)
    st.success("PDF processed and ready for Q&A!")

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# Chat interface
if st.session_state.vectorstore:
    question = st.chat_input("Ask a question about the document...")

    if question:
        # Get answer from Gemini
        response = answer_question(question, st.session_state.vectorstore)

        # Save to chat history
        st.session_state.chat_history.append((question, response))

    # Show chat bubbles
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
