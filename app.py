import os
import sys
import streamlit as st
from core_function import clear_chroma_db_data, process_pdf_and_create_vectorstore, get_rag_response

# --- Configuration ---
# Get the Gemini API key from Streamlit secrets
try:
  API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API Key not found.")
    st.stop()

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "model", "parts": [{"text": "Hello! Upload a PDF document to start asking questions."}]})

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.document_uploaded = False

# --- Callback Functions for UI Interactions ---

def handle_pdf_upload_callback():
    """Callback function for PDF file uploader."""
    uploaded_file = st.session_file_uploader
    if uploaded_file:
        # Clear previous data before processing new PDF
        clear_chroma_db_data() # This clears the persisted ChromaDB data
        st.session_state.vectorstore = None
        st.session_state.document_uploaded = False
        st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "model", "parts": [{"text": "Processing new PDF..."}]})
        
        # Process the PDF and create vectorstore using the function from rag_core
        with st.spinner("Processing PDF and building knowledge base..."):
            st.session_state.vectorstore = process_pdf_and_create_vectorstore(uploaded_file, API_KEY)
            if st.session_state.vectorstore:
                st.session_state.document_uploaded = True
                st.session_state.chat_history.append({"role": "model", "parts": [{"text": f"PDF '{uploaded_file.name}' processed successfully! You can now ask questions."}]})
            else:
                st.session_state.document_uploaded = False
                st.session_state.chat_history.append({"role": "model", "parts": [{"text": "Failed to process PDF. Please try another file."}]})
    else:
        st.session_state.document_uploaded = False
        st.session_state.chat_history.append({"role": "model", "parts": [{"text": "No PDF uploaded. Please upload a document."}]})

def clear_all_data_callback():
    """Clears all session state data and ChromaDB."""
    clear_chroma_db_data() # Clear persisted ChromaDB data
    st.session_state.vectorstore = None
    st.session_state.document_uploaded = False
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "model", "parts": [{"text": "All data and chat history cleared. Please upload a new PDF."}]})
    # Reset the file uploader widget
    st.session_state.pdf_uploader = None
    st.rerun() # Rerun to update the UI

# --- Main App Logic ---
st.set_page_config(layout="wide", page_title="Internal AI Chatbot")

st.title("⚡ AI Chatbot for Document Q&A")
st.markdown("---")

col1, col2 = st.columns([1, 2]) # Document input on left, Chat on right

with col1:
    st.subheader("1. Upload Your PDF Document")
    
    # PDF File Uploader
    st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        key="pdf_uploader",
        on_change=handle_pdf_upload_callback, # Callback when file changes
        help="Upload a PDF document. Its text will be extracted, chunked, and embedded into a vector database for Q&A."
    )
    
    # Button to clear current document and ChromaDB
    st.button("Clear Document & Chat", on_click=clear_all_data_callback)

    if st.session_state.document_uploaded:
        st.success("Knowledge base ready. You can now chat!")
    else:
        st.info("Please upload a PDF to begin.")


with col2:
    st.subheader("2. Chat with the AI")
    
    # Display chat history
    chat_container = st.container(height=450, border=True)
    for message in st.session_state.chat_history:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["parts"][0]["text"])

    # Chat input
    user_query = st.chat_input("Type your message here...", disabled=not st.session_state.document_uploaded)

    if user_query:
        if not st.session_state.document_uploaded:
            st.warning("Please upload a PDF document and wait for it to process before asking questions.")
            st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_query}]})
            st.rerun()
        else:
            st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_query}]})
            
            with chat_container.chat_message("model"):
                with st.spinner("AI is thinking..."):
                    # Call the RAG function from rag_core
                    response_text = get_rag_response(user_query, st.session_state.vectorstore, API_KEY)
            
            st.session_state.chat_history.append({"role": "model", "parts": [{"text": response_text}]})
            st.rerun()
