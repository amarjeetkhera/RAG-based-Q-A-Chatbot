import os
import sys
import streamlit as st
from core_function import clear_chroma_db_data, process_pdf_and_create_vectorstore, get_rag_response

try:
  API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API Key not found.")
    st.stop()
