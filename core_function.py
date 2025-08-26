import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
import google.generativeai as genai
import tempfile
from typing import List

# Initialize Gemini
def init_gemini():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)

# Load and split PDF into chunks
def load_and_prepare_docs(pdf_file) -> VectorStore:
    init_gemini()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # Load PDF using LangChain loader
    loader = PDFPlumberLoader(tmp_path)
    pages = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    documents = splitter.split_documents(pages)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Answer a question using Gemini + context from vectorstore
def answer_question(question: str, vectorstore: VectorStore) -> str:
    init_gemini()

    # Retrieve relevant context
    relevant_docs: List[Document] = vectorstore.similarity_search(question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Use the context below to answer the question. 
If the answer isn't in the context, check if the question is still relevant to the context. If it is, answer the question from your knowledge in brief.
If the question is irrelevant to the document and the answer is not in the context, say "I couldn't find that in the document."

Context:
{context_text}

Question: {question}
Answer:"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text.strip()
