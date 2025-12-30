import os
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
from rank_bm25 import BM25Okapi
from collections import defaultdict

# Initialize Gemini
def init_gemini():
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

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
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    documents = splitter.split_documents(pages)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    # Prepare BM25
    tokenized_docs = [doc.page_content.split() for doc in documents] #simple white space tokenizer
    bm25 = BM25Okapi(tokenized_docs)
    return vectorstore, bm25, documents

# Hybrid Retrieval using combination of FAISS & BM25
def hybrid_retrieval(query: str, vectorstore: FAISS, bm25: BM25Okapi, documents: List[Document], k=5):
    # FAISS Semantic Search
    semantic_docs = vectorstore.similarity_search(query, k=k)

    # BM25 exact keyword search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = bm25_scores.argsort()[-k:][::-1] #top-k indices
    bm25_docs = [documents[i] for i in top_bm25_indices]

    # Merge to avoid duplicates by page_content
    seen = set()
    merged_docs = []
    for doc in semantic_docs + bm25_docs:
        if doc.page_content not in seen:
            merged_docs.append(doc)
            seen.add(doc.page_content)
        if len(merged_docs) >= k:
            break
    return merged_docs

# Answer a question using Gemini + context from vectorstore
def answer_question(question: str, vectorstore: FAISS, bm25: BM25Okapi, documents: List[Document]) -> str:
    init_gemini()

    # Retrieve relevant context
    relevant_docs = hybrid_retrieval(question, vectorstore, bm25, documents, k=5)
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
