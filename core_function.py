import os
import requests
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# --- Constants ---
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.0-flash"


# --- PDF to FAISS Vectorstore ---
def process_pdf_and_create_vectorstore(uploaded_file, api_key):
    """
    Processes an uploaded PDF and returns a FAISS vectorstore using Gemini embeddings.
    
    Args:
        uploaded_file: PDF uploaded via Streamlit
        api_key: Gemini API key

    Returns:
        FAISS vectorstore or None if processing fails
    """
    if uploaded_file is None:
        print("[ERROR] No file uploaded.")
        return None

    temp_file_path = "temp_uploaded_document.pdf"

    try:
        # Save PDF temporarily
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 1. Load PDF content
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        print(f"[INFO] Loaded {len(documents)} pages.")

        # 2. Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        if not chunks:
            print("[ERROR] No content to embed.")
            return None
        print(f"[INFO] Split into {len(chunks)} chunks.")

        # 3. Embed chunks using Gemini
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            task_type="retrieval_document",
            google_api_key=api_key
        )
        print("[INFO] Embeddings initialized.")

        # 4. Store embeddings in FAISS vector store (in-memory)
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        print("[INFO] FAISS vectorstore created.")

        return vectorstore

    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {e}")
        return None

    finally:
        # Clean up the vector store
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[INFO] Removed temp file '{temp_file_path}'.")


# --- RAG Response Function ---
def get_rag_response(user_query, vectorstore_instance, api_key):
    """
    Retrieves relevant chunks using FAISS and generates a Gemini response.
    """
    if not api_key:
        return "❌ Error: Missing API key."

    if vectorstore_instance is None:
        return "❌ Error: No vectorstore available."

    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATION_MODEL}:generateContent?key={api_key}"

    try:
        # Retrieve similar chunks
        retrieved_docs = vectorstore_instance.similarity_search(user_query, k=4)
        print(f"[INFO] Retrieved {len(retrieved_docs)} chunks.")

        # Combine context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""
You are a helpful assistant. Use only the following context to answer the question.
If the answer is not in the context, say: "The document does not contain that information."

Context:
{context}

Question:
{user_query}
"""

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(gemini_api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        candidates = result.get("candidates", [])
        if candidates and candidates[0].get("content", {}).get("parts", []):
            return candidates[0]["content"]["parts"][0]["text"]
        else:
            return "⚠️ Gemini returned no valid response."

    except requests.exceptions.RequestException as e:
        return f"❌ API Error: {e}"

    except json.JSONDecodeError:
        return "❌ Response parsing failed: Invalid JSON."

    except Exception as e:
        return f"❌ Unexpected error: {e}"
