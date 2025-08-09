import os
import shutil
import requests
import json

# LangChain imports for RAG components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration Constants ---
# Directory to store ChromaDB persistent data (vector store)
CHROMA_DB_DIR = "chroma_db"
# Embedding model
EMBEDDING_MODEL = "gemini-embedding-001"
# Generative model
GENERATION_MODEL = "gemini-2.0-flash"

# Utility Function for vector store initialization
def clear_chroma_db_data():
    """
    Removes the ChromaDB directory to ensure a fresh start
    or when uploading a new PDF document.
    """
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print(f"ChromaDB data cleared from '{CHROMA_DB_DIR}'.")

# Core RAG Processing Function
def process_pdf_and_create_vectorstore(uploaded_file_buffer, api_key):
    """
    Processes an uploaded PDF file:
    1. Extracts text
    2. Splits text into manageable chunks
    3. Generates numerical embeddings for each chunk
    4. Stores the embeddings in a ChromaDB vector store for retrieval.

    Returns:
        The initialized ChromaDB vector store instance or None if an error occurs.
    """
    if uploaded_file_buffer is None:
        print("No PDF file provided for precessing.")
        return None
    
    # Saving the file to a temporary path
    temp_file_path = "temp_uploaded_document.pdf"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file_buffer.read())

        # 1. Loading the document
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from the PDF.")

        # 2. Splitting the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            print("No text could be extracted for chunks creation from the PDF.")
            return None
        print(f"Document split into {len(chunks)} chunks.")

        # 3. Creating embeddings for the text chunks
        os.environ["GOOGLE_API_KEY"] = api_key
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, task_type='retrieval_document')
        print(f"Embedding model initialized.")

        # 4. Initializing the ChromaDB vector store
        clear_chroma_db_data() # Clear previous data if any

        vectorstore = Chroma.from_documents(
            documents = chunks,
            embedding = embeddings,
            persist_directory = CHROMA_DB_DIR # Saving the vector to disk
        )
        vectorstore.persist() # Ensure data is written to disk
        print(f"ChromaDB created and persisted with {len(chunks)} document chunks.")
        return vectorstore
    
    except Exception as e:
        print(f"Error during PDF precessing or vector store creation: {e}")
        return None
    finally:
        # Clean up temporary PDF file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file '{temp_file_path}' removed.")

# Response Generation Function
def get_rag_response(user_query, vectorstore_instance, api_key):
    """
    Generates a response using the RAG (Retrieval Augmented Generation) approach:
    1. Retrieves relevant document chunks from the vector store based on the user's query.
    2. Constructs a prompt for the LLM using the retrieved context.
    3. Calls the Gemini API to get the final answer.

    Returns:
        The generated answer from the LLM, or an error message.
    """
    if not api_key:
        return "Error: API key is not configured."
    if vectorstore_instance is None:
        return "Error: No document loaded or knowledge base available."

    # API URL for the generation model
    gemini_api_url =  f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATION_MODEL}:generateContent?key={api_key}"

    try:
        # 1. Retrieve relevant document chunks from the vector store
        # LangChain's similarity_search handles embedding the user_query internally
        retrieved_docs = vectorstore_instance.similarity_search(user_query, k=4) #Retrieve top 4 relevant chunks
        print(f"Retrieved {len(retrieved_docs)} relevant document chunks for the query.")
        
        # Combine the content of the retrieved documents to form the context for the LLM
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 2. Construct the prompt for the LLM, including the retrieved context
        prompt_template = f"""
        You are an internal assistant for a company. Your task is to answer questions based *only* on the provided document context.
        If the answer is not explicitly present in the document context, state that you cannot find the information in the provided context.
        Do not use external knowledge.
        
        Document Context:
        {context}
        
        User's Question:
        {user_query}
        """
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt_template}]}],
        }

        headers = {
            "Content-Type": "application/json"
        }

        # 3. Make the API call to Gemini
        response = requests.post(gemini_api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404, 500)
        result = response.json()

        # Extract the model's response
        if result.get("candidates") and len(result["candidates"]) > 0 and \
        result["candidates"][0].get("content") and \
        result["candidates"][0]["content"].get("parts") and \
        len(result["candidates"][0]["content"]["parts"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Could not get a valid response from the AI model. Please try again."

    except requests.exceptions.RequestException as e:
        # Handle network-related or API-specific errors
        print(f"Network related or API error call: {e}")
        return f"Error: Could not connect to the AI service. Details: {e}"
    except json.JSONDecodeError:
        # Handle cases where the API response is not valid JSON
        print("Failed to decode JSON response from API.")
        return "Error: Invalid response format from AI service."
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred during RAG response generation: {e}")
        return f"An unexpected error occurred: {e}"