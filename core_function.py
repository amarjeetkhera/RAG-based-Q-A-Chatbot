import os
import tempfile
from collections import defaultdict
from typing import List, Tuple

from google import genai
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi


class GeminiGenAIEmbeddings(Embeddings):
    """Embeddings wrapper using the official google-genai SDK with batching."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.models.embed_content(
                model=self.model,
                contents=batch,
            )
            all_embeddings.extend([emb.values for emb in result.embeddings])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return result.embeddings[0].values


def load_and_prepare_docs(
    pdf_file, api_key: str
) -> Tuple[FAISS, BM25Okapi, List[Document]]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    try:
        loader = PDFPlumberLoader(tmp_path)
        pages = loader.load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    documents = splitter.split_documents(pages)

    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. Please upload a text-based PDF."
        )

    embeddings = GeminiGenAIEmbeddings(
        api_key=api_key,
        model="gemini-embedding-001",
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    tokenized_docs = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return vectorstore, bm25, documents


def hybrid_retrieval(
    query: str,
    vectorstore: FAISS,
    bm25: BM25Okapi,
    documents: List[Document],
    k: int = 5,
) -> List[Document]:
    candidate_k = k * 2

    semantic_docs = vectorstore.similarity_search(query, k=candidate_k)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = bm25_scores.argsort()[-candidate_k:][::-1]
    bm25_docs = [documents[i] for i in top_bm25_indices]

    rrf_scores = defaultdict(float)
    c = 60
    doc_map = {}

    for rank, doc in enumerate(semantic_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] += 1.0 / (c + (rank + 1))

    for rank, doc in enumerate(bm25_docs):
        content = doc.page_content
        doc_map[content] = doc
        rrf_scores[content] += 1.0 / (c + (rank + 1))

    sorted_contents = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_map[content] for content, _ in sorted_contents[:k]]


def answer_question(
    question: str,
    vectorstore: FAISS,
    bm25: BM25Okapi,
    documents: List[Document],
    api_key: str,
) -> str:
    relevant_docs = hybrid_retrieval(question, vectorstore, bm25, documents, k=5)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Use the context below to answer the question. 
If the answer isn't in the context, check if the question is still relevant to the context.
If the question is irrelevant to the document context and the answer is not in the context, say "I couldn't find that in the document."

Context:
{context_text}

Question: {question}
Answer:"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    return response.text.strip()
