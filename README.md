# RAG-based Q&A Chatbot

## Overview
This project presents an intuitive Retrieval-Augmented Generation (RAG) based chat assistant designed to allow users to ask questions directly from their PDF documents.

The system uses a hybrid retrieval pipeline combining semantic search (FAISS) and keyword-based search (BM25) to improve both contextual understanding and exact keyword matching. This enables more robust and reliable document-grounded answers, especially for technical, numerical and definition-heavy queries.

## Description
The RAG Chatbot is implemented using a Python backend and Streamlit frontend, making it deployable as a user-friendly web application. It uses Google’s Gemini 2.0 Flash LLM for answer generation and a hybrid retrieval strategy that combines:

- FAISS vector search for semantic similarity

- BM25 keyword search for exact term matching

Retrieved results from both methods are merged and passed as grounded context to the language model, reducing hallucinations and improving recall for keyword-sensitive questions.

At a high level, the tool follows an architecture as shown in the diagram below:

<img width="1920" height="1080" alt="Q A Chatbot_20250903_211203_0000" src="https://github.com/user-attachments/assets/40d4990c-ecce-4ff9-ae47-bc7c3e024f55" />


(Kindly cite this repository if you use this flow chart)

## RAG vs Traditional Chatbots
Traditional chatbots often rely on static rule-based logic or general-purpose LLMs with no access to specific documents. In contrast, RAG bridges the gap by providing the model with retrieved context directly from your uploaded files.

This enhances:

- Accuracy: Reduces hallucinations and generic answers
- Explainability: Responses are traceable to original document content
- Relevance: Answers stay grounded to the PDF uploaded by the user

## Hybrid Retrieval: FAISS + BM25

While dense vector search excels at capturing semantic meaning, it may miss exact keyword matches, numerical references, or domain-specific terminology. To address this limitation, the chatbot uses a hybrid retrieval approach:

- FAISS retrieves semantically relevant document chunks using embeddings.

- BM25 retrieves keyword-relevant chunks based on term frequency and inverse document frequency.

- Results from both retrievers are merged and deduplicated before being sent to the LLM.

This approach improves:

- Recall for factual and definition-based queries

- Robustness for technical and numeric questions

- Overall retrieval quality in heterogeneous documents

## Evaluation Results
To evaluate the performance of the tool with FAISS as the vector store, a ground truth dataset of 25 Q&A pairs was created from a PDF.
The PDF chosen was a scientific report published on the Global Agro-Ecological Zoning version 4 (GAEZ v4) methodology developed by the Food and Agriculture Organization of the United Nations (FAO) and the International Institute for Applied System Analysis (IIASA). It discusses the global maize harvestation and utilization patterns, as well as the effects of climate change and crop constraints on crop suitability and irrigation needs in different parts of the world. This PDF was chosen due to its versatile content which includes text, numerical quantifications, dates, tabular data, bar charts, heat maps and references. The tool was restricted to textual content only (no OCR yet) and was tested on 4 different setups. For each setup, a different combination of chunk size, chunk overlap and the number of top retrieved chunks (k) was used as shown below:

| Setup | Chunk Size | Overlap | k |
| ----- | ---------- | ------- | --|
| A     | 1000       | 150     | 3 |
| B     | 500        | 50      | 3 |
| C     | 500        | 50      | 5 |
| D     | 300        | 50      | 5 |

The bar chart below depicts the accuracy comparison for each setup:

![Screenshot_20250903_211801_Adobe Acrobat](https://github.com/user-attachments/assets/09034f10-455e-434d-a620-ad9443a297c5)


  The best performing setup was further used to evaluate retrieval quality metrics like Coverage/Recall and Reciprocal Rate and the overall results were as follows:

| Metric   | Result |
| -------- | ------ |
| Coverage | 0.29   |
| MRR      | 0.81   |

- **Coverage** measures how often relevant chunks were present among the retrieved ones.
- **MRR (Mean Reciprocal Rank)** evaluates the ranking quality of retrieval.

### Insights
- Reducing chunk size from 1000 to 500 tokens improved retrieval precision.
- Increasing retrieval depth (k=5) led to the highest accuracy.
- Very small chunks (300 tokens) reduced accuracy due to probable loss of semantic meaning.
- Best performing setup: chunk_size=500, chunk_overlap=50, k=5 resulting in 96% Accuracy, which is reasonably strong end-to-end performance.
- High MRR (0.81) shows that when relevant chunks are retrieved, they are usually ranked high (often at rank 1).
- Lower Coverage (0.29) indicates the system sometimes misses relevant chunks, suggesting more tuning of chunk size, overlap, or hybrid retrieval may improve recall.

 More detailed results and plots are available in the tests/test_results folder.

## Note:
The evaluation results above were obtained using a FAISS-only retriever.

A hybrid FAISS + BM25 retrieval pipeline was introduced afterward to address the observed low coverage (0.29). This enhancement is expected to improve recall and robustness, particularly for keyword-heavy and structured queries. Updated evaluation results for the hybrid retriever will be added in future iterations.

## Deployment
The tool was built keeping best-practices in mind such as usage of modular functions for better reusability and understanding. Currently, it is deployed as a web application on Streamlit Cloud for quick accessibility and demonstration.

Streamlit App link: https://rag-based-q-a-chatbot-4csgzsn7ey2mc34gon3mv4.streamlit.app

How to Use:

1. Visit the Streamlit app link.
2. Enter your Google API Key.
3. Upload your PDF data and ask any questions related to the document content.
4. Get real-time, LLM-powered context-aware answers.

## Future Work
- OCR for charts and figures

- Structured table extraction

- Hybrid retriever evaluation

