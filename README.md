# RAG-based Q&A Chatbot

## Overview
This project presents an intuitive Retrieval-Augmented Generation (RAG) based chat assistant designed to allow users to ask questions directly from their PDF documents. Built to make document understanding faster, easier, and more interactive, the tool provides instant, AI-powered answers from large files like research papers, manuals, or reports — without the need to read through them line by line.

Using state-of-the-art language models and embeddings, the chatbot retrieves relevant document chunks and generates accurate, context-aware responses in real-time, enhancing productivity and decision-making across multiple domains like legal, education, research, and enterprise documentation. It demonstrates the power of combining document parsing, semantic search, and large language models to enable smart, personalized, and accurate document-based question-answering.

## Description
The RAG Chatbot is implemented using a Python backend and Streamlit frontend, making it deployable as a user-friendly web application. It uses the Gemini 2.0 Flash LLM from Google to perform contextual question-answering based on document retrieval.

At a high level, the tool follows an architecture as shown in the diagram below:

<img width="1920" height="1080" alt="Q A Chatbot_20250903_211203_0000" src="https://github.com/user-attachments/assets/40d4990c-ecce-4ff9-ae47-bc7c3e024f55" />


(Kindly cite this repository if you use this flow chart)

## RAG vs Traditional Chatbots
Traditional chatbots often rely on static rule-based logic or general-purpose LLMs with no access to specific documents. In contrast, RAG bridges the gap by providing the model with retrieved context directly from your uploaded files.

This enhances:

- Accuracy: Reduces hallucinations and generic answers
- Explainability: Responses are traceable to original document content
- Relevance: Answers stay grounded to the PDF uploaded by the user

  RAG doesn’t just simulate intelligence — it gives your chat assistant access to real relevant knowledge.

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


## Deployment
The tool was built keeping best-practices in mind such as usage of modular functions for better reusability and understanding. Currently, it is deployed as a web application on Streamlit Cloud for quick accessibility and demonstration.

Streamlit App link: https://rag-based-q-a-chatbot-4csgzsn7ey2mc34gon3mv4.streamlit.app

How to Use:

1. Visit the Streamlit app link.
2. Enter your Google API Key.
3. Upload your PDF data and ask any questions related to the document content.
4. Get real-time, LLM-powered context-aware answers.
