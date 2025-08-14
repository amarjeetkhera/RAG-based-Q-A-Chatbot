# RAG-based Q&A Chatbot
## Overview
This project presents an intuitive Retrieval-Augmented Generation (RAG) based chat assistant designed to allow users to ask questions directly from their PDF documents. Built to make document understanding faster, easier, and more interactive, the tool provides instant, AI-powered answers from large files like research papers, manuals, or reports — without the need to read through them line by line.

Using state-of-the-art language models and embeddings, the chatbot retrieves relevant document chunks and generates accurate, context-aware responses in real-time, enhancing productivity and decision-making across multiple domains like legal, education, research, and enterprise documentation. It demonstrates the power of combining document parsing, semantic search, and large language models to enable smart, personalized, and accurate document-based question-answering.

## Description
The RAG Chatbot is implemented using a Python backend and Streamlit frontend, making it deployable as a user-friendly web application. It uses the Gemini Flash LLM from Google to perform contextual question-answering based on document retrieval.

At a high level, the tool follows an architecture as shown in the diagram below:

![Q A Chatbot_20250814_134159_0000](https://github.com/user-attachments/assets/1dac383d-e4bb-4a2f-b954-ba8fe591c10e)

(Kindly cite this repository if you use this flow chart)

## RAG vs Traditional Chatbots
Traditional chatbots often rely on static rule-based logic or general-purpose LLMs with no access to specific documents. In contrast, RAG bridges the gap by providing the model with retrieved context directly from your uploaded files.

This enhances:

- Accuracy: Reduces hallucinations and generic answers
- Explainability: Responses are traceable to original document content
- Relevance: Answers stay grounded to the PDF uploaded by the user

  RAG doesn’t just simulate intelligence — it gives your chat assistant access to real relevant knowledge.

## Deployment
The tool was built keeping best-practices in mind such as usage of modular functions for better reusability and understanding. Currently, it is deployed as a web application on Streamlit Cloud for quick accessibility and demonstration.

Streamlit App link: https://rag-based-q-a-chatbot-4csgzsn7ey2mc34gon3mv4.streamlit.app

How to Use:

1. Visit the Streamlit app link.
2. Upload your PDF data and ask any question related to the document content.
3. Get real-time, LLM-powered context-aware answers.
