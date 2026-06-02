# Multimodal RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system that processes 
PDF, CSV, and image documents using AWS and LLMs.

## Architecture
- **AWS S3** — Document storage
- **AWS Lambda** — Serverless auto-triggered ingestion on S3 upload
- **Amazon Bedrock (Titan)** — Document embeddings
- **FAISS** — Vector similarity search
- **ML Classifier** — Auto-routes documents (text/tabular/vision)
- **Confidence Scorer** — Flags low-certainty answers
- **FastAPI** — Backend API
- **Streamlit** — Interactive frontend UI

## Tech Stack
Python | AWS S3 | AWS Lambda | Amazon Bedrock | FAISS | 
FastAPI | Streamlit | Scikit-learn | Groq (LLaMA3)

## How to Run
1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Add `.env` file with AWS credentials and GROQ_API_KEY
6. Start FastAPI: `uvicorn app.main:app`
7. Start Streamlit: `streamlit run frontend/app.py`

## Features
- Upload PDF, CSV, or image documents
- Automatic document classification
- Semantic search using vector embeddings
- Confidence scoring on every answer
- Serverless AWS Lambda pipeline