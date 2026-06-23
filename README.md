# AI Notes Search Engine with RAG

Search notes and PDFs using semantic search, FAISS vector retrieval, and Gemini-powered question answering.

## 🔗 Features

- Semantic Search
- Text & PDF Support
- Sentence Transformer Embeddings
- FAISS Vector Database
- Top-K Document Retrieval
- Streamlit Web Interface
- Retrieval-Augmented Generation (RAG)
- Gemini-Powered Question Answering

## 🛠️ Tech Stack

- Python
- Sentence Transformers
- FAISS
- Streamlit
- Google Gemini API
- NumPy
- PyPDF
- python-dotenv

## 🏗️ Architecture

User Question
      ↓
Sentence Transformer
      ↓
FAISS Retrieval
      ↓
Relevant Document
      ↓
Gemini 2.5 Flash
      ↓
Generated Answer

## Run

pip install -r requirements.txt
python app.py