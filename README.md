# AI Notes Search Engine with RAG

> Semantic search and Question Answering across academic notes, textbooks, and PDF documents using Sentence Transformers, FAISS vector indexing, and Gemini Generative AI.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-009688)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Problem & Overview

Finding specific concepts across dozens of slide decks, lecture summaries, and PDF documents using traditional keyword search fails when synonyms or phrased questions are used. 

**AI Notes Search Engine** builds a local Retrieval-Augmented Generation (RAG) pipeline:
1. Ingests heterogeneous notes (`.txt`, `.pdf`).
2. Encodes document passages into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
3. Performs sub-millisecond similarity search using a **FAISS** L2 flat index.
4. Synthesizes a grounded answer using the **Google Gemini API** (`gemini-2.5-flash`), strictly anchored to the retrieved context.

---

## 🏗️ Architecture

```
User Query ──► SentenceTransformer (all-MiniLM-L6-v2) ──► Query Vector
                                                               │
                                                               ▼
Document Corpus ──► PDF/Text Loader ──► Embeddings ──► FAISS Index
                                                               │
                                  Top-1 Relevant Document ◄────┘
                                             │
                                             ▼
                             Context + Query ──► Gemini 2.5 Flash ──► Final Answer
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### 2. Installation
```bash
git clone https://github.com/div339175/ai-notes-search-engine.git
cd ai-notes-search-engine
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Build or Re-Index Notes
Place your `.txt` or `.pdf` notes inside the `notes/` directory, then generate embeddings and index:
```bash
python src/embed.py
python src/build_index.py
```

### 5. Launch Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
ai-notes-search-engine/
├── notes/                   # Text notes and PDF documents
├── src/
│   ├── embed.py             # Parses documents and produces dense embeddings
│   ├── build_index.py       # Constructs FAISS L2 flat index from embeddings
│   ├── rag.py               # Gemini RAG prompting with API fallback handling
│   └── search.py            # CLI semantic search demonstration
├── app.py                   # Streamlit interactive web interface
├── requirements.txt         # Project dependencies
├── LICENSE                  # MIT License
└── README.md
```

---

## ⚠️ Limitations & Future Work
- **Chunking:** Currently processes documents at the file level; future iterations will implement sliding-window paragraph chunking for long textbooks.
- **Multi-Hop Search:** Expand retrieval to top-$k$ passages with reranking (e.g., FlashRank or Cross-Encoder).

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.