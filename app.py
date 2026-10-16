import os
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.rag import answer_question
from pypdf import PdfReader

st.set_page_config(page_title="AI Notes Search Engine", page_icon="📚", layout="centered")

st.title("📚 AI Notes Search Engine with RAG")
st.markdown("Semantic search and retrieval-augmented question answering across course notes and documents.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_index_and_files():
    if not os.path.exists("filenames.npy") or not os.path.exists("notes.index"):
        return None, None
    filenames = np.load("filenames.npy", allow_pickle=True)
    index = faiss.read_index("notes.index")
    return filenames, index

model = load_model()
filenames, index = load_index_and_files()

if filenames is None or index is None:
    st.error("Index or filenames not found! Please run `python src/embed.py` and `python src/build_index.py` first.")
else:
    query = st.text_input("Ask a question about your notes:")

    if query:
        with st.spinner("Searching notes..."):
            query_embedding = model.encode([query])
            query_embedding = np.array(query_embedding).astype("float32")

            D, I = index.search(query_embedding, 1)
            best_idx = I[0][0]

            path = f"notes/{filenames[best_idx]}"
            text = ""

            if filenames[best_idx].endswith(".txt"):
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
            elif filenames[best_idx].endswith(".pdf"):
                if os.path.exists(path):
                    reader = PdfReader(path)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"

            if not text:
                st.warning("Could not extract text from the matched document.")
            else:
                answer = answer_question(query, text[:4000])

                st.subheader("💡 Answer")
                st.write(answer)

                with st.expander(f"📄 Source Document: {filenames[best_idx]}"):
                    st.text(text[:1500] + ("..." if len(text) > 1500 else ""))