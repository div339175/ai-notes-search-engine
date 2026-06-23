import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.rag import answer_question
from pypdf import PdfReader

st.title("AI Notes Search Engine")

query = st.text_input("Ask a question")

if query:

    model = SentenceTransformer("all-MiniLM-L6-v2")

    filenames = np.load("filenames.npy")

    index = faiss.read_index("notes.index")

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, 1)

    best_idx = I[0][0]

    path = f"notes/{filenames[best_idx]}"

    if filenames[best_idx].endswith(".txt"):

        with open(path, "r") as f:
            text = f.read()

    else:

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

    answer = answer_question(
        query,
        text[:4000]
    )

    st.subheader("Answer")

    st.write(answer)

    with st.expander("Source Document"):
        st.write(filenames[best_idx])
        st.write(text[:1000])