import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.title("AI Notes Search Engine")

query = st.text_input("Ask a question")

if query:

    model = SentenceTransformer("all-MiniLM-L6-v2")

    filenames = np.load("filenames.npy")

    index = faiss.read_index("notes.index")

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, 3)

    st.subheader("Top Results")

    for idx in I[0]:

        st.write(f"📄 {filenames[idx]}")

        path = f"notes/{filenames[idx]}"

        if filenames[idx].endswith(".txt"):

            with open(path, "r") as f:
                text = f.read()

        else:

            from pypdf import PdfReader

            reader = PdfReader(path)

            text = ""

            for page in reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted

        st.write(text[:500])

        st.divider()