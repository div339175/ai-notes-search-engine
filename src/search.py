from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import faiss
from pypdf import PdfReader


model=SentenceTransformer('all-MiniLM-L6-v2')
filenames=np.load("filenames.npy")

index=faiss.read_index("notes.index")

query=input("Ask:")
query_embedding=model.encode([query])
query_embedding=np.array(query_embedding).astype("float32")

D,I=index.search(query_embedding,3)
best_idx=I[0][0]

print("\nMost Relevant Note:")
print(filenames[best_idx])

file=filenames[best_idx]
path=f"notes/{file}"
if file.endswith(".txt"):
    with open(path,"r")as f:
        print(f.read())
elif file.endswith(".pdf"):
    reader=PdfReader(path)
    text=""
    for page in reader.pages:
        extracted=page.extract_text()
        if extracted:
            text+=extracted
    print(text[:2000])

