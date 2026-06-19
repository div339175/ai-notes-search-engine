from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

model=SentenceTransformer('all-MiniLM-L6-v2')


embeddings = np.load("embedding.npy")
filenames=np.load("filenames.npy")

query=input("Ask:")
query_embedding=model.encode([query])
scores=cosine_similarity(query_embedding,embeddings)[0]

best_idx=scores.argmax()

print("\nMost Relevant Note:")
print(filenames[best_idx])

with open(f"notes/{filenames[best_idx]}","r")as f:
    print(f.read())


