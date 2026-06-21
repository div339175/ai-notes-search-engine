import numpy as np
import faiss

embeddings=np.load("embedding.npy")
embeddings=embeddings.astype("float32")
dimension=embeddings.shape[1]

index=faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index,"notes.index")
print("FAISS index created successfully!")
