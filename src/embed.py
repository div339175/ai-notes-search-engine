from sentence_transformers import SentenceTransformer

import numpy as np
import os

model=SentenceTransformer('all-MiniLM-L6-v2')
notes=[]
filenames=[]

for file in os.listdir("notes"):
    with open(f"notes/{file}","r")as f:
        text=f.read()
    notes.append(text)
    filenames.append(file)

embeddings=model.encode(notes)
np.save("embedding.npy",embeddings)
np.save("filenames.npy",filenames)
print("Embedding saved!")