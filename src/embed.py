from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import numpy as np
import os

model=SentenceTransformer('all-MiniLM-L6-v2')
notes=[]
filenames=[]

for file in os.listdir("notes"):
    path=f"notes/{file}"
    if file.endswith(".txt"):
        with open(path,"r")as f:
            text=f.read()
    elif file.endswith(".pdf"):
        reader=PdfReader(path)
        text=""
        for page in reader.pages:
            extracted=page.extract_text()

            if extracted:
                text+=extracted+"\n"
    else:
        continue
    notes.append(text)
    filenames.append(file)
        

embeddings=model.encode(notes)
np.save("embedding.npy",embeddings)
np.save("filenames.npy",filenames)
print("Embedding saved!")