import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import json

# Load BioBERT
biobert_model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed")
biobert_tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")

# Load MedQuAD dataset
with open("medquad.json", "r") as f:
    dataset = json.load(f)

# Function to generate embeddings
def get_embedding(text):
    inputs = biobert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = biobert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token

# Generate and store embeddings
embeddings = np.array([get_embedding(d["question"]) for d in dataset])
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 Distance
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "medquad_faiss.index")