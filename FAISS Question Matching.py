import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import json
# Load MedQuAD Q&A pairs
with open("medquad.json", "r") as f:
    medquad_data = json.load(f)

questions = [q["question"] for q in medquad_data]
answers = [q["answer"] for q in medquad_data]

# Load BioBERT embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed").to(device)
embed_tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")


def encode_text(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


# Encode questions and build FAISS index
question_embeddings = np.vstack([encode_text(q) for q in questions])
index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)


def retrieve_best_match(user_question):
    query_embedding = encode_text(user_question)
    _, best_match_idx = index.search(query_embedding, 1)

    matched_question = questions[best_match_idx[0][0]]
    matched_answer = answers[best_match_idx[0][0]]

    return matched_question, matched_answer


# Test retrieval
user_query = "What are the treatments for lung cancer?"
matched_q, matched_a = retrieve_best_match(user_query)
print(f"🔍 Matched Question: {matched_q}\n📖 Matched Answer: {matched_a}")