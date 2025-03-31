import torch
import json
import os
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoTokenizer, AutoModel

# 🔹 Enable Mac Metal Backend for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 🔹 Import FAISS for fast retrieval
try:
    import faiss
except ImportError:
    print("❌ FAISS not installed! Install it with `pip install faiss-cpu`.")
    exit()

# 🔹 Load BioBERT Models
class_model = AutoModelForSequenceClassification.from_pretrained("monologg/biobert_v1.1_pubmed").to(device)
qa_model = AutoModelForQuestionAnswering.from_pretrained("monologg/biobert_v1.1_pubmed").to(device)
embed_model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed").to(device)

# 🔹 Load Tokenizers
class_tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
qa_tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
embed_tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")

# 🔹 Load MedQuAD Dataset
medquad_file = "medquad.json"
if not os.path.exists(medquad_file):
    print(f"❌ Error: {medquad_file} not found! Run preprocessing first.")
    exit()

with open(medquad_file, "r") as f:
    medquad_data = json.load(f)

questions = [entry["question"] for entry in medquad_data if "question" in entry]
answers = [entry["answer"] for entry in medquad_data if "answer" in entry]

if not questions:
    print("❌ Error: No valid questions found in MedQuAD dataset.")
    exit()

print(f"✅ Loaded {len(questions)} questions.")

# 🔹 Encode Questions (Batch Processing for Mac Optimization)
def encode_text_batch(texts, batch_size=16):
    """Batch-process text into embeddings using BioBERT (optimized for Mac Metal)."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = embed_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# 🔹 Encode all MedQuAD questions (Batch Processing)
question_embeddings = encode_text_batch(questions, batch_size=16)

# 🔹 Build FAISS Index
index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

def retrieve_best_match(user_question):
    """Retrieve the most relevant question-answer pair from MedQuAD."""
    query_embedding = encode_text_batch([user_question])[0].reshape(1, -1)
    _, best_match_idx = index.search(query_embedding, 1)
    return questions[best_match_idx[0][0]], answers[best_match_idx[0][0]]

def classify_question(question):
    """Classify question category using BioBERT."""
    inputs = class_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = class_model(**inputs)
    predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
    return f"Category {predicted_class_idx}"

def answer_question(question, context):
    """Extract answer using BioBERT QA model."""
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_idx = torch.argmax(outputs.start_logits).item()
    end_idx = torch.argmax(outputs.end_logits).item()
    if end_idx < start_idx:
        return "No valid answer found."
    return qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx + 1])
    ).strip()

# 🔹 Example Query
user_question = "What are the treatments for lung cancer?"

# Step 1: Retrieve Context from MedQuAD
retrieved_question, retrieved_context = retrieve_best_match(user_question)
print(f"🔍 Retrieved Question: {retrieved_question}")
print(f"📖 Retrieved Context: {retrieved_context}")

# Step 2: Classify the Question
category = classify_question(user_question)
print(f"🔹 Predicted Category: {category}")

# Step 3: Answer Using BioBERT QA Model
answer = answer_question(user_question, retrieved_context)
print(f"✅ Answer: {answer}")