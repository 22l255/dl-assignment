import json
from sklearn.model_selection import train_test_split

# Load preprocessed MedQuAD dataset
json_path = "/Users/raghavan/Desktop/dl/medquad_preprocessed.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Extract Questions and Categories (Labels)
texts = [item["question"] for item in data]
labels = [item["category"] for item in data]  # Category as label

# Encode Labels as Numbers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split into Train & Test
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

print(f"✅ Loaded {len(data)} Q&A pairs. Train Size: {len(train_texts)}, Test Size: {len(test_texts)}")
import json

# Load preprocessed MedQuAD dataset
with open(json_path, "r") as f:
    data = json.load(f)

# Format for BioBERT Q&A
qa_dataset = [
    {"context": item["question"], "question": item["question"], "answer": item["answer"]}
    for item in data
]

# Save formatted dataset
qa_json_path = "/Users/raghavan/Desktop/dl/medquad_QA_formatted.json"
with open(qa_json_path, "w") as f:
    json.dump(qa_dataset, f, indent=4)

print(f"✅ Formatted {len(qa_dataset)} Q&A pairs for BioBERT Q&A training.")