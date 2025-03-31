import torch
import json
import os
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoTokenizer

# Load Classification Model
class_model_path = "./biobert_classification"  # Change this if model is in a different directory
qa_model_name = "deepset/bert-base-cased-squad2"

# Try loading models, handle errors
try:
    class_model = AutoModelForSequenceClassification.from_pretrained(class_model_path)
    class_tokenizer = AutoTokenizer.from_pretrained(class_model_path)
except Exception as e:
    print(f"❌ Error loading classification model: {e}")
    exit()

try:
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
except Exception as e:
    print(f"❌ Error loading QA model: {e}")
    exit()

# Set Device (Use Mac GPU if available, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_model.to(device)
qa_model.to(device)

# Load Label Mapping for Classification
label_file = "label_mapping.json"
label_mapping = {}

if os.path.exists(label_file):
    with open(label_file, "r") as f:
        label_mapping = json.load(f)
else:
    print(f"⚠️ Warning: {label_file} not found! Using default labels.")
    label_mapping = {
        "0": "Unknown Category"
    }

def classify_question(question):
    """Classifies the question and returns the predicted category."""
    inputs = class_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = class_model(**inputs)

    predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
    print(f"🔍 Raw Predicted Class Index: {predicted_class_idx}")  # Debugging line
    predicted_category = label_mapping.get(str(predicted_class_idx), "Unknown")
    return predicted_category


def answer_question(question, context):
    """Uses the QA model to extract an answer from the given context."""
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
        device)

    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_idx = torch.argmax(outputs.start_logits).item()
    end_idx = torch.argmax(outputs.end_logits).item()

    # Ensure start index comes before end index
    if end_idx < start_idx:
        return "No valid answer found."

    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx + 1])
    )
    return answer.strip()

# Example Question
question = "What are the treatments for lung cancer?"
context = """Lung cancer treatment depends on the stage and type of cancer. Common treatments include surgery, chemotherapy, radiation therapy, targeted therapy, and immunotherapy. Early-stage lung cancer may be treated with surgery, while advanced cases often require a combination of therapies."""

# Run Classification
category = classify_question(question)
print(f"🔹 Predicted Category: {category}")

# Run QA
answer = answer_question(question, context)
print(f"✅ Answer: {answer}")