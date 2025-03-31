import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./biobert_classification")
tokenizer = AutoTokenizer.from_pretrained("./biobert_classification")

# Load label mapping
with open("label_encoder.json", "r") as f:
    label_mapping = json.load(f)


# Function to classify question
def classify_question(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
    predicted_category = label_mapping.get(str(predicted_class_idx), "Unknown")

    return predicted_category


# Test the classifier
question = "What are the treatments for lung cancer?"
predicted_category = classify_question(question)

print(f"🔍 Predicted Category: {predicted_category}")