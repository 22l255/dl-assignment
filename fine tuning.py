import os
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ✅ Fix Parallelism Issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Set Device (MPS for Mac, CUDA for NVIDIA, CPU fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ✅ Load Preprocessed Dataset
with open("/Users/raghavan/Desktop/dl/medquad_preprocessed.json", "r") as f:
    data = json.load(f)

# ✅ Encode Labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform([item["category"] for item in data])

# ✅ Prepare Train/Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    [item["question"] for item in data], labels, test_size=0.1, random_state=42
)

# ✅ Load BioBERT Tokenizer
model_name = "monologg/biobert_v1.1_pubmed"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Tokenize Data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# ✅ Convert to PyTorch Dataset
class MedQuADDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = MedQuADDataset(train_encodings, train_labels)
test_dataset = MedQuADDataset(test_encodings, test_labels)

# ✅ Load BioBERT for Classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))
model.to(device)  # Move model to selected device

# ✅ Training Arguments (Optimized for Speed)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,  # Increased batch size for faster training
    per_device_eval_batch_size=16,  # Larger eval batch for speed
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,  # More frequent logging for progress updates
    report_to="none",
    load_best_model_at_end=True,
    fp16=False if device.type == "mps" else True,  # Disable FP16 for Mac MPS
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# ✅ Train Model
trainer.train()

# ✅ Save Model & Tokenizer
model.save_pretrained("./biobert_classification")
tokenizer.save_pretrained("./biobert_classification")

print("✅ Classification Model Training Complete!")