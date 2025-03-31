import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer,
    TrainingArguments, Trainer, default_data_collator
)

# ✅ Set Device (Use Apple Metal acceleration if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# ✅ Load BioBERT Model & Tokenizer
model_name = "monologg/biobert_v1.1_pubmed"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

# ✅ Load MedQuAD Dataset
dataset_path = "./medquad.json"
if not os.path.exists(dataset_path):
    print(f"❌ Dataset {dataset_path} not found!")
    exit()

with open(dataset_path, "r") as f:
    data = json.load(f)

# ✅ Convert JSON to HuggingFace Dataset
contexts, questions, answers = [], [], []

for entry in data:
    if "question" in entry and "answer" in entry:
        questions.append(entry["question"])
        contexts.append(entry["answer"])
        answers.append({"text": entry["answer"], "answer_start": 0})

dataset = {"question": questions, "context": contexts, "answers": answers}
train_size = int(0.8 * len(questions))

train_dataset = {key: dataset[key][:train_size] for key in dataset}
val_dataset = {key: dataset[key][train_size:] for key in dataset}

# ✅ Tokenize Data & Prepare for Question Answering
def preprocess(example):
    encoding = tokenizer(
        example["question"], example["context"],
        truncation=True, padding="max_length",
        max_length=512
    )
    encoding["start_positions"] = [example["answers"]["answer_start"]]
    encoding["end_positions"] = [example["answers"]["answer_start"] + len(example["answers"]["text"])]
    return encoding

# ✅ Convert to Dataset format & Apply Preprocessing
train_dataset = Dataset.from_dict(train_dataset).map(preprocess)
val_dataset = Dataset.from_dict(val_dataset).map(preprocess)

# ✅ Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none"
)

# ✅ Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# ✅ Train Model
trainer.train()

# ✅ Save Model
model.save_pretrained("./fine_tuned_biobert")
tokenizer.save_pretrained("./fine_tuned_biobert")
print("✅ Model saved!")

# ✅ Generate Accuracy Graph
log_file = "./results/trainer_state.json"

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        log_data = json.load(f)

    # ✅ Extract Steps & Accuracy
    steps, accuracy = [], []
    for log in log_data["log_history"]:
        if "eval_loss" in log:
            steps.append(log["step"])
            accuracy.append(1 - log["eval_loss"])

    # ✅ Plot Graph
    plt.figure(figsize=(8, 5))
    plt.plot(steps, accuracy, marker="o", linestyle="-", color="b", label="Accuracy")
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy")
    plt.title("Fine-Tuning Accuracy Over Steps")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("❌ Training log file not found. Skipping accuracy graph.")