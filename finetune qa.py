'''from transformers import AutoModelForQuestionAnswering
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
qa_model.to(device)
# Load preprocessed Q&A dataset
with open("/Users/raghavan/Desktop/dl/medquad_QA_formatted.json", "r") as f:
    qa_data = json.load(f)

# Split Data
train_data, test_data = train_test_split(qa_data, test_size=0.1, random_state=42)

# Tokenize Q&A Data
train_encodings = tokenizer(
    [item["context"] for item in train_data],
    [item["question"] for item in train_data],
    truncation=True, padding=True, max_length=512, return_tensors="pt"
)

test_encodings = tokenizer(
    [item["context"] for item in test_data],
    [item["question"] for item in test_data],
    truncation=True, padding=True, max_length=512, return_tensors="pt"
)

# Convert to PyTorch Dataset
class MedQuADQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, answers):
        self.encodings = encodings
        self.answers = answers

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

train_dataset = MedQuADQADataset(train_encodings, [item["answer"] for item in train_data])
test_dataset = MedQuADQADataset(test_encodings, [item["answer"] for item in test_data])

# Load BioBERT for Q&A
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Trainer
trainer = Trainer(
    model=qa_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train Model
trainer.train()

# Save Model
qa_model.save_pretrained("./biobert_qa")
tokenizer.save_pretrained("./biobert_qa")

print("✅ Q&A Model Training Complete!")'''
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/biobert-base-cased-squad2")
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/biobert-base-cased-squad2")