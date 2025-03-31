from datasets import load_metric

# Load evaluation metric
metric = load_metric("squad")

# Ground truth answer
ground_truth = {"id": "1", "answers": [{"text": "surgery, radiation therapy, chemotherapy, targeted therapy, and immunotherapy"}]}

# Model prediction
prediction = {"id": "1", "prediction_text": answer}

# Compute metrics
results = metric.compute(predictions=[prediction], references=[ground_truth])
print(f"Exact Match: {results['exact_match']:.2f}%")
print(f"F1 Score: {results['f1']:.2f}%")