from transformers import TrainerCallback
import matplotlib.pyplot as plt


class AccuracyLogger(TrainerCallback):
    def __init__(self):
        self.train_accuracies = []
        self.eval_accuracies = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_accuracy" in metrics:
            self.eval_accuracies.append(metrics["eval_accuracy"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "train_loss" in logs:
            self.train_accuracies.append(1 - logs["train_loss"])  # Approximate accuracy

accuracy_logger = AccuracyLogger()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[accuracy_logger]  # Attach the logger
)
train_accuracies = accuracy_logger.train_accuracies
eval_accuracies = accuracy_logger.eval_accuracies
epochs = list(range(1, len(eval_accuracies) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, eval_accuracies, label="Validation Accuracy", marker="o", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Fine-Tuned BioBERT Accuracy Progress")
plt.legend()
plt.grid()
plt.show()