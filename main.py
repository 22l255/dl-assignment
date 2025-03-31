import json
from sklearn.preprocessing import LabelEncoder

# Load preprocessed data
with open("/Users/raghavan/Desktop/dl/medquad_preprocessed.json", "r") as f:
    data = json.load(f)

# Ensure the dataset contains categories
if not data or "category" not in data[0]:
    raise ValueError("❌ Data is empty or missing 'category' field.")

# Encode labels
label_encoder = LabelEncoder()
categories = [item["category"] for item in data]

if not categories:
    raise ValueError("❌ No categories found in the dataset!")

labels = label_encoder.fit_transform(categories)

# Save label encoder mapping
label_mapping = {str(i): label for i, label in enumerate(label_encoder.classes_)}

with open("label_encoder.json", "w") as f:
    json.dump(label_mapping, f, indent=4)

print("✅ label_encoder.json saved successfully!")