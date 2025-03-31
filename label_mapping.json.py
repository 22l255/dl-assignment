import json

# Define the label mapping
label_mapping = {
    "0": "Lung Cancer Treatment",
    "1": "Breast Cancer Treatment",
    "2": "Colon Cancer Treatment",
    "3": "General Cancer Treatment"
}



# Save to a JSON file
with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f, indent=4)
import json

import json

label_file = "label_mapping.json"
try:
    with open(label_file, "r") as f:
        label_mapping = json.load(f)
    print("✅ label_mapping.json loaded successfully:", label_mapping)
except FileNotFoundError:
    print("❌ Error: label_mapping.json not found!")
except json.JSONDecodeError:
    print("❌ Error: Invalid JSON format in label_mapping.json!")