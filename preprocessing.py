import tensorflow as tf

# Check GPU Support
print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Set GPU for training
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

import os
import xml.etree.ElementTree as ET
import json

# 📌 Set Dataset Path (Update if needed)
dataset_path = "/Users/raghavan/Desktop/dl/MedQuAD-master"


# 📌 Extract Q&A Data
def extract_medquad_data(base_folder):
    dataset = []

    for class_folder in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_folder)

        if os.path.isdir(class_path):  # Ensure it's a folder
            for file in os.listdir(class_path):
                if file.endswith(".xml"):
                    tree = ET.parse(os.path.join(class_path, file))
                    root = tree.getroot()

                    # Find <QAPairs> and extract all <QAPair> inside
                    qa_pairs = root.find("QAPairs")
                    if qa_pairs is not None:
                        for qa in qa_pairs.findall("QAPair"):
                            question_tag = qa.find("Question")
                            question = question_tag.text.strip() if question_tag is not None and question_tag.text else "No question available"

                            answer_tag = qa.find("Answer")
                            answer = answer_tag.text.strip() if answer_tag is not None and answer_tag.text else "No answer available"

                            if question and answer:  # Ensure it's valid
                                dataset.append({"question": question, "answer": answer, "category": class_folder})

    return dataset


# 📌 Extract Data
medquad_data = extract_medquad_data(dataset_path)

# 📌 Save Processed Data
json_path = "/Users/raghavan/Desktop/dl/medquad_preprocessed.json"
with open(json_path, "w") as f:
    json.dump(medquad_data, f, indent=4)

print(f"✅ Extracted {len(medquad_data)} Q&A pairs. Saved at {json_path}")