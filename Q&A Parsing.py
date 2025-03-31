import xml.etree.ElementTree as ET
import json
import os

# Define dataset path
medquad_dir = "/Users/raghavan/Desktop/dl/MedQuAD-master/"
output_json = "medquad.json"


# Function to parse a single XML file
def parse_medquad_file(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    qa_pairs = []
    for item in root.findall(".//QAPair"):
        question = item.find("Question").text if item.find("Question") is not None else None
        answer = item.find("Answer").text if item.find("Answer") is not None else None

        if question and answer:
            qa_pairs.append({"question": question.strip(), "answer": answer.strip()})

    return qa_pairs


# Iterate over all MedQuAD XML files
all_qa_pairs = []
for folder in os.listdir(medquad_dir):
    folder_path = os.path.join(medquad_dir, folder)
    if os.path.isdir(folder_path):  # Check if it is a directory
        for file in os.listdir(folder_path):
            if file.endswith(".xml"):
                file_path = os.path.join(folder_path, file)
                all_qa_pairs.extend(parse_medquad_file(file_path))

# Save extracted Q&A pairs to JSON
if all_qa_pairs:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, indent=4, ensure_ascii=False)
    print(f"✅ Successfully extracted {len(all_qa_pairs)} Q&A pairs from MedQuAD.")
else:
    print("❌ No valid Q&A pairs found. Check XML structure!")