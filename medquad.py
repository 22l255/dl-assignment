import os
import json
import xml.etree.ElementTree as ET

# Path to the MedQuAD dataset folder
dataset_folder = "/Users/raghavan/Desktop/dl/MedQuAD-master/"
output_file = "medquad.json"


def extract_qa_from_xml(file_path):
    """Extract questions and answers from a MedQuAD XML file."""
    qa_pairs = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for qa in root.findall(".//QAPair"):
            question = qa.find("Question").text if qa.find("Question") is not None else ""
            answer = qa.find("Answer").text if qa.find("Answer") is not None else ""

            if question and answer:
                qa_pairs.append({"question": question.strip(), "answer": answer.strip()})
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

    return qa_pairs


def process_medquad_dataset(dataset_folder, output_file):
    """Process all XML files in the MedQuAD dataset and save them as JSON."""
    all_qa_pairs = []

    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                qa_pairs = extract_qa_from_xml(file_path)
                all_qa_pairs.extend(qa_pairs)

    if all_qa_pairs:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_qa_pairs, f, indent=4, ensure_ascii=False)
        print(f"✅ MedQuAD JSON file created: {output_file} ({len(all_qa_pairs)} Q&A pairs)")
    else:
        print("❌ No valid Q&A pairs found.")


# Run the script
process_medquad_dataset(dataset_folder, output_file)