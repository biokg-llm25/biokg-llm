"""
Script to extract and normalize medical terms from a text file using SciSpaCy and NLTK.

- Accepts a text file with Q/A pairs related to medical terms
- Filters and normalizes medical entities using a biomedical NER model
- Removes English stopwords
- Saves the final term list to an output file
"""

import spacy
import nltk
import argparse
import os

# Download required NLTK resources
nltk.download("stopwords")

# Load SciSpaCy Biomedical Model for biomedical entity recognition
nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")  


def extract_medical_terms(txt_file_path, output_path):
    """
    Extracts, filters, and saves medical terms from a structured text file.
    
    Parameters:
    - txt_file_path (str): Path to the input text file
    - output_path (str): Path to save the output text file
    """
    # These questions are used to identify relevant medical entities in the text file. The questions are based on the structure of the input text file and the expected answers.
    medical_questions = {
        "List the side-effects found in the leaflet as a comma-separated list of names only.",
        "List the active ingredient(s) found in the leaflet as a comma-separated list of names only.",
        "List the inactive ingredients found in the leaflet as a comma-separated list of names only.",
        "List the contraindications found in the leaflet as a comma-separated list of names only.",
        "List the warnings and precautions found in the leaflet as a comma-separated list of names only.",
    }

    # Define valid medical entity types for filtering
    valid_medical_types = {"CHEMICAL", "DISEASE", "DRUG"}

    # Read extracted medical terms from the TXT file
    entities = []

    with open(txt_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        # Check if the line contains a relevant medical question from the list defined above
        if any(line.startswith(f"Q: {question}") for question in medical_questions):
            answer_line = next((l.strip() for l in lines[i + 1:] if l.startswith("A: ")), None)  # Extract the answer
            if answer_line:
                answer = answer_line.replace("A: ", "").strip()
                if "," in answer:
                    entities.extend([v.strip().lower() for v in answer.split(",")])
                else:
                    entities.append(answer.lower())

    # Normalize terms using SciSpaCy (filtering by medical entity type)
    filtered_medical_terms = set()
    for term in entities:
        doc = nlp_bc5cdr(term)
        for ent in doc.ents:
            if ent.label_ in valid_medical_types:  # Only keep valid medical categories
                filtered_medical_terms.add(ent.text)

    # Remove stopwords using NLTK
    try:
        nltk_stopwords = set(nltk.corpus.stopwords.words("english"))
        filtered_medical_terms = {
            term for term in filtered_medical_terms if term.lower() not in nltk_stopwords
        }
    except LookupError:
        print("Warning: NLTK stopwords not found, skipping stopword filtering.")

    # Save final cleaned medical terms
    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted(filtered_medical_terms):
            f.write(term + "\n")

    print(f"Final medical terms list saved with {len(filtered_medical_terms)} terms!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and normalize medical terms from HPRA text file.")
    parser.add_argument("--input", required=True, help="Path to the input .txt file")
    parser.add_argument("--output", required=True, help="Path to save the final .txt file with extracted terms")
    args = parser.parse_args()

    extract_medical_terms(args.input, args.output)
