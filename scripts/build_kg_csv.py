"""
Script to process the text file containing structured Q&A extracted from HPRA drug leaflets,
to build a structured medical knowledge graph in CSV format.

- Reads Q/A text and maps known questions to relationship types
- Normalizes extracted medical terms using SciSpaCy and NLTK
- Applies fuzzy matching to correct spelling variations
- Extracts min/max dosages, standardizes units
- Outputs relationships as (Subject, Predicate, Object) triples in a CSV file

Usage:
    python build_kg_csv.py --input hpra_kg.txt --terms final_medical_terms_hpra.txt --output hpra_complete_network.csv
"""

# Import necessary libraries
import re
import pandas as pd
import spacy
import nltk
from fuzzywuzzy import process
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import argparse

# Start timestamp
start_time = datetime.now()
print(f"Script started at: {start_time}")

nltk.download("wordnet")
nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")  # Model for biomedical NER

# Initialize NLTK Lemmatizer
lemmatizer = WordNetLemmatizer()

parser = argparse.ArgumentParser(description="Creates a Knowledge Graph as a CSV from HPRA drug leaflet text file.")
parser.add_argument("--input", required=True, help="Path to input .txt file")
parser.add_argument("--terms", required=True, help="Path to medical terms .txt file")
parser.add_argument("--output", required=True, help="Path to output .csv file")
args = parser.parse_args()

txt_file_path = args.input
medical_terms_file = args.terms
csv_output_path = args.output

# Load predefined medical terms for fuzzy matching
with open(medical_terms_file, "r", encoding="utf-8") as f:
    medical_terms = [line.strip().lower() for line in f.readlines()]

# Define relationship mappings
relationship_mappings = {
    "What is the name of the drug/medicine?": "HAS_NAME",
    "List the side-effects found in the leaflet as a comma-separated list of names only.": "HAS_SIDE_EFFECT",
    "List the active ingredient(s) found in the leaflet as a comma-separated list of names only.": "HAS_ACTIVE_INGREDIENT",
    "List the inactive ingredients found in the leaflet as a comma-separated list of names only.": "HAS_INACTIVE_INGREDIENT",
    "List the contraindications found in the leaflet as a comma-separated list of names only.": "HAS_CONTRAINDICATION",
    "List the warnings and precautions found in the leaflet as a comma-separated list of names only.": "HAS_WARNING",
    "Extract storage conditions into standardized format": "HAS_STORAGE_INFO",
    "Extract categorized dosage information": "HAS_DOSAGE_INFO",
    "Extract only the shape of the drug": "HAS_SHAPE",
    "Extract only the color of the drug": "HAS_COLOUR",
}

# Define units for standardization in dosage information
unit_mapping = {
    "milligram": "mg",
    "gram": "g",
    "liter": "L",
    "litre": "L",
    "microgram": "mcg",
    "nanogram": "ng",
    "kilogram": "kg",
}

# Function to normalize medical entities
def normalize_medical_entity(entity):
    doc = nlp_bc5cdr(entity.lower())
    if doc.ents:
        return doc.ents[0].text  
    return entity 

# Function to lemmatize words
def lemmatize_entity(entity):
    return lemmatizer.lemmatize(entity.lower())

# Function to apply fuzzy matching using the list of curated medical terms
def fuzzy_match_entity(entity):
    match, score = process.extractOne(entity, medical_terms)
    return match if score > 85 else entity  # Replace if confidence > 85%

# Function to standardize dosage units
def standardize_dosage(dosage):
    for long_unit, short_unit in unit_mapping.items():
        dosage = re.sub(rf"\b{long_unit}\b", short_unit, dosage, flags=re.IGNORECASE)
    return dosage

# Function to extract min and max dosage values
def extract_dosage_values(dosage_list):
    numeric_values = []
    contains_prescribed_text = False
    for dosage in dosage_list:
        dosage = standardize_dosage(dosage) 
        match = re.search(r"(\d+)\s*mg", dosage)  
        if match:
            numeric_values.append(int(match.group(1)))  
        elif "Dosage to be prescribed by doctor" in dosage:
            contains_prescribed_text = True
    if numeric_values:
        return [f"{min(numeric_values)} mg", f"{max(numeric_values)} mg"] if len(numeric_values) > 1 else [f"{numeric_values[0]} mg"]
    elif contains_prescribed_text:
        return ["Dosage to be prescribed by doctor"]
    return []


# Read the TXT file and extract relationships
data = []
current_drug = None
drug_name_mapping = {}  
with open(txt_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
for i, line in enumerate(lines):
    line = line.strip()
    if line.startswith("Drug Leaflet:"):
        current_pdf = line.split(":")[1].strip()  # Store PDF filename
        print(f"Processing drug leaflet: {current_pdf} at {datetime.now()}")  # Timestamp each leaflet start
    if line.startswith("Q: What is the name of the drug/medicine?"):
        answer_line = next((l.strip() for l in lines[i + 1:] if l.startswith("A: ")), None)
        if answer_line:
            current_drug = answer_line.replace("A: ", "").strip()
            drug_name_mapping[current_pdf] = current_drug  # Store drug name mapping

    # Extract entities & relationships
    for question, relation in relationship_mappings.items():
        if line.startswith(f"Q: {question}"):
            answer_line = next((l.strip() for l in lines[i + 1:] if l.startswith("A: ")), None)
            if answer_line:
                answer = answer_line.replace("A: ", "").strip()
                subject = drug_name_mapping.get(current_pdf, current_pdf)
                if relation == "HAS_DOSAGE_INFO":
                    dosage_lines = []
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith("- "):
                            dosage_lines.append(lines[j].strip().replace("- ", "").strip())
                        elif lines[j].strip().startswith("Q: "):
                            break
                    processed_dosages = extract_dosage_values(dosage_lines)
                    for dosage in processed_dosages:
                        data.append([subject, relation, dosage])
                elif relation == "HAS_STORAGE_INFO":
                    data.append([subject, relation, answer])
                elif relation == "HAS_COLOUR":
                    colors = [c.strip() for c in answer.split(",")]
                    for color in colors:
                        data.append([subject, relation, color])
                elif relation in {
                    "HAS_SIDE_EFFECT",
                    "HAS_ACTIVE_INGREDIENT",
                    "HAS_INACTIVE_INGREDIENT",
                    "HAS_CONTRAINDICATION",
                    "HAS_WARNING",
                }:
                    if "," in answer:
                        values = [v.strip() for v in answer.split(",")]
                        for value in values:
                            normalized = normalize_medical_entity(value)
                            lemmatized = lemmatize_entity(normalized)
                            matched = fuzzy_match_entity(lemmatized)
                            data.append([subject, relation, matched])
                    else:
                        normalized = normalize_medical_entity(answer)
                        lemmatized = lemmatize_entity(normalized)
                        matched = fuzzy_match_entity(lemmatized)
                        data.append([subject, relation, matched])
                else:
                    data.append([subject, relation, answer])


df = pd.DataFrame(data, columns=["Subject", "Predicate", "Object"])
df.drop_duplicates(inplace=True)
df.to_csv(csv_output_path, index=False)
end_time = datetime.now()

# Print summary
print(f"Script finished at: {end_time}")
print(f"Total execution time: {end_time - start_time}")
print(f"Knowledge Graph CSV saved as {csv_output_path}")
