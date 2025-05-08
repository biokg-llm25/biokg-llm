"""
Script that performs final post-processing on a Knowledge Graph (KG) CSV file 
generated from structured drug information extracted from HPRA leaflets.

It runs in two main phases:
1. Cleans and expands comma-separated values from the "Object" column
   into separate rows (e.g., converts "nausea, headache" into two rows) and also removes any extra stop-words.
2. Uses a locally or remotely hosted LLM to shorten long biomedical terms
   (e.g., "high blood pressure causing persistent headache" → "hypertensive headache").

---
Run from the command line:

    python postprocess_kg.py \
        --input <path_to_input_csv> \
        --final_output <path_to_final_output_csv> \
        --log_dir <path_to_log_directory> \
        --llm_url <llm_api_url> \
        --llm_model <model_name>

OPTIONAL:
    --mid_output: Path to save the intermediate cleaned CSV (comma-split version).
                  If not provided, it will be auto-generated from the input filename.

ARGUMENTS:
    --input         Path to the raw KG CSV file (from Step 5)
    --final_output  Output path to save the final cleaned KG
    --log_dir       Directory to store logs; will be created if not present
    --llm_url       URL to your LLM endpoint (e.g., a local server or cloud API)
    --llm_model     Name of the LLM model to use for entity shortening

NOTE:
Models like LLaMA 3 via Ollama can be used or connect to
other local or cloud-based models by providing the appropriate URL and model name.

"""
# ------------------ Install necessary libraries ------------------
import pandas as pd
import csv
import argparse
import requests
import time
import os

# ------------------ Argparse for Reproducibility ------------------
parser = argparse.ArgumentParser(description="Post-process HPRA KG: clean comma-separated values and shorten long entities using LLM.")
parser.add_argument("--input", required=True, help="Path to input KG CSV (e.g., hpra_complete_network.csv)")
parser.add_argument("--final_output", required=True, help="Path to final cleaned CSV (e.g., hpra_final_network.csv)")
parser.add_argument("--log_dir", required=True, help="Directory to store log files")
parser.add_argument("--llm_url", required=True, help="URL of the LLM endpoint")
parser.add_argument("--llm_model", required=True, help="Name of the model to be used")
parser.add_argument("--mid_output", required=False, help="(Optional) Path to intermediate cleaned CSV")
args = parser.parse_args()

# ------------------ Phase 1: Expand comma-separated values ------------------
csv_path = args.input
mid_output = args.mid_output or os.path.splitext(csv_path)[0] + "_intermediate.csv"  # Auto-generate if not supplied

df = pd.read_csv(csv_path, dtype=str, quotechar='"')
object_column = "Object"

def clean(df, object_col):
    cleaned_data = []
    for _, row in df.iterrows():
        subject, predicate, obj = row["Subject"], row["Predicate"], str(row[object_col]).strip()
        if obj.startswith('"') and obj.endswith('"'):
            obj = obj[1:-1]
        values = [v.strip() for v in obj.split(",") if v.strip()]
        for value in values:
            cleaned_data.append([subject, predicate, value])
    return pd.DataFrame(cleaned_data, columns=["Subject", "Predicate", object_col])

df_clean = clean(df, object_column)
df_clean.to_csv(mid_output, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"Processed knowledge graph saved at {mid_output}")

# ------------------ Phase 2: Shorten long entities using LLM ------------------
log_dir = args.log_dir
cleaned_csv_path = args.final_output
llm_url = args.llm_url
llm_model = args.llm_model

df = pd.read_csv(mid_output)
columns_to_clean = ["Subject", "Object"]
word_threshold = 3

os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"kg_cleaning_log_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt")
entity_cache = {}

def log_message(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)

def query_model(context, question, retries=3, delay=5):
    data = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": "You are an expert in biomedical terminology and drug information extraction. Ensure responses are concise (2-3 words), standardized, and medically accurate."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        "max_tokens": 50,
    }
    for attempt in range(retries):
        try:
            response = requests.post(llm_url, json=data)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()
            entity_cache[context] = result
            log_message(f"API Success: {context} → {result}")
            return result
        except requests.exceptions.RequestException as e:
            log_message(f"API Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    log_message(f"API Failed after {retries} retries: Returning original entity → {context}")
    return context

def shorten_entity(row, col_name, index):
    entity = row[col_name]
    if entity in entity_cache:
        log_message(f"Using Cached Response for: {entity} → {entity_cache[entity]}")
        return entity_cache[entity]
    if isinstance(entity, str) and len(entity.split()) > word_threshold:
        question = f"Convert the following medical term into a concise 2-3 word phrase while preserving its meaning: '{entity}'"
        if index % 10 == 0:
            log_message(f"Processing row {index}/{len(df)}: {entity}")
        return query_model(entity, question)
    return entity

start_time = time.time()
log_message("Starting entity cleaning process.")
for col in columns_to_clean:
    df[col] = df.apply(lambda row: shorten_entity(row, col, row.name), axis=1)

df.to_csv(cleaned_csv_path, index=False)
end_time = time.time()
elapsed_time = end_time - start_time

log_message(f"Post-processing complete. Cleaned knowledge graph saved at {cleaned_csv_path}")
log_message(f"Total execution time: {elapsed_time:.2f} seconds (~{elapsed_time/60:.2f} minutes)")
log_message(f"Log file saved at {log_file_path}")

