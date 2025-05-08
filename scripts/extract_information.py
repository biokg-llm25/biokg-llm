"""
Script to extract structured medical Q&A pairs from HPRA drug leaflet PDFs using a locally or remotely hosted LLM.
It converts PDF text to plain text, then queries a chat-style language model to extract drug-related facts like name, side-effects, dosage, storage, appearance and so on.

Main Phases:
1. PDF to text extraction (with timeout handling)
2. LLM-powered extraction of structured drug attributes
3. Optional refinement using follow-up LLM queries (e.g., standardizing dosage or shape/color)
"""
# ------------------ Import necessary libraries ------------------
import os
import requests
import logging
import multiprocessing
from PyPDF2 import PdfReader
import time
import argparse

# ------------------ CLI Arguments ------------------
parser = argparse.ArgumentParser(description="Extract structured Q&A from HPRA drug leaflets using LLM.")
parser.add_argument("--pdf_dir", required=True, help="Directory containing the PDF files")
parser.add_argument("--output", required=True, help="Path to save the extracted Q&A .txt file")
parser.add_argument("--llm_url", required=True, help="URL of the LLM endpoint")
parser.add_argument("--llm_model", required=True, help="Name of the LLM model to be used")
args = parser.parse_args()

folder_path = args.pdf_dir
output_txt = args.output
LLM_URL = args.llm_url
LLM_MODEL = args.llm_model

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ------------------ Utility Functions ------------------
def is_already_processed(filename):
    """Check if a given PDF filename has already been processed and logged in the output file."""
    if os.path.exists(output_txt):
        with open(output_txt, "r", encoding="utf-8") as f:
            return f"Drug Leaflet: {filename}" in f.read()
    return False

def split_list_values(values):
    """Split a stringified list (comma-separated) into a Python list, cleaning brackets and quotes."""
    if isinstance(values, str):
        values = values.strip("[]").replace("'", "")
        return [val.strip() for val in values.split(',')]
    return values if isinstance(values, list) else [values]

def extract_text_worker(pdf_path, result_queue):
    """Worker function to extract text from a PDF using PyPDF2. Runs in a subprocess for timeout control."""
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("Extracted text is empty")
        result_queue.put(text)
    except Exception as e:
        result_queue.put(f"ERROR: {e}")

def extract_text_from_pdf(pdf_path, timeout=120):
    """Safely extract text from a PDF with a timeout limit to prevent hangs or crashes."""
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=extract_text_worker, args=(pdf_path, result_queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        logging.error(f"Timeout reached for {pdf_path}. Terminating process.")
        process.terminate()
        process.join()
        return None
    extracted_text = result_queue.get() if not result_queue.empty() else None
    if extracted_text and extracted_text.startswith("ERROR:"):
        logging.error(f"Error extracting text from {pdf_path}: {extracted_text[7:]}")
        return None
    return extracted_text

def query_model(context, question, filename, retries=3, delay=5):
    """Query the LLM with a given context and question. Retry on failure, log responses."""
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert in medical drug information extraction. Respond concisely."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        "max_tokens": 500,
    }
    for attempt in range(retries):
        try:
            logging.info(f"Attempt {attempt+1}: Querying LLM for file: {filename} | Question: '{question}'")
            start_time = time.time()
            response = requests.post(LLM_URL, json=data, timeout=30)
            duration = time.time() - start_time
            logging.info(f"LLM response received in {duration:.2f} seconds for {filename}")
            response.raise_for_status()
            time.sleep(1)
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", None)
        except requests.exceptions.Timeout:
            logging.error(f"Timeout on attempt {attempt+1} for {filename}. Retrying in {delay} sec...")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {filename} on attempt {attempt+1}: {e}")
        if attempt == retries - 1:
            return None

def extract_temperature_info(storage_text, filename):
    """Use the LLM to refine storage temperature from extracted storage text."""
    if isinstance(storage_text, str):
        refined_question = "From the extracted storage information, classify the storage condition only as '> X degrees', '< X degrees', 'X - Y degrees', or 'No special storage'. Extract and replace the actual numeric value of X or Y with the extracted value. Do not generate any extra information."
        refined_storage = query_model(storage_text, refined_question, filename)
        return refined_storage.strip() if refined_storage else "No special storage"
    return "No special storage"

def extract_dosage_info(dosage_text, filename):
    """Use the LLM to break down dosage into standardized formats for Adults, Children, Elderly."""
    if isinstance(dosage_text, str):
        refined_question = "From the extracted dosage information, extract and categorize doses as 'Adults (General)', 'Children', 'Elderly' only. Format as '<Category> - <X mg a day>' and return each dose in a separate line. Ignore subcategories. If no specific dosage amount in mg is available, return 'Dosage to be prescribed by doctor'. Do not generate any extra information."
        refined_dosage = query_model(dosage_text, refined_question, filename)
        if refined_dosage is None:
            return []
        return refined_dosage.split('\n') if refined_dosage else []
    return []

def extract_appearance_info(appearance_text, filename):
    """Use the LLM to extract shape and color of the drug from appearance description."""
    if isinstance(appearance_text, str):
        shape_question = "From the extracted drug appearance description, extract only the shape of the drug as a single word. For multiple shapes list only the name of the shapes separated by comma. Do not generate any extra information."
        color_question = "From the extracted drug appearance description, extract only the color of the drug as a single word. For multiple colours list only the name of the colours separated by comma. Do not generate any extra information."
        shape = query_model(appearance_text, shape_question, filename)
        color = query_model(appearance_text, color_question, filename)
        return split_list_values(shape) if shape else [], split_list_values(color) if color else []
    return [], []

def process_pdfs(folder_path):
    """Main function: loops through PDFs, extracts Q&A info via LLM, and writes results to .txt."""
    questions = [
        ("Drug", "HAS_NAME", "What is the name of the drug/medicine? Provide only the name."),
        ("Drug", "HAS_SIDE_EFFECT", "List the side-effects found in the leaflet as a comma-separated list of names only."),
        ("Drug", "HAS_ACTIVE_INGREDIENT", "List the active ingredient(s) found in the leaflet as a comma-separated list of names only."),
        ("Drug", "HAS_INACTIVE_INGREDIENT", "List the inactive ingredients found in the leaflet as a comma-separated list of names only."),
        ("Drug", "HAS_CONTRAINDICATION", "List the contraindications found in the leaflet as a comma-separated list of names only."),
        ("Drug", "HAS_WARNING", "List the warnings and precautions found in the leaflet as a comma-separated list of names only."),
        ("Drug", "HAS_STORAGE_INFO", "Extract the storage conditions as a paragraph."),
        ("Drug", "HAS_DOSAGE_INFO", "Extract dosage instructions as a paragraph."),
        ("Drug", "HAS_APPEARANCE", "Extract how the drug looks as a paragraph."),
    ]

    with open(output_txt, mode='a', encoding='utf-8') as txtfile:
        for filename in sorted([f for f in os.listdir(folder_path) if f.endswith(".pdf")]):
            if is_already_processed(filename):
                logging.info(f"Skipping already processed file: {filename}")
                continue

            pdf_path = os.path.join(folder_path, filename)
            logging.info(f"Processing file: {pdf_path}")

            try:
                context = extract_text_from_pdf(pdf_path)
                if not context:
                    raise ValueError("Empty text extracted.")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue

            txtfile.write("=" * 40 + "\n")
            txtfile.write(f"Drug Leaflet: {filename}\n")
            txtfile.write("=" * 40 + "\n\n")

            drug_name = query_model(context, "What is the name of the drug/medicine? Provide only the name.", filename) or "Drug name not found"
            txtfile.write(f"Q: What is the name of the drug/medicine?\n")
            txtfile.write(f"A: {drug_name}\n\n")

            for _, predicate, question in questions:
                if predicate == "HAS_NAME":
                    continue

                logging.info(f"Querying LLM for file: {filename} | Question: '{question}'")
                answer = query_model(context, question, filename)

                if answer is None:
                    logging.error(f"Skipping question for {filename} due to API failure: '{question}'")
                    txtfile.write(f"Q: {question}\nA: [No response due to API timeout]\n\n")
                    continue

                txtfile.write(f"Q: {question}\n")
                txtfile.write(f"A: {answer}\n\n")

                if predicate == "HAS_STORAGE_INFO":
                    refined_storage = extract_temperature_info(answer, filename)
                    txtfile.write(f"Q: Extract storage conditions into standardized format\n")
                    txtfile.write(f"A: {refined_storage if refined_storage else '[No response]'}\n\n")

                elif predicate == "HAS_DOSAGE_INFO":
                    dosage_list = extract_dosage_info(answer, filename)
                    if not dosage_list:
                        txtfile.write(f"Q: Extract categorized dosage information\nA: [No response]\n\n")
                    else:
                        txtfile.write(f"Q: Extract categorized dosage information\nA:\n")
                        for dosage in dosage_list:
                            txtfile.write(f"- {dosage}\n")
                        txtfile.write("\n")

                elif predicate == "HAS_APPEARANCE":
                    shapes, colors = extract_appearance_info(answer, filename)
                    txtfile.write(f"Q: Extract only the shape of the drug\n")
                    txtfile.write(f"A: {', '.join(shapes) if shapes else '[No response]'}\n\n")
                    txtfile.write(f"Q: Extract only the color of the drug\n")
                    txtfile.write(f"A: {', '.join(colors) if colors else '[No response]'}\n\n")

            txtfile.write("\n\n")
            logging.info("Sleeping for 5 seconds before processing next PDF...")
            time.sleep(5)

    processed_count = sum(1 for line in open(output_txt, encoding="utf-8") if "Drug Leaflet:" in line)
    total_pdfs = len([f for f in os.listdir(folder_path) if f.endswith(".pdf")])
    skipped_count = total_pdfs - processed_count

    logging.info(f"Processing complete. Total PDFs processed: {processed_count}, Skipped: {skipped_count}")
    logging.info(f"Results saved to {output_txt}")

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    process_pdfs(folder_path)
