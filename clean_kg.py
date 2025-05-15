# Import necessary libraries
import pandas as pd
import re
import os
import argparse

def clean_kg(input_path, output_path):
    """
    Cleans the KG by removing nulls, placeholder entries,
    noisy patterns, short meaningless objects, and duplicates.

    Parameters:
    - input_path (str): Path to input KG CSV file
    - output_path (str): Path to save the cleaned KG CSV
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load dataset
    df = pd.read_csv(input_path)

    # Step 1: Remove null 'Subject' or 'Object'
    df = df[~(df['Subject'].isna() | df['Object'].isna())]

    # Step 2: Remove known noisy values from 'Subject' and 'Object'
    placeholder_start_pattern = (
        r"^(data not|none|unknown|not found|not given|not available|"
        r"drug name not found|not provided|not|name not given|"
        r"inadequate data|insufficient data)\b"
    )
    mask_placeholder = (
        df['Subject'].str.contains(placeholder_start_pattern, case=False, na=False) |
        df['Object'].str.contains(placeholder_start_pattern, case=False, na=False)
    )
    df = df[~mask_placeholder]

    # Step 3: Remove noisy values in 'Object' column
    object_contains_pattern = (
        r"(unknown|inadequate data|insufficient data|no colour|no shape|"
        r"no warning|no contraindication|no data|no information)\b"
    )
    df = df[~df['Object'].str.contains(object_contains_pattern, case=False, na=False)]

    # Step 4: Remove noisy values in 'Subject' column
    subject_flag_phrases = ['product name', 'not stated', 'invented name', 'unreadable text']
    subject_flag_pattern = '|'.join([re.escape(p) for p in subject_flag_phrases])
    df = df[~df['Subject'].str.contains(subject_flag_pattern, case=False, na=False)]

    # Step 5: Clean special characters
    def clean_specials(text):
        if isinstance(text, str):
            return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text.strip())
        return text

    df['Subject'] = df['Subject'].apply(clean_specials)
    df['Object'] = df['Object'].apply(clean_specials)

    # Step 6: Drop duplicates
    df = df.drop_duplicates()

    # Step 7: Remove specific unwanted values
    target_values = {'c', 'ig', 'na'}
    df = df[~df['Object'].astype(str).str.strip().str.lower().isin(target_values)]

    # Save cleaned file
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleaning script for MEDAKA.")
    parser.add_argument("--input", required=True, help="Path to the input file.")
    parser.add_argument("--output", required=True, help="Path to save the cleaned file.")

    args = parser.parse_args()
    clean_kg(args.input, args.output)
