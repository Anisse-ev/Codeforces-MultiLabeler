import json
import os
import re
import pandas as pd
import datetime
from typing import List
from config.links_and_paths import RAW_DATA_DIR, CLEAN_DATA_DIR
from config.data_config import (COLUMNS_RENAMING,
                                SELECTED_COLUMNS,
                                SELECTED_TAGS,
                                MINIMUM_TAGS,
                                COLUMNS_TO_LOWERCASE,
                                CREATION_TIME_COLUMN,
                                TEXT_FEATURE_COLUMNS,
                                CODE_FEATURE_COLUMNS,
                                PROBLEM_ID_COLUMN,)

# Concatenate instruction columns from text and code features.
instruction_columns = TEXT_FEATURE_COLUMNS + CODE_FEATURE_COLUMNS

# --- Utility functions for parsing limits ---
def _parse_time_limit(time_str: str) -> float:
    if pd.isna(time_str):
        return None
    match = re.search(r"([\d\.]+)", time_str)
    if match:
        return float(match.group(1))
    return None

def _parse_memory_limit(mem_str: str) -> float:
    if pd.isna(mem_str):
        return None
    match = re.search(r"([\d\.]+)", mem_str)
    if match:
        return float(match.group(1))
    return None

# --- Data Loading Functions ---
def _process_json_file(file_path: str) -> pd.DataFrame:
    """
    Load a JSON file and return a DataFrame.
    Returns an empty DataFrame on failure.
    """
    try:
        with open(file_path, 'r') as f:
            data = pd.json_normalize(json.load(f))
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all JSON files from raw_dir.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory '{raw_dir}' does not exist.")
    
    json_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in '{raw_dir}'.")
    
    data_frames = [_process_json_file(file) for file in json_files]
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    print(f"Total records loaded: {len(combined_data)}")
    return combined_data

# --- Cleaning Functions ---
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by:
     - Renaming columns.
     - Keeping only selected columns.
     - Creating binary tag columns.
     - Parsing time and memory limits.
     - Converting lists to strings and lowercasing.
     - Dropping duplicates.
     - Sorting by creation time.
     - Filtering rows with sufficient tag counts.
    """
    # Rename columns using COLUMNS_RENAMING
    df.rename(columns=COLUMNS_RENAMING, inplace=True)
    
    # Keep only SELECTED_COLUMNS and optionally the "tags" column if available
    columns_to_keep = SELECTED_COLUMNS + [CREATION_TIME_COLUMN]
    available_cols = [col for col in columns_to_keep if col in df.columns]
    df = df[ available_cols + (["tags"] if "tags" in df.columns else []) ]
    
    # Create binary columns for each tag in SELECTED_TAGS based on the original "tags" column.
    for tag in SELECTED_TAGS:
        df[tag] = df.get("tags", pd.Series([[]] * len(df))).apply(
            lambda x: 1 if isinstance(x, list) and tag in x else 0
        )
    
    # Parse time_limit and memory_limit if present.
    if "time_limit" in df.columns:
        df["time_limit"] = df["time_limit"].apply(_parse_time_limit)
        parsed_time = df["time_limit"].notna().sum()
        print(f"Successfully parsed time limits for {parsed_time} rows.")
    if "memory_limit" in df.columns:
        df["memory_limit"] = df["memory_limit"].apply(_parse_memory_limit)
        parsed_memory = df["memory_limit"].notna().sum()
        print(f"Successfully parsed memory limits for {parsed_memory} rows.")
    
    # Convert list columns to strings and lowercase specified text columns.
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(str)
        if col in COLUMNS_TO_LOWERCASE:
            df[col] = df[col].astype(str).str.lower()
    
    # Drop duplicate rows.
    df.drop_duplicates(inplace=True)
    
    # Sort by creation time if available and add a creation_order column.
    if CREATION_TIME_COLUMN in df.columns:
        df.sort_values(CREATION_TIME_COLUMN, inplace=True)
    df["creation_order"] = range(1, len(df) + 1)
    
    # Filter rows: only keep those with at least MINIMUM_TAGS from SELECTED_TAGS.
    df['selected_tags_count'] = df[SELECTED_TAGS].sum(axis=1)
    df = df[df['selected_tags_count'] >= MINIMUM_TAGS]
    df.drop(columns=["selected_tags_count"], inplace=True)
    df = df.reset_index(drop=True)
    
    return df

# --- Prompt and Output Generation Functions ---

def build_prompt_template(selected_tags: List[str]) -> str:
    """
    Build the base prompt template using the selected tags.
    This template includes instructions and the list of tags.
    """
    # Create a string with tags wrapped in quotes for the JSON keys line.
    string_formatted_tags = ", ".join([f'"{tag}"' for tag in selected_tags])
    
    template = (
        "You are given a programming problem with various details. Your task is to analyze the problem and determine which tags apply from the following list:\n"
        f"{', '.join(selected_tags)}.\n\n"
        "For each tag, output 1 if the tag applies to the problem or 0 if it does not.\n"
        "It is important that you consider that none of the tags might apply, and that multiple tags may apply simultaneously.\n"
        "IMPORTANT: Your output must be a single valid JSON object, and it must be wrapped between <output> and </output>.\n"
        f"The JSON object must only contain the following keys: {string_formatted_tags}.\n"
        "Each key must have either the value 0 or 1.\n\n"
        "Below are the problem details:"
    )
    return template

def generate_prompt_from_row(row: pd.Series, columns_for_instruction: List[str], selected_tags: List[str]) -> str:
    """
    Given a DataFrame row with problem details and a list of columns to use as instruction details,
    this function generates a complete prompt that includes:
      - A base template with instructions (using build_prompt_template)
      - The details extracted from the specified columns in the row
      - An example JSON output wrapped between <output> and </output>
    """
    prompt_parts = [build_prompt_template(selected_tags)]
    
    # Append each instruction detail from the row.
    for col in columns_for_instruction:
        if pd.notna(row.get(col)):
            # Replace underscores with spaces and capitalize the feature name.
            feature_name = col.replace("_", " ").capitalize()
            prompt_parts.append(f"{feature_name}: {row[col]}")
    
    prompt_parts.append("")  # Add an empty line for separation.
    
    # Create a dynamic example where even-indexed tags are marked as applied (1) and odd-indexed as not (0).
    example_dict = {tag: (1 if i % 2 == 0 else 0) for i, tag in enumerate(selected_tags)}
    example_output = json.dumps(example_dict)
    
    prompt_parts.append("Respond with only one JSON response wrapped in <output> and </output> with no additional text.")
    prompt_parts.append("For example:")
    prompt_parts.append(f"<output>{example_output}</output>")
    
    return "\n".join(prompt_parts)
def generate_output_from_row(row: pd.Series, selected_tags: List[str]) -> str:
    output_dict = {tag: int(row.get(tag, 0)) for tag in selected_tags}
    json_output = json.dumps(output_dict)
    return f"<output>{json_output}</output>"

def add_prompt_and_output_columns(df: pd.DataFrame,
                                  columns_for_instruction: List[str],
                                  selected_tags: List[str]) -> pd.DataFrame:
    """
    Add prompt and output columns to the DataFrame.
    'prompt' is built from the instruction columns.
    'output' is a JSON wrapped in <output> tags for the tag indicators.
    """
    prompts = []
    outputs = []
    
    for _, row in df.iterrows():
        prompt = generate_prompt_from_row(row, columns_for_instruction, selected_tags)
        output = generate_output_from_row(row, selected_tags)
        prompts.append(prompt)
        outputs.append(output)
    
    df = df.copy()
    df['prompt'] = prompts
    df['output'] = outputs
    return df

# --- Saving Function ---
def save_dataframe_with_unique_name(df: pd.DataFrame, output_dir: str, base_prefix: str = "codeforces_clean_data") -> str:
    """
    Save the DataFrame as a CSV file in the specified output directory.
    Uses today's date and a counter to ensure the filename is unique.
    Returns the path to the saved file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    today_str = datetime.date.today().isoformat()
    base_filename = f"{base_prefix}_{today_str}"
    csv_filename = base_filename + ".csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    counter = 1
    while os.path.exists(csv_path):
        csv_filename = f"{base_filename}_{counter}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        counter += 1
    
    df.to_csv(csv_path, index=True, index_label=PROBLEM_ID_COLUMN)
    print(f"Final cleaned and preprocessed data saved to {csv_path}")
    return csv_path

# --- Main Integrated Pipeline ---
def run_pipeline(raw_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Run the full pipeline:
       1. Load raw data.
       2. Clean the data.
       3. Add prompt and output columns.
       4. Save the result as a single CSV file.
    Returns the final DataFrame.
    """
    # Step 1: Load data from JSON files.
    raw_df = load_raw_data(raw_dir)
    
    # Step 2: Clean the DataFrame.
    clean_df = clean_dataframe(raw_df)
    
    # Step 3: Add prompt and output columns.
    full_df = add_prompt_and_output_columns(clean_df, instruction_columns, SELECTED_TAGS)
    
    # Step 4: Save the final DataFrame.
    save_dataframe_with_unique_name(full_df, output_dir)
    
    # Optional diagnostics.
    print("Missing values per column:")
    print(full_df.isna().sum())
    
    print("\nTag distribution:")
    for tag in SELECTED_TAGS:
        count = full_df[tag].sum()
        percentage = (count / len(full_df) * 100) if len(full_df) else 0
        print(f"  {tag}: {count} ({percentage:.2f}%)")
    
    return full_df

if __name__ == "__main__":
    # Execute the pipeline: load, clean, generate prompts/outputs, and save the final CSV.
    final_df = run_pipeline(RAW_DATA_DIR, CLEAN_DATA_DIR)
