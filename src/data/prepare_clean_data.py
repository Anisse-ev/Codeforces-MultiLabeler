import json
import os
import re
import pandas as pd
import datetime

from config.links_and_paths import RAW_DATA_DIR, CLEAN_DATA_DIR
from config.data_config import (COLUMNS_RENAMING,
                                    SELECTED_COLUMNS,
                                    SELECTED_TAGS,
                                    MINIMUM_TAGS,
                                    COLUMNS_TO_LOWERCASE,
                                    CREATION_TIME_COLUMN)


def _parse_time_limit(time_str):
    """
    Convert a time limit string (e.g., "1 second") to a float in seconds.
    Handles cases like "2 seconds", "1.5 sec", etc.
    """
    if pd.isna(time_str):
        return None
    match = re.search(r"([\d\.]+)", time_str)
    if match:
        return float(match.group(1))
    return None

def _parse_memory_limit(mem_str):
    """
    Convert a memory limit string (e.g., "256 megabytes") to a float in MB.
    Handles cases like "1024 MB", "512 megabytes", etc.
    """
    if pd.isna(mem_str):
        return None
    match = re.search(r"([\d\.]+)", mem_str)
    if match:
        return float(match.group(1))
    return None

def _process_json_file(file_path):
    """
    Load a JSON file and return a DataFrame. Returns an empty DataFrame on failure.
    """
    try:
        with open(file_path, 'r') as f:
            data = pd.json_normalize(json.load(f))
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def _load_raw_data(raw_dir):
    """
    Load and concatenate all JSON files found in raw_dir.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory '{raw_dir}' does not exist.")
    
    json_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in '{raw_dir}'.")
    
    # Use a list comprehension to process files and concatenate the results.
    data_frames = [_process_json_file(file) for file in json_files]
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    print(f"Total records loaded: {len(combined_data)}")
    return combined_data
    

def clean_data(raw_dir, output_dir):
    # Load data from all JSON files in the provided raw directory
    codeforce_df = _load_raw_data(raw_dir)
    
    # Rename columns using COLUMNS_RENAMING
    codeforce_df.rename(columns=COLUMNS_RENAMING, inplace=True)

    columns_to_keep = SELECTED_COLUMNS + [CREATION_TIME_COLUMN] 

    # Keep only SELECTED_COLUMNS (if they exist in the dataframe) plus the "tags" column if available
    available_cols = [col for col in columns_to_keep if col in codeforce_df.columns]
    codeforce_df = codeforce_df[ available_cols + (["tags"] if "tags" in codeforce_df.columns else []) ]
    
    # Create new binary columns for each tag in SELECTED_TAGS based on the original "tags" column.
    for tag in SELECTED_TAGS:
        codeforce_df[tag] = codeforce_df.get("tags", pd.Series([[]] * len(codeforce_df))).apply(lambda x: 1 if isinstance(x, list) and tag in x else 0)

    # Transform time_limit and memory_limit columns if present.
    if "time_limit" in codeforce_df.columns:
        codeforce_df["time_limit"] = codeforce_df["time_limit"].apply(_parse_time_limit)
        parsed_time = codeforce_df["time_limit"].notna().sum()
        print(f"Successfully parsed time limits for {parsed_time} rows.")
    if "memory_limit" in codeforce_df.columns:
        codeforce_df["memory_limit"] = codeforce_df["memory_limit"].apply(_parse_memory_limit)
        parsed_memory = codeforce_df["memory_limit"].notna().sum()
        print(f"Successfully parsed memory limits for {parsed_memory} rows.")

    # Lowercase specified text columns (if they exist) and transform lists to string
    for col in codeforce_df.columns:
        if codeforce_df[col].apply(lambda x: isinstance(x, list)).any():
            codeforce_df[col] = codeforce_df[col].apply(str)
        if col in COLUMNS_TO_LOWERCASE:
            codeforce_df[col] = codeforce_df[col].astype(str).str.lower()

    # Drop duplicate rows
    codeforce_df.drop_duplicates(inplace=True)

    # Add creation_order column: rank rows by CREATION_TIME_COLUMN
    if CREATION_TIME_COLUMN in codeforce_df.columns:
        codeforce_df.sort_values(CREATION_TIME_COLUMN, inplace=True)
        codeforce_df["creation_order"] = range(1, len(codeforce_df) + 1)
    else:
        codeforce_df["creation_order"] = range(1, len(codeforce_df) + 1)

    # Filtering: only keep rows that have at least MINIMUM_TAGS of the SELECTED_TAGS present.
    codeforce_df['selected_tags_count'] = codeforce_df[SELECTED_TAGS].sum(axis=1)
    codeforce_df = codeforce_df[codeforce_df['selected_tags_count'] >= MINIMUM_TAGS]
    codeforce_df.drop(columns=["selected_tags_count"], inplace=True)
    codeforce_df = codeforce_df.reset_index(drop=True)
    

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate filename with today's date
    today_str = datetime.date.today().isoformat()
    base_filename = f"codeforces_clean_data_{today_str}"
    csv_filename = base_filename + ".csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # Check if the file exists and add _1, _2, ... if needed
    counter = 1
    while os.path.exists(csv_path):
        csv_filename = f"{base_filename}_{counter}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        counter += 1

    # Save the cleaned data
    codeforce_df.to_csv(csv_path, index=False)
    print(f"Clean data saved to {csv_path}")

    # Display missing value stats per column
    print(f"Missing values per column:")
    print(codeforce_df.isna().sum())

    # Tag distribution
    print("\nTag distribution:")
    for tag in SELECTED_TAGS:
        count = codeforce_df[tag].sum()
        percentage = count / len(codeforce_df) * 100
        print(f"  {tag}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    clean_data(RAW_DATA_DIR, CLEAN_DATA_DIR)

