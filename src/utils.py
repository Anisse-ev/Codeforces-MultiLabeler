import os
import json
import re
import glob
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from config.data_config import SELECTED_TAGS, TEXT_FEATURE_COLUMNS, CODE_FEATURE_COLUMNS

# --- Prompt Generation ---

def get_prompt_template():
    """
    Returns the base prompt template (instructions only) used to generate complete prompts.
    """
    string_formatted_tags = ', '.join([f'"{tag}"' for tag in SELECTED_TAGS])
    template_lines = [
        "You are given a programming problem with various details. Your task is to analyze the problem and determine which tags apply from the following list:",
        f"{', '.join(SELECTED_TAGS)}.",
        "",
        "For each tag, output 1 if the tag applies to the problem or 0 if it does not.",
        "It is important that you consider that none of the tags might apply, and that multiple tags may apply simultaneously.",
        "IMPORTANT: Your output must be a single valid JSON object, and it must be wrapped between <output> and </output>.",
        f"The JSON object must only contain the following keys: {string_formatted_tags}.",
        "Each key must have either the value 0 or 1.",
        "",
        "Below are the problem details:"
    ]
    return "\n".join(template_lines)

def generate_prompt(row: pd.Series) -> str:
    """
    Given a DataFrame row with the problem details, this function generates a prompt
    for the LLM to predict applicable programming problem tags in a strict JSON format,
    wrapped between <output> and </output>.
    """
    prompt_parts = [get_prompt_template()]

    # Add text feature columns to prompt.
    for column in TEXT_FEATURE_COLUMNS:
        if pd.notna(row.get(column)):
            feature_name = column.replace('_', ' ').capitalize()
            prompt_parts.append(f"{feature_name}: {row[column]}")

    # Add code feature columns to prompt.
    for column in CODE_FEATURE_COLUMNS:
        if pd.notna(row.get(column)):
            feature_name = column.replace('_', ' ').capitalize()
            prompt_parts.append(f"{feature_name}: {row[column]}")

    prompt_parts.append("")

    # Dynamically create an example based on SELECTED_TAGS.
    # Here we use an alternating pattern: 1 for even indices, 0 for odd indices.
    example_dict = { tag: (1 if i % 2 == 0 else 0) for i, tag in enumerate(SELECTED_TAGS) }
    example_output = json.dumps(example_dict)

    prompt_parts.append("Respond with only one JSON response wrapped in <output> and </output> with no additional text.")
    prompt_parts.append("For example:")
    prompt_parts.append(f"<output>{example_output}</output>")

    return "\n".join(prompt_parts)


# --- Model Output Parsing ---

def parse_model_output(output_text: str) -> np.ndarray:
    """
    Parses the model output to extract the JSON object enclosed between <output> and </output>.
    Returns a numpy array of predicted tag values, defaulting to 0s on failure.
    """
    try:
        # Extract content between <output> and </output>
        # More robust regex to handle potential leading/trailing whitespace and newlines
        match = re.search(r"<output>\s*(\{.*?\})\s*</output>", output_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_text = match.group(1)
            # Clean potential markdown code block markers if present
            json_text = re.sub(r'^```json\s*', '', json_text, flags=re.IGNORECASE)
            json_text = re.sub(r'\s*```$', '', json_text)
            json_data = json.loads(json_text)
            # Ensure all SELECTED_TAGS are present, default to 0 if missing
            predictions = [int(json_data.get(tag, 0)) for tag in SELECTED_TAGS]
            # Validate values are 0 or 1
            if not all(p in [0, 1] for p in predictions):
                 raise ValueError(f"Invalid prediction value found in {json_data}. Expected 0 or 1.")
        else:
            # Attempt to find JSON even without explicit tags as a fallback
            json_match = re.search(r'(\{.*?\})', output_text, re.DOTALL)
            if json_match:
                 print(f"Warning: <output> tags not found, attempting to parse JSON directly from: {output_text[:100]}...")
                 json_text = json_match.group(1)
                 json_data = json.loads(json_text)
                 predictions = [int(json_data.get(tag, 0)) for tag in SELECTED_TAGS]
                 if not all(p in [0, 1] for p in predictions):
                      raise ValueError(f"Invalid prediction value found in fallback JSON {json_data}. Expected 0 or 1.")
            else:
                 raise ValueError("No <output>...</output> tags or parsable JSON object found.")

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Error parsing model output: {e}. Output was: '{output_text[:200]}...'")
        predictions = [0] * len(SELECTED_TAGS) # Return default (all zeros) on any parsing error
    except Exception as e:
        print(f"Unexpected error parsing model output: {e}. Output was: '{output_text[:200]}...'")
        predictions = [0] * len(SELECTED_TAGS)
    return np.array(predictions)


# --- File/Directory Management ---

def get_incremental_directory_path(base_dir: str, prefix: str = "") -> str:
    """
    Creates and returns a directory path within base_dir named with
    today's date (YYYYMMDD) followed by an incremental index (XXX).
    Example: base_dir/YYYYMMDD_001/
    If prefix is provided: base_dir/prefix_YYYYMMDD_001/
    """
    today = datetime.today().strftime("%Y%m%d")
    pattern = os.path.join(base_dir, f"{prefix}{today}_*")
    existing_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    max_index = 0
    for dir_path in existing_dirs:
        try:
            base_name = os.path.basename(dir_path)
            # Extract index after the date part (and optional prefix)
            index_str = base_name.split("_")[-1]
            index = int(index_str)
            if index > max_index:
                max_index = index
        except (IndexError, ValueError):
            continue # Ignore directories not matching the pattern

    new_index = max_index + 1
    dir_name = f"{prefix}{today}_{new_index:03d}"
    full_path = os.path.join(base_dir, dir_name)

    os.makedirs(full_path, exist_ok=True)
    return full_path

def get_latest_subdirectory(base_dir: str) -> str | None:
    """
    Finds the subdirectory within base_dir that has the latest date and highest index
    based on the naming convention YYYYMMDD_XXX or prefix_YYYYMMDD_XXX.
    Returns the full path to the latest directory or None if no matching directories found.
    """
    pattern = os.path.join(base_dir, "*_*") # Match any prefix_date_index structure
    subdirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    latest_dir = None
    latest_date = datetime.min
    latest_index = -1

    for dir_path in subdirs:
        try:
            base_name = os.path.basename(dir_path)
            parts = base_name.split("_")
            if len(parts) < 2: # Need at least date and index
                continue

            index_str = parts[-1]
            date_str = parts[-2]

            # Check if the part before index looks like a date
            current_date = datetime.strptime(date_str, "%Y%m%d")
            current_index = int(index_str)

            if current_date > latest_date:
                latest_date = current_date
                latest_index = current_index
                latest_dir = dir_path
            elif current_date == latest_date and current_index > latest_index:
                latest_index = current_index
                latest_dir = dir_path

        except (ValueError, IndexError):
            # Handles cases where parsing fails (incorrect format)
            continue

    return latest_dir


# --- System ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")