# Data Processing Scripts

This folder processes the Codeforces dataset through three main steps: download, clean, and split into train/test sets.

## Overview

1. **Download Raw Data:**  
   Retrieves a ZIP file from Google Drive, extracts JSON files, and stores them in the raw data directory.
2. **Clean Data:**  
   Loads raw JSON files, applies transformations (column renaming, tag extraction, time/memory conversions, etc.), and saves a cleaned CSV file. The file is named using today’s date, with an incrementing suffix if needed.
3. **Train-Test Split:**  
   Splits the cleaned data chronologically into training and test datasets and also provides a zero-shot set (the complete dataset). By default, the script uses the latest CSV file, but you can also specify a particular file.

## Scripts

- **load_raw_data.py**

  - **Purpose:** Downloads and extracts raw JSON files.
  - **Usage:**
    ```bash
    python3 -m load_raw_data.py
    ```

- **prepare_clean_data.py**

  - **Purpose:** Cleans the raw data and saves a CSV file into `CLEAN_DATA_DIR` with a date-based filename.
  - **Usage:**
    ```bash
    python3 -m prepare_clean_data.py
    ```

- **train_test_split.py**
  - **Purpose:** Splits the cleaned CSV into training, testing, and zero-shot datasets.
  - **Usage:**
    - Default (latest file):
      ```bash
      python3 -m train_test_split.py
      ```
    - Specific file:
      ```bash
      python3 -m train_test_split.py --file your_file_name.csv
      ```

## Configuration

- **Paths and Data Settings:**
  - All directory paths and data configurations (like column renaming, selected tags, and test set proportion) are defined in `config/links_and_paths.py` and `config/data_config.py`.
- **Important Columns:**
  - The `creation_order` column is generated to maintain chronological order during the split.
