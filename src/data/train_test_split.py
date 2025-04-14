import pandas as pd
import os
import argparse
from config.links_and_paths import TRAIN_TEST_DATA_FILE_NAMES, TRAIN_SPLIT_DATA_DIR, CLEAN_DATA_DIR
from config.data_config import SELECTED_TAGS, TEST_SET_SIZE

def get_latest_csv(directory):
    """
    Returns the latest CSV file in the given directory.
    """
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{directory}'.")
    latest_file = max(csv_files, key=os.path.getctime)
    return latest_file

def analyze_train_test_split(csv_path, test_proportion):
    """
    Split the dataset chronologically by creation_order and analyze tag distribution.
    
    Args:
        csv_path: Path to the clean CSV file.
        test_proportion: Proportion of data to use for testing (most recent data).
    
    Returns:
        train_codeforces: DataFrame for the training set.
        test_codeforces: DataFrame for the test set.
        codeforces: DataFrame of the whole dataset.
    """
    print(f"Loading data from {csv_path}")
    codeforces = pd.read_csv(csv_path)
    
    # Check for creation_order column or create one from the index.
    if 'creation_order' not in codeforces.columns:
        print("Warning: 'creation_order' column not found. Creating one based on index.")
        codeforces['creation_order'] = range(1, len(codeforces) + 1)
    
    # Ensure data is sorted by creation_order for chronological splitting.
    codeforces = codeforces.sort_values('creation_order')
    
    # Calculate the split index
    split_idx = int(len(codeforces) * (1 - test_proportion))
    
    # Split the data
    train_codeforces = codeforces.iloc[:split_idx].copy()
    test_codeforces = codeforces.iloc[split_idx:].copy()
    
    # Dataset size and tag distributions
    print(f"Total dataset size: {len(codeforces)} examples")
    print(f"Training set size: {len(train_codeforces)} examples ({len(train_codeforces)/len(codeforces):.2%})")
    print(f"Test set size: {len(test_codeforces)} examples ({len(test_codeforces)/len(codeforces):.2%})")
    print("\nTag distributions:")
    
    print(f"{'Tag':<15} {'Train %':<10} {'Test %':<10} {'Difference':<10}")
    print("-" * 45)
    
    tag_diffs = []
    
    for tag in SELECTED_TAGS:
        # Determine column name in the CSV: either the tag itself or a modified version.
        tag_col = tag if tag in codeforces.columns else f"tag_{tag.replace(' ', '_')}"
        if tag_col not in codeforces.columns:
            print(f"Warning: Column for tag '{tag}' not found in dataset!")
            continue
        
        train_pct = train_codeforces[tag_col].mean() * 100
        test_pct = test_codeforces[tag_col].mean() * 100
        relative_diff = 2 * (test_pct - train_pct)
        
        tag_diffs.append((tag, train_pct, test_pct, relative_diff))
        
        print(f"{tag:<15} {train_pct:>8.2f}% {test_pct:>8.2f}% {relative_diff:>+10.2f}%")
    
    if tag_diffs:
        all_diffs = [abs(diff) for _, _, _, diff in tag_diffs]
        avg_diff = sum(all_diffs) / len(all_diffs)
        max_diff = max(all_diffs)
        max_diff_idx = all_diffs.index(max_diff)
        max_diff_tag, max_diff_train, max_diff_test, max_diff_value = tag_diffs[max_diff_idx]
        
        print(f"\nAverage absolute difference: {avg_diff:.2f}%")
        print(f"Biggest difference: {max_diff:.2f}% for tag '{max_diff_tag}'")
        print(f"  - Training: {max_diff_train:.2f}%")
        print(f"  - Testing: {max_diff_test:.2f}%")
    
    return train_codeforces, test_codeforces, codeforces

def main():
    parser = argparse.ArgumentParser(
        description="Analyze train-test split based on the clean CSV file."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Specific CSV file name in CLEAN_DATA_DIR to use. If not provided, the latest file will be used."
    )
    args = parser.parse_args()
    
    if args.file:
        file_path = os.path.join(CLEAN_DATA_DIR, args.file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Specified file '{args.file}' not found in '{CLEAN_DATA_DIR}'.")
    else:
        file_path = get_latest_csv(CLEAN_DATA_DIR)
    
    # Analyze the split
    train_codeforces, test_codeforces, codeforces = analyze_train_test_split(file_path, TEST_SET_SIZE)
    print("\nSaving train, test, and zero-shot sets...")

    # Ensure the output directory exists
    if not os.path.exists(TRAIN_SPLIT_DATA_DIR):
        os.makedirs(TRAIN_SPLIT_DATA_DIR)
    
    # Save the train, test and entire (zero-shot) datasets.
    train_path = os.path.join(TRAIN_SPLIT_DATA_DIR, TRAIN_TEST_DATA_FILE_NAMES["train"])
    test_path = os.path.join(TRAIN_SPLIT_DATA_DIR, TRAIN_TEST_DATA_FILE_NAMES["test"])
    zero_shot_path = os.path.join(TRAIN_SPLIT_DATA_DIR, TRAIN_TEST_DATA_FILE_NAMES["zero_shot"])
    
    train_codeforces.to_csv(train_path, index=False)
    test_codeforces.to_csv(test_path, index=False)
    codeforces.to_csv(zero_shot_path, index=False)
    
    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    print(f"Zero-shot data saved to: {zero_shot_path}")

if __name__ == "__main__":
    main()
