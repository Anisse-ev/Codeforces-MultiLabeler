import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Attempt to import config, handle if running script directly vs as module
try:
    from config.links_and_paths import CLEAN_DATA_DIR
    from config.data_config import SELECTED_TAGS, OTHER_FEATURE_COLUMNS
except ModuleNotFoundError:
    print("Warning: Could not import from config files. Defining defaults locally.")
    # Define fallbacks if run standalone - adjust paths as needed
    CLEAN_DATA_DIR = "../../data/clean_data" # Example relative path
    SELECTED_TAGS = [
        'math', 'graphs', 'strings', 'number theory', 'trees',
        'geometry', 'games', 'probabilities'
    ]
    OTHER_FEATURE_COLUMNS = ['time_limit', 'memory_limit', 'difficulty_rating']


# Columns provided by the user for analysis
SELECTED_COLUMNS = [
    "time_limit",
    "sample_outputs",
    "problem_notes",
    "problem_description",
    "output_specification",
    "input_specification",
    "difficulty_rating",
    "memory_limit",
    "sample_inputs",
    "execution_result", # Note: Might not be very useful unless cleaned/categorized
    "solution_code",
]

# Identify text columns within SELECTED_COLUMNS for specific analysis
TEXT_COLUMNS_FOR_ANALYSIS = [
    "problem_notes", "problem_description", "output_specification",
    "input_specification", "sample_inputs", "sample_outputs",
    "solution_code", "execution_result"
]
# Ensure text columns are actually in SELECTED_COLUMNS
TEXT_COLUMNS_FOR_ANALYSIS = [col for col in TEXT_COLUMNS_FOR_ANALYSIS if col in SELECTED_COLUMNS]


def generate_tag_heatmap(df, tags, output_path):
    """Generates and saves a heatmap of tag co-occurrences."""
    print("Generating tag co-occurrence heatmap...")
    tag_data = df[tags].fillna(0).astype(int)
    cooccurrence_matrix = tag_data.T.dot(tag_data)
    # Normalize by diagonal to get conditional probability? Optional.
    # Or just show raw counts. Let's show raw counts.
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence_matrix, annot=True, fmt="d", cmap="viridis")
    plt.title("Tag Co-occurrence Counts")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Tag heatmap saved to {output_path}")

def plot_tag_distribution(df, tags, output_path):
    """Generates and saves a plot of the number of active tags per problem."""
    print("Generating tag count distribution plot...")
    tag_counts = df[tags].fillna(0).astype(int).sum(axis=1)
    avg_tags = tag_counts.mean()

    plt.figure(figsize=(10, 6))
    counts_dist = tag_counts.value_counts(normalize=True).sort_index() * 100
    ax = sns.barplot(x=counts_dist.index, y=counts_dist.values, palette="viridis")
    plt.title("Distribution of Number of Active Tags per Problem")
    plt.xlabel("Number of Active Tags")
    plt.ylabel("Percentage of Problems (%)")
    # Add percentage labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Tag distribution plot saved to {output_path}")
    return avg_tags

def generate_correlation_heatmap(df, columns, title, output_path):
    """Generates and saves a correlation heatmap for specified columns."""
    print(f"Generating correlation heatmap: {title}...")
    # Convert relevant columns to numeric, coercing errors
    numeric_df = df[columns].apply(pd.to_numeric, errors='coerce')
    # Drop columns that couldn't be converted entirely (all NaN)
    numeric_df = numeric_df.dropna(axis=1, how='all')
    if numeric_df.empty:
         print(f"Warning: No valid numeric data found for columns: {columns}. Skipping heatmap.")
         return
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

def calculate_text_stats(df, text_columns):
    """Calculates missing percentage and average length for text columns."""
    print("Calculating text column statistics...")
    stats = {}
    total_rows = len(df)
    if total_rows == 0:
        return {}

    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Text column '{col}' not found in DataFrame. Skipping.")
            continue
        # Calculate missing/empty percentage
        # Treat None, NaN, and empty strings as missing/empty
        missing_empty_count = df[col].isna().sum() + (df[col] == '').sum()
        missing_perc = (missing_empty_count / total_rows) * 100

        # Calculate average length of non-empty strings
        non_empty_lengths = df.loc[df[col].notna() & (df[col] != ''), col].astype(str).str.len()
        avg_len = non_empty_lengths.mean() if not non_empty_lengths.empty else 0

        stats[col] = {
            "missing_perc": round(missing_perc, 2),
            "avg_len": round(avg_len, 2)
        }
    print("Text stats calculated.")
    return stats

def generate_readme(stats, output_dir, input_csv_path):
    """Generates the README.md file content."""
    print("Generating README content...")
    readme_content = f"""# Data Analysis Report for Cleaned Codeforces Data

Analysis performed on: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`
Data source: `{os.path.basename(input_csv_path)}`

## 1. Basic Statistics

* Total problems analyzed: `{stats['total_rows']}`
* Columns included in analysis: `{', '.join(stats['analyzed_columns'])}`

## 2. Tag Analysis

### Tag Co-occurrence Heatmap

This heatmap shows how often pairs of tags appear together in the dataset (raw counts).

![Tag Co-occurrence Heatmap](tag_cooccurrence_heatmap.png)

### Distribution of Active Tags per Problem

This plot shows the percentage of problems having a specific number of assigned tags.

![Tag Count Distribution](tag_count_distribution.png)

* Average number of tags per problem: `{stats['avg_tags_per_problem']:.2f}`

## 3. Numerical Feature Correlation

### Correlation between Numerical Features

Correlation between `{', '.join(stats['numerical_columns'])}`.

![Numerical Feature Correlation](numerical_correlation.png)

### Correlation between Numerical Features and Tags

Correlation between numerical features and the presence of specific tags. Note: Correlation with binary (0/1) tags indicates association strength.

![Numerical vs Tag Correlation](numerical_tag_correlation.png)

## 4. Text Feature Analysis

Statistics for selected text columns:

| Feature                 | % Missing/Empty | Avg. Length (Non-Empty) |
| ----------------------- | --------------- | ----------------------- |
"""
    # Add text stats rows
    for col, data in stats['text_stats'].items():
        readme_content += f"| {col:<23} | {data['missing_perc']}%{' ':<13} | {data['avg_len']}{' ':<21} |\n"

    readme_path = os.path.join(output_dir, "EDA_README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"README generated at {readme_path}")


def main(args):
    # --- Setup ---
    input_csv_path = args.csv_path
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations") # Subdir for plots
    os.makedirs(viz_dir, exist_ok=True)


    print(f"Loading data from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Data loaded successfully with {len(df)} rows.")

    # Filter DataFrame to only include SELECTED_COLUMNS + SELECTED_TAGS
    cols_to_keep = list(set(SELECTED_COLUMNS + SELECTED_TAGS + OTHER_FEATURE_COLUMNS))
    # Check which columns actually exist in the loaded dataframe
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    missing_cols = [col for col in cols_to_keep if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following requested columns were not found in the CSV and will be skipped: {missing_cols}")

    df_analysis = df[existing_cols].copy()

    # Ensure tag columns are numeric (0/1)
    for tag in SELECTED_TAGS:
        if tag in df_analysis.columns:
            df_analysis[tag] = pd.to_numeric(df_analysis[tag], errors='coerce').fillna(0).astype(int)
        else:
             print(f"Warning: Tag column '{tag}' not found. Skipping.")

    # Ensure numerical columns are numeric
    valid_numerical_cols = []
    for col in OTHER_FEATURE_COLUMNS:
        if col in df_analysis.columns:
             df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
             # Keep track of numerical columns that are actually numeric after coercion
             if pd.api.types.is_numeric_dtype(df_analysis[col]):
                  valid_numerical_cols.append(col)
        else:
             print(f"Warning: Numerical column '{col}' not found. Skipping.")


    # --- Perform Analyses ---
    stats = {}
    stats['total_rows'] = len(df_analysis)
    stats['analyzed_columns'] = existing_cols
    stats['numerical_columns'] = valid_numerical_cols

    # Ignore warnings during plotting (e.g., from missing fonts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. Tag Heatmap
        heatmap_path = os.path.join(viz_dir, "tag_cooccurrence_heatmap.png")
        generate_tag_heatmap(df_analysis, [t for t in SELECTED_TAGS if t in df_analysis.columns], heatmap_path)

        # 2. Tag Distribution
        dist_path = os.path.join(viz_dir, "tag_count_distribution.png")
        stats['avg_tags_per_problem'] = plot_tag_distribution(df_analysis, [t for t in SELECTED_TAGS if t in df_analysis.columns], dist_path)

        # 3. Numerical Correlation
        if valid_numerical_cols:
            num_corr_path = os.path.join(viz_dir, "numerical_correlation.png")
            generate_correlation_heatmap(df_analysis, valid_numerical_cols, "Numerical Feature Correlation", num_corr_path)
        else:
             print("Skipping numerical correlation heatmap as no valid numerical columns were found.")


        # 4. Numerical vs Tag Correlation
        num_tag_cols = valid_numerical_cols + [t for t in SELECTED_TAGS if t in df_analysis.columns]
        if valid_numerical_cols and any(t in df_analysis.columns for t in SELECTED_TAGS):
             num_tag_corr_path = os.path.join(viz_dir, "numerical_tag_correlation.png")
             # Only show correlations between numerical and tags, not tags vs tags again
             temp_corr_df = df_analysis[num_tag_cols].corr()
             # Select rows for numerical, columns for tags, and vice versa, then combine? Or just show full matrix?
             # Let's show the relevant slice: Numerical rows, Tag columns
             corr_slice = temp_corr_df.loc[valid_numerical_cols, [t for t in SELECTED_TAGS if t in df_analysis.columns]]

             plt.figure(figsize=(12, max(6, len(valid_numerical_cols)*0.8))) # Adjust size
             sns.heatmap(corr_slice, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
             plt.title("Correlation between Numerical Features and Tags")
             plt.xticks(rotation=45, ha='right')
             plt.yticks(rotation=0)
             plt.tight_layout()
             plt.savefig(num_tag_corr_path)
             plt.close()
             print(f"Numerical vs Tag correlation heatmap saved to {num_tag_corr_path}")

        else:
             print("Skipping numerical vs tag correlation heatmap due to missing numerical or tag columns.")


    # 5. Text Stats
    stats['text_stats'] = calculate_text_stats(df_analysis, [col for col in TEXT_COLUMNS_FOR_ANALYSIS if col in df_analysis.columns])

    # --- Generate README ---
    # Note: README expects images in the same directory or a relative path
    # We saved images to viz_dir, so README should reference them relative to output_dir
    # Let's adjust the README generation to use relative paths from output_dir
    generate_readme(stats, output_dir, input_csv_path) # Pass output_dir

    print("\nEDA Report Generation Complete.")
    print(f"Report saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDA report and visualizations for the cleaned Codeforces dataset.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the cleaned data CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eda_report",
        help="Directory to save the generated README.md and visualization images."
    )
    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.csv_path):
         print(f"Error: Input CSV path does not exist: {args.csv_path}")
    else:
         main(args)

