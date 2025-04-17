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
    # Assumes the script is run from the project root or PYTHONPATH is set
    from config.links_and_paths import CLEAN_DATA_DIR # Not strictly needed if path is argument
    from config.data_config import SELECTED_TAGS, OTHER_FEATURE_COLUMNS
except ModuleNotFoundError:
    print("Warning: Could not import from config files. Defining defaults locally.")
    # Define fallbacks if run standalone - adjust paths as needed
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

# --- Plotting Functions ---

def generate_tag_heatmap(df, tags, output_path):
    """Generates and saves a heatmap of tag co-occurrences."""
    print("Generating tag co-occurrence heatmap...")
    tag_data = df[tags].fillna(0).astype(int)
    cooccurrence_matrix = tag_data.T.dot(tag_data)
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
    numeric_df = df[columns].apply(pd.to_numeric, errors='coerce')
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

# --- Text Analysis Function ---

def calculate_text_stats(df, text_columns):
    """Calculates missing percentage and average length for text columns."""
    print("Calculating text column statistics...")
    stats = {}
    total_rows = len(df)
    if total_rows == 0: return {}

    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Text column '{col}' not found in DataFrame. Skipping.")
            continue
        missing_empty_count = df[col].isna().sum() + (df[col] == '').sum()
        missing_perc = (missing_empty_count / total_rows) * 100
        non_empty_lengths = df.loc[df[col].notna() & (df[col] != ''), col].astype(str).str.len()
        avg_len = non_empty_lengths.mean() if not non_empty_lengths.empty else 0
        stats[col] = {
            "missing_perc": round(missing_perc, 2),
            "avg_len": round(avg_len, 2)
        }
    print("Text stats calculated.")
    return stats

# --- README Generation Function ---

def generate_readme(stats, readme_path, viz_subdir_name, input_csv_path):
    """Generates the README.md file content."""
    print("Generating README content...")
    # Use relative paths for images within the README
    tag_heatmap_relpath = f"{viz_subdir_name}/tag_cooccurrence_heatmap.png"
    tag_dist_relpath = f"{viz_subdir_name}/tag_count_distribution.png"
    num_corr_relpath = f"{viz_subdir_name}/numerical_correlation.png"
    num_tag_corr_relpath = f"{viz_subdir_name}/numerical_tag_correlation.png"

    readme_content = f"""# Data Analysis Report for Cleaned Codeforces Data

Analysis performed on: `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}`
Data source: `{os.path.basename(input_csv_path)}`

## 1. Basic Statistics

* Total problems analyzed: `{stats['total_rows']}`
* Columns included in analysis: `{', '.join(stats['analyzed_columns'])}`

## 2. Tag Analysis

### Tag Co-occurrence Heatmap

This heatmap shows how often pairs of tags appear together in the dataset (raw counts).

![Tag Co-occurrence Heatmap]({tag_heatmap_relpath})

### Distribution of Active Tags per Problem

This plot shows the percentage of problems having a specific number of assigned tags.

![Tag Count Distribution]({tag_dist_relpath})

* Average number of tags per problem: `{stats['avg_tags_per_problem']:.2f}`

## 3. Numerical Feature Correlation

### Correlation between Numerical Features

Correlation between `{', '.join(stats['numerical_columns'])}`.

![Numerical Feature Correlation]({num_corr_relpath})

### Correlation between Numerical Features and Tags

Correlation between numerical features and the presence of specific tags. Note: Correlation with binary (0/1) tags indicates association strength.

![Numerical vs Tag Correlation]({num_tag_corr_relpath})

## 4. Text Feature Analysis

Statistics for selected text columns:

| Feature                 | % Missing/Empty | Avg. Length (Non-Empty) |
| ----------------------- | --------------- | ----------------------- |
"""
    # Add text stats rows
    for col, data in stats['text_stats'].items():
        readme_content += f"| {col:<23} | {data['missing_perc']}%{' ':<13} | {data['avg_len']}{' ':<21} |\n"

    # Write the README file
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"README generated at {readme_path}")


# --- Main Execution ---

def main(args):
    # --- Setup Output Paths ---
    input_csv_path = args.csv_path
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define README path in the script's directory
    readme_path = os.path.join(script_dir, "EDA_README.md")
    # Define visualization subdirectory relative to the script's directory
    viz_subdir_name = "visualizations" # Name of the subdirectory for plots
    viz_dir = os.path.join(script_dir, viz_subdir_name)

    # Create visualization directory if it doesn't exist
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Outputting README to: {readme_path}")
    print(f"Outputting visualizations to: {viz_dir}")

    # --- Load Data ---
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

    # --- Prepare Data for Analysis ---
    cols_to_keep = list(set(SELECTED_COLUMNS + SELECTED_TAGS + OTHER_FEATURE_COLUMNS))
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    missing_cols = [col for col in cols_to_keep if col not in df.columns]
    if missing_cols:
        print(f"Warning: Skipping missing columns: {missing_cols}")
    df_analysis = df[existing_cols].copy()

    # Ensure tag columns are numeric
    valid_tags = []
    for tag in SELECTED_TAGS:
        if tag in df_analysis.columns:
            df_analysis[tag] = pd.to_numeric(df_analysis[tag], errors='coerce').fillna(0).astype(int)
            valid_tags.append(tag)
        else: print(f"Warning: Tag column '{tag}' not found.")

    # Ensure numerical columns are numeric
    valid_numerical_cols = []
    for col in OTHER_FEATURE_COLUMNS:
        if col in df_analysis.columns:
             df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
             if pd.api.types.is_numeric_dtype(df_analysis[col]):
                  valid_numerical_cols.append(col)
        else: print(f"Warning: Numerical column '{col}' not found.")

    # --- Perform Analyses ---
    stats = {}
    stats['total_rows'] = len(df_analysis)
    stats['analyzed_columns'] = existing_cols
    stats['numerical_columns'] = valid_numerical_cols

    with warnings.catch_warnings(): # Ignore plotting warnings
        warnings.simplefilter("ignore")

        # 1. Tag Heatmap (Use full path for saving)
        if valid_tags:
             heatmap_path = os.path.join(viz_dir, "tag_cooccurrence_heatmap.png")
             generate_tag_heatmap(df_analysis, valid_tags, heatmap_path)
        else: print("Skipping tag heatmap: No valid tag columns found.")


        # 2. Tag Distribution (Use full path for saving)
        if valid_tags:
             dist_path = os.path.join(viz_dir, "tag_count_distribution.png")
             stats['avg_tags_per_problem'] = plot_tag_distribution(df_analysis, valid_tags, dist_path)
        else:
             stats['avg_tags_per_problem'] = 0
             print("Skipping tag distribution: No valid tag columns found.")

        # 3. Numerical Correlation (Use full path for saving)
        if len(valid_numerical_cols) > 1: # Need at least 2 numerical cols for correlation
            num_corr_path = os.path.join(viz_dir, "numerical_correlation.png")
            generate_correlation_heatmap(df_analysis, valid_numerical_cols, "Numerical Feature Correlation", num_corr_path)
        else: print("Skipping numerical correlation: Less than 2 valid numerical columns found.")

        # 4. Numerical vs Tag Correlation (Use full path for saving)
        if valid_numerical_cols and valid_tags:
             num_tag_corr_path = os.path.join(viz_dir, "numerical_tag_correlation.png")
             num_tag_cols = valid_numerical_cols + valid_tags
             temp_corr_df = df_analysis[num_tag_cols].corr()
             corr_slice = temp_corr_df.loc[valid_numerical_cols, valid_tags] # Slice relevant part

             plt.figure(figsize=(max(8, len(valid_tags)*0.8), max(6, len(valid_numerical_cols)*0.8)))
             sns.heatmap(corr_slice, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, vmin=-1, vmax=1) # Set vmin/vmax
             plt.title("Correlation between Numerical Features and Tags")
             plt.xticks(rotation=45, ha='right')
             plt.yticks(rotation=0)
             plt.tight_layout()
             plt.savefig(num_tag_corr_path)
             plt.close()
             print(f"Numerical vs Tag correlation heatmap saved to {num_tag_corr_path}")
        else: print("Skipping numerical vs tag correlation: Missing valid numerical or tag columns.")

    # 5. Text Stats
    valid_text_cols = [col for col in TEXT_COLUMNS_FOR_ANALYSIS if col in df_analysis.columns]
    stats['text_stats'] = calculate_text_stats(df_analysis, valid_text_cols)

    # --- Generate README ---
    # Pass the path for the README and the name of the viz subdirectory
    generate_readme(stats, readme_path, viz_subdir_name, input_csv_path)

    print("\nEDA Report Generation Complete.")
    print(f"Report saved in directory: {script_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EDA report and visualizations for the cleaned Codeforces dataset.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the cleaned data CSV file."
    )
    # Removed --output_dir argument
    # parser.add_argument(
    #     "--output_dir", ...
    # )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
         print(f"Error: Input CSV path does not exist: {args.csv_path}")
    else:
         main(args)
