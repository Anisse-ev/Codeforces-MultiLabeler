# Data Analysis Report for Cleaned Codeforces Data

Analysis performed on: `2025-04-17 09:01:54`
Data source: `codeforces_clean_data_2025-04-17.csv`

## 1. Basic Statistics

* Total problems analyzed: `4982`
* Columns included in analysis: `trees, geometry, solution_code, input_specification, problem_description, problem_notes, difficulty_rating, strings, output_specification, time_limit, memory_limit, probabilities, sample_inputs, math, graphs, execution_result, games, number theory, sample_outputs`

## 2. Tag Analysis

### Tag Co-occurrence Heatmap

This heatmap shows how often pairs of tags appear together in the dataset (raw counts).

![Tag Co-occurrence Heatmap](tag_cooccurrence_heatmap.png)

### Distribution of Active Tags per Problem

This plot shows the percentage of problems having a specific number of assigned tags.

![Tag Count Distribution](tag_count_distribution.png)

* Average number of tags per problem: `0.68`

## 3. Numerical Feature Correlation

### Correlation between Numerical Features

Correlation between `time_limit, memory_limit, difficulty_rating`.

![Numerical Feature Correlation](numerical_correlation.png)

### Correlation between Numerical Features and Tags

Correlation between numerical features and the presence of specific tags. Note: Correlation with binary (0/1) tags indicates association strength.

![Numerical vs Tag Correlation](numerical_tag_correlation.png)

## 4. Text Feature Analysis

Statistics for selected text columns:

| Feature                 | % Missing/Empty | Avg. Length (Non-Empty) |
| ----------------------- | --------------- | ----------------------- |
| problem_notes           | 0.0%              | 319.3                      |
| problem_description     | 0.0%              | 951.21                      |
| output_specification    | 0.0%              | 204.11                      |
| input_specification     | 0.0%              | 403.23                      |
| sample_inputs           | 0.0%              | 68.82                      |
| sample_outputs          | 0.0%              | 29.73                      |
| solution_code           | 0.0%              | 1232.57                      |
| execution_result        | 0.0%              | 6.0                      |
