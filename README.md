# Codeforces Multi-Label Problem Classification

## Project Goal

This project aims to automatically classify programming problems from Codeforces into relevant categories (tags) based on their descriptions, notes, code solutions, and other metadata. The primary focus is on predicting a selected set of 8 common tags: `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`.

## Models Implemented

Four different approaches have been implemented and evaluated:

1.  **TF-IDF + RandomForest (`tf_idf`)**

    - **Description:** A baseline approach using traditional machine learning. It combines TF-IDF features extracted from the problem description text with numerical features (like time/memory limits, difficulty rating).
    - **Preprocessing:** Numerical features undergo mean imputation for missing values and standard scaling.
    - **Training:** To handle the inherent class imbalance (more non-tagged examples than tagged ones for each label), random undersampling is applied to the training data for each tag individually. A separate RandomForest classifier is then trained for each of the 8 target tags on the balanced data.
    - **Saved Artifacts:** The fitted TF-IDF vectorizer, numerical preprocessor (imputer+scaler) combined in a `ColumnTransformer`, and the 8 trained RandomForest models are saved using `joblib`, along with training parameters in a JSON file.

2.  **Zero-Shot Qwen (`zero_shot_qwen`)**

    - **Description:** This model uses a pre-trained Large Language Model (LLM) from the Qwen family (e.g., Qwen2-1.5B-Instruct or Qwen2.5-7B-Instruct) directly without any fine-tuning.
    - **Method:** The LLM is prompted with the problem details (description, notes, code) and specific instructions to output a JSON object indicating the presence (1) or absence (0) of each target tag.
    - **Evaluation:** Performance is measured based on how well the pre-trained model understands the task and follows instructions in a zero-shot setting. _(Note: Assumes a corresponding predict script exists in `src/models/zero_shot_qwen/`)_

3.  **Fine-tuned Qwen - Generative (`qwen_generative_finetune`)**

    - **Description:** This approach fine-tunes a pre-trained Qwen LLM using PEFT (Parameter-Efficient Fine-Tuning) techniques like LoRA.
    - **Method:** The model is trained specifically to generate the target JSON output (`<output>{"tag": 1, ...}</output>`) when given the problem details as input. It uses `SFTTrainer` for sequence-to-sequence fine-tuning.
    - **Goal:** To adapt the LLM to better understand the relationship between problem details and tags, and to improve its ability to generate the correctly formatted JSON output. _(Note: Assumes corresponding train/predict scripts exist in `src/models/qwen_generative_finetune/`)_

4.  **Fine-tuned Qwen - Classification (`qwen_peft_classification`)**
    - **Description:** This approach also fine-tunes a Qwen LLM with PEFT/LoRA but reframes the task as a direct multi-label classification problem.
    - **Method:** It uses the `AutoModelForSequenceClassification` architecture, adding a classification head on top of the base Qwen model. The model is trained to predict a multi-hot vector representing the tags present, using a standard classification loss (Binary Cross-Entropy).
    - **Goal:** To potentially achieve more stable training and prediction by directly optimizing for classification metrics, avoiding the need for potentially brittle JSON parsing from a generative model.

## Train/Test Split

The dataset is split into training and testing sets based on the problem creation time (`creation_order` column, derived from `creation_timestamp`). This simulates a real-world scenario where we train on past data and test on more recent, unseen problems.

- **Method:** Chronological split.
- **Test Set Size:** The most recent `X%` of the data (defined by `TEST_SET_SIZE` in `config.data_config`) is reserved for the test set.
- **Files:** The split results are saved as `codeforces_train_data.csv` and `codeforces_test_data.csv` in the `data/train_test_split/` directory by the `src/data/train_test_split.py` script. A `codeforces_zero_shot_data.csv` containing the full dataset might also be saved.

## Evaluation Results

Metrics are calculated on the test set (`codeforces_test_data.csv`).

### 1. TF-IDF + RandomForest Results

(Based on run `tfidf_rf_sampled_20250417_002`)

--- Per-Tag Metrics ---Tag: math | P: 0.504 | R: 0.728 | F1: 0.596 | Sup: 475Tag: graphs | P: 0.414 | R: 0.471 | F1: 0.441 | Sup: 138Tag: strings | P: 0.528 | R: 0.879 | F1: 0.660 | Sup: 107Tag: number theory | P: 0.242 | R: 0.746 | F1: 0.366 | Sup: 126Tag: trees | P: 0.556 | R: 0.771 | F1: 0.646 | Sup: 96Tag: geometry | P: 0.380 | R: 0.594 | F1: 0.463 | Sup: 32Tag: games | P: 0.537 | R: 0.853 | F1: 0.659 | Sup: 34Tag: probabilities | P: 0.110 | R: 0.696 | F1: 0.189 | Sup: 23Macro Avg F1: 0.502Micro Avg P: 0.411 | R: 0.715 | F1: 0.522--- Custom Evaluation Stats ---Total problems: 1246Perfect matches: 374 (30.02%)At least one correct tag: 606 (48.64%)No incorrect tags AND >=1 correct tag: 299 (24.0%)

### 2. Zero-Shot Qwen Results

_(Please fill in results after running prediction using `src/models/zero_shot_qwen/predict.py`)_

--- Per-Tag Metrics ---Tag: math | P: ... | R: ... | F1: ... | Sup: ...Tag: graphs | P: ... | R: ... | F1: ... | Sup: ...Tag: strings | P: ... | R: ... | F1: ... | Sup: ...Tag: number theory | P: ... | R: ... | F1: ... | Sup: ...Tag: trees | P: ... | R: ... | F1: ... | Sup: ...Tag: geometry | P: ... | R: ... | F1: ... | Sup: ...Tag: games | P: ... | R: ... | F1: ... | Sup: ...Tag: probabilities | P: ... | R: ... | F1: ... | Sup: ...Macro Avg F1: ...Micro Avg P: ... | R: ... | F1: ...--- Custom Evaluation Stats ---Total problems: ...Perfect matches: ... (...)%At least one correct tag: ... (...)%No incorrect tags AND >=1 correct tag: ... (...)%

### 3. Fine-tuned Qwen (Generative) Results

_(Please fill in results after running training and prediction using scripts in `src/models/qwen_generative_finetune/`)_

--- Per-Tag Metrics ---Tag: math | P: ... | R: ... | F1: ... | Sup: ...Tag: graphs | P: ... | R: ... | F1: ... | Sup: ...... (add rows for other tags) ...Macro Avg F1: ...Micro Avg P: ... | R: ... | F1: ...--- Custom Evaluation Stats ---Total problems: ...Perfect matches: ... (...)%At least one correct tag: ... (...)%No incorrect tags AND >=1 correct tag: ... (...)%

### 4. Fine-tuned Qwen (Classification) Results

_(Please fill in results after running training and prediction using scripts in `src/models/qwen_peft_classification/`)_

--- Per-Tag Metrics ---Tag: math | P: ... | R: ... | F1: ... | Sup: ...Tag: graphs | P: ... | R: ... | F1: ... | Sup: ...... (add rows for other tags) ...Macro Avg F1: ...Micro Avg P: ... | R: ... | F1: ...--- Custom Evaluation Stats ---Total problems: ...Perfect matches: ... (...)%At least one correct tag: ... (...)%No incorrect tags AND >=1 correct tag: ... (...)%

## Usage

This section explains how to run the individual training and prediction scripts for each model. Ensure all dependencies are installed (`pip install -r requirements.txt`) and run commands from the project root directory.

### 1. TF-IDF + RandomForest (`tf_idf`)

- **Training:**

  ```bash
  python -m src.models.tf_idf.train_model --train_csv_path <path/to/train_data.csv>
  ```

  _(Defaults to `data/train_test_split/codeforces_train_data.csv`)_

  - This saves the preprocessor and models in a new timestamped directory inside `src/models/tf_idf/saved_model/`.

- **Prediction:**
  ```bash
  python -m src.models.tf_idf.predict --csv_path <path/to/predict_data.csv> [--model_dir_path <path/to/saved_model_dir>] [--run_evaluation]
  ```
  - Defaults to predicting on `data/train_test_split/codeforces_test_data.csv`.
  - If `--model_dir_path` is omitted, it uses the latest directory found in `src/models/tf_idf/saved_model/`.
  - Use `--run_evaluation` to calculate metrics if the input CSV contains ground truth columns.
  - Predictions are saved in a new timestamped directory inside `src/models/tf_idf/predictions/`.

### 2. Zero-Shot Qwen (`zero_shot_qwen`)

- **Training:** Not applicable for zero-shot models. Ensure the base model is downloaded if needed (may happen automatically on first predict run or via a separate load script).

- **Prediction:**
  ```bash
  python -m src.models.zero_shot_qwen.predict --csv_path <path/to/predict_data.csv> [--run_evaluation] [--hf_token <token>]
  ```
  - Defaults to predicting on the test set.
  - Requires the base Qwen model (e.g., Qwen2-1.5B-Instruct) to be accessible.
  - Use `--run_evaluation` to calculate metrics.
  - Use `--hf_token` if needed for model access.
  - Predictions saved in `src/models/zero_shot_qwen/predictions/`.

### 3. Fine-tuned Qwen - Generative (`qwen_generative_finetune`)

_(Assumes scripts `train_model.py` and `predict.py` exist in `src/models/qwen_generative_finetune/`)_

- **Training:**

  ```bash
  python -m src.models.qwen_generative_finetune.train_model --train_csv_path <path/to/train_data.csv> [training_options...]
  ```

  - Specify training options like `--num_epochs`, `--learning_rate`, `--base_model`, `--hf_token`, etc.
  - Saves fine-tuned adapters (LoRA weights) in a new timestamped directory inside `src/models/qwen_generative_finetune/runs/`.

- **Prediction:**
  ```bash
  python -m src.models.qwen_generative_finetune.predict --csv_path <path/to/predict_data.csv> [--adapter_run_dir <path/to/run_dir>] [--run_evaluation] [--hf_token <token>]
  ```
  - Defaults to predicting on the test set.
  - If `--adapter_run_dir` is omitted, uses the latest run found in `src/models/qwen_generative_finetune/runs/`.
  - Use `--run_evaluation` to calculate metrics.
  - Predictions saved in `src/models/qwen_generative_finetune/predictions/`.

### 4. Fine-tuned Qwen - Classification (`qwen_peft_classification`)

- **Training:**

  ```bash
  python -m src.models.qwen_peft_classification.train_model --train_csv_path <path/to/train_data.csv> [training_options...]
  ```

  - Specify training options like `--num_epochs`, `--learning_rate`, `--base_model`, `--lora_r`, `--hf_token`, etc.
  - Saves fine-tuned adapters (LoRA weights) in a new timestamped directory inside `src/models/qwen_peft_classification/runs/`.

- **Prediction:**
  ```bash
  python -m src.models.qwen_peft_classification.predict --csv_path <path/to/predict_data.csv> [--adapter_run_dir <path/to/run_dir>] [--run_evaluation] [--hf_token <token>]
  ```
  - Defaults to predicting on the test set.
  - If `--adapter_run_dir` is omitted, uses the latest run found in `src/models/qwen_peft_classification/runs/`.
  - Use `--run_evaluation` to calculate metrics.
  - Predictions saved in `src/models/qwen_peft_classification/predictions/`.

**Recommendation:** Start by running the TF-IDF model (`python -m src.models.tf_idf.train_model` followed by `python -m src.models.tf_idf.predict --run_evaluation`) to establish a baseline performance before exploring the more complex LLM approaches.
