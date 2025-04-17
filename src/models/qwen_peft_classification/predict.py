import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import warnings

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", message=".*Could not find the quantized class.*")

from config.links_and_paths import TEST_DATA_CSV, MODELS_DIR
from config.data_config import (
        SELECTED_TAGS, TEXT_FEATURE_COLUMNS, CODE_FEATURE_COLUMNS,
        PROBLEM_ID_COLUMN, GROUND_TRUTH_TAG_COLUMNS
    )

DEFAULT_MAX_SEQ_LENGTH = 512
# --- Project-Specific Imports ---
try:
    from src.utils import get_incremental_directory_path, get_latest_subdirectory
    from src.evaluate import run_evaluation
    evaluation_possible = True
except ImportError:
    # Basic fallbacks
    def get_latest_subdirectory(base_dir):
        try:
            subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not subdirs: return None
            subdirs.sort(key=lambda x: os.path.basename(x).split('_')[-2] + os.path.basename(x).split('_')[-1] if '_' in os.path.basename(x) else os.path.basename(x), reverse=True)
            return subdirs[0]
        except FileNotFoundError: return None
        except Exception as e: print(f"Fallback error: {e}"); return None
    def get_incremental_directory_path(base_dir, prefix=""):
        today = datetime.today().strftime("%Y%m%d"); i = 1
        while True:
            path = os.path.join(base_dir, f"{prefix}{today}_{i:03d}")
            if not os.path.exists(path): os.makedirs(base_dir, exist_ok=True); os.makedirs(path, exist_ok=True); return path
            i += 1
    run_evaluation = None; evaluation_possible = False
    print("Warning: src.utils or src.evaluate not found, using basic fallbacks. Evaluation disabled.")


# --- Library Imports ---
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments, # Minimal args needed for Trainer predict
    BitsAndBytesConfig
)
from peft import PeftModel # To load adapters

# --- Script Configuration ---
MODEL_OUTPUT_SUBDIR = "qwen_peft_classification" # Subdirectory within MODELS_DIR
RUNS_SUBDIR = "runs" # Subdirectory where training runs are saved
PREDICTIONS_SUBDIR = "predictions" # Subdirectory for saving predictions
ADAPTER_SUBDIR_NAME = "final_model_adapters" # Name of the subdir saved by trainer
PARAMS_FILENAME = "training_run_params.json" # Name of the params file saved by train script
SIGMOID_THRESHOLD = 0.5 # Threshold for converting probabilities

# --- Helper Functions ---
def create_input_text(example):
    text_parts = []
    for col in TEXT_FEATURE_COLUMNS + CODE_FEATURE_COLUMNS:
        content = example.get(col)
        if pd.notna(content): text_parts.append(str(content))
    return "\n\n".join(text_parts).strip()

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["input_text"], padding="max_length", truncation=True, max_length=max_length
    )

def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# --- Main Prediction Function ---
def main(args):
    # --- Determine Device and Dtype ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = False; use_bf16 = False; model_dtype = torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported(): use_bf16 = True; model_dtype = torch.bfloat16
        else: use_fp16 = True; model_dtype = torch.float16
    print(f"Device: {device}, Use FP16: {use_fp16}, Use BF16: {use_bf16}")

    # --- Find Model Adapter Directory ---
    search_dir = os.path.join(MODELS_DIR, MODEL_OUTPUT_SUBDIR, RUNS_SUBDIR)
    if args.adapter_run_dir: # User specified a specific run directory
        run_dir = args.adapter_run_dir
        adapter_dir = os.path.join(run_dir, ADAPTER_SUBDIR_NAME)
        if not os.path.isdir(adapter_dir):
             raise FileNotFoundError(f"Could not find '{ADAPTER_SUBDIR_NAME}' in specified run directory: {run_dir}")
        print(f"Using specified run directory: {run_dir}")
    else: # Find the latest run directory
        print(f"No specific run directory provided, searching in {search_dir}...")
        latest_run_dir = get_latest_subdirectory(search_dir)
        if not latest_run_dir:
            raise FileNotFoundError(f"No saved model run directories found in {search_dir}.")
        adapter_dir = os.path.join(latest_run_dir, ADAPTER_SUBDIR_NAME)
        if not os.path.isdir(adapter_dir):
             raise FileNotFoundError(f"Could not find '{ADAPTER_SUBDIR_NAME}' in latest run directory: {latest_run_dir}")
        run_dir = latest_run_dir
        print(f"Found latest run directory: {run_dir}")
    print(f"Using adapter directory: {adapter_dir}")

    # --- Load Training Params from adapter directory ---
    params_load_path = os.path.join(adapter_dir, PARAMS_FILENAME)
    if not os.path.exists(params_load_path):
        raise FileNotFoundError(f"{PARAMS_FILENAME} not found in {adapter_dir}")

    print(f"Loading training parameters from: {params_load_path}")
    with open(params_load_path, 'r') as f:
        training_params = json.load(f)

    base_model_name = training_params.get("base_model")
    load_4bit = training_params.get("quantization") == "4-bit"
    load_8bit = training_params.get("quantization") == "8-bit"
    max_seq_length = training_params.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
    num_labels_trained = len(training_params.get("selected_tags", SELECTED_TAGS))

    if not base_model_name:
         raise ValueError("Base model name not found in training_run_params.json")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer from adapter directory: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    # --- Configure Quantization for Loading ---
    bnb_config = None
    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype, bnb_4bit_use_double_quant=True,
        )
        print("Loading base model with 4-bit quantization.")
    elif load_8bit:
         bnb_config = BitsAndBytesConfig(load_in_8bit=True)
         print("Loading base model with 8-bit quantization.")

    # --- Load Base Model ---
    print(f"Loading base model {base_model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels_trained,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        torch_dtype=model_dtype,
        quantization_config=bnb_config,
        token=args.hf_token
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Base model loaded.")

    # --- Apply PEFT Adapters ---
    print(f"Loading PEFT adapters from {adapter_dir}...")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    print("PEFT model ready for prediction.")

    # --- Load and Prepare Prediction Data ---
    print(f"Loading prediction data from: {args.csv_path}")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Prediction CSV not found: {args.csv_path}")
    df_pred_orig = pd.read_csv(args.csv_path)

    df_pred = df_pred_orig.copy()
    if args.subset_size > 0:
        print(f"Using subset of {args.subset_size} prediction examples.")
        df_pred = df_pred.sample(n=args.subset_size, random_state=42).reset_index(drop=True)

    print("Preprocessing prediction data...")
    pred_dataset_hf = Dataset.from_pandas(df_pred)
    pred_dataset_hf = pred_dataset_hf.map(lambda x: {"input_text": create_input_text(x)}, num_proc=1)
    pred_dataset_tokenized = pred_dataset_hf.map(
        lambda examples: tokenize_function(examples, tokenizer, max_seq_length),
        batched=True, num_proc=1, desc="Tokenizing Predict"
    )

    model_input_columns = ["input_ids", "attention_mask"]
    pred_dataset_final = pred_dataset_tokenized.remove_columns(
        [col for col in pred_dataset_tokenized.column_names if col not in model_input_columns]
    )
    pred_dataset_final.set_format("torch")
    print("Prediction dataset prepared.")

    # --- Run Prediction using Trainer ---
    print("Running predictions...")
    predict_args = TrainingArguments(
        output_dir="./temp_predict_output", # Not used for saving
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=1,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
    )
    trainer = Trainer(model=model, args=predict_args, tokenizer=tokenizer)
    predictions_output = trainer.predict(pred_dataset_final)

    # --- Process Predictions ---
    logits = predictions_output.predictions[0] if isinstance(predictions_output.predictions, tuple) else predictions_output.predictions
    print("Processing predictions (sigmoid + threshold)...")
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits)).numpy()
    predicted_labels = np.zeros(probs.shape); predicted_labels[np.where(probs >= SIGMOID_THRESHOLD)] = 1
    predicted_labels = predicted_labels.astype(int)

    # --- Format and Save Results ---
    print("Formatting and saving results...")
    pred_df = pd.DataFrame(predicted_labels, columns=[f"pred_{tag}" for tag in SELECTED_TAGS], index=df_pred.index)
    probs_df = pd.DataFrame(probs, columns=[f"prob_{tag}" for tag in SELECTED_TAGS], index=df_pred.index)

    id_col_present = PROBLEM_ID_COLUMN in df_pred.columns
    actual_gt_cols = [col for col in GROUND_TRUTH_TAG_COLUMNS if col in df_pred.columns]
    gt_cols_present = bool(actual_gt_cols)

    if not id_col_present: print(f"Warning: Problem ID column '{PROBLEM_ID_COLUMN}' not found in input CSV.")
    if not gt_cols_present: print(f"Warning: Ground truth tag columns not found in input CSV.")

    cols_to_include_orig = []
    if id_col_present: cols_to_include_orig.append(PROBLEM_ID_COLUMN)
    if gt_cols_present: cols_to_include_orig.extend(actual_gt_cols)

    # Start with original (subsetted) data to preserve all original columns + index
    results_df = df_pred.copy()
    # Add predictions and probabilities - join on index
    results_df = results_df.join(pred_df).join(probs_df)

    # Select relevant columns for final output (ID, GT, Preds, Probs) - keep all original cols too? User choice.
    # Let's just add preds and probs to the original subsetted df
    # final_output_cols = list(df_pred.columns) + list(pred_df.columns) + list(probs_df.columns)
    # results_df = results_df[final_output_cols] # Reorder if needed

    # Define output directory for predictions
    pred_output_base_dir = os.path.join(MODELS_DIR, MODEL_OUTPUT_SUBDIR, PREDICTIONS_SUBDIR)
    run_prefix = f"{base_model_name.split('/')[-1]}_pred_"
    output_dir_run = get_incremental_directory_path(pred_output_base_dir, prefix=run_prefix)
    predictions_csv_path = os.path.join(output_dir_run, "predictions.csv")
    info_json_path = os.path.join(output_dir_run, "prediction_run_info.json")

    results_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions (including probabilities) saved to: {predictions_csv_path}")

    info_data = {
        "input_csv_path": os.path.abspath(args.csv_path),
        "model_adapter_loaded_from_dir": os.path.abspath(adapter_dir),
        "base_model_used": base_model_name,
        "prediction_timestamp": datetime.now().isoformat(),
        "total_examples_predicted": len(df_pred),
        "subset_size_used": args.subset_size if args.subset_size > 0 else "all",
        "sigmoid_threshold": SIGMOID_THRESHOLD,
    }
    with open(info_json_path, "w") as f:
        json.dump(info_data, f, indent=4, default=convert_numpy_types)
    print(f"Prediction run info saved to: {info_json_path}")

    # --- Optional: Trigger Evaluation ---
    can_evaluate = gt_cols_present and evaluation_possible
    if args.run_evaluation:
        if can_evaluate:
            print("\nRunning evaluation...")
            try:
                evaluation_results = run_evaluation(predictions_csv_path)
                eval_results_path = os.path.join(output_dir_run, "evaluation_results.json")
                with open(eval_results_path, "w") as f:
                     json.dump(evaluation_results, f, indent=4, default=convert_numpy_types)
                print(f"Evaluation results saved to: {eval_results_path}")
            except Exception as e:
                print(f"Error during evaluation: {e}. Skipping evaluation.")
        else:
            reason = []
            if not gt_cols_present: reason.append("ground truth columns missing in input")
            if not evaluation_possible: reason.append("evaluate script not found/imported")
            print(f"Skipping evaluation: {', '.join(reason)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction using a fine-tuned Qwen PEFT classification model.")

    parser.add_argument("--csv_path", type=str, default=TEST_DATA_CSV, help="Path to the input CSV file for prediction.")
    # Changed argument name for clarity
    parser.add_argument("--adapter_run_dir", type=str, default=None, help="Path to the specific *training run* directory (containing 'final_model_adapters'). If None, uses the latest run found.")
    parser.add_argument("--subset_size", type=int, default=-1, help="Number of examples to sample (-1 for all).")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size for prediction.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--run_evaluation", action='store_true', help="Run evaluation if ground truth labels are available.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (optional).")

    args = parser.parse_args()
    main(args)
