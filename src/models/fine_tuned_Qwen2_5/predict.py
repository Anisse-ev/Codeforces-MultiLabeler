import argparse
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Third-party Libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Project Imports
from config.links_and_paths import (
    TEST_DATA_CSV, ZERO_SHOT_SAVED_MODEL_PATH,
    FINE_TUNED_LORA_WEIGHTS_DIR, FINE_TUNED_PREDICTIONS_DIR,
    BASE_MODEL_NAME
)
from config.data_config import (
     PROBLEM_ID_COLUMN, GROUND_TRUTH_TAG_COLUMNS,
     PREDICTION_TAG_COLUMNS,
     TEXT_FEATURE_COLUMNS, CODE_FEATURE_COLUMNS
)
from src.utils import (
    generate_prompt, parse_model_output,
    get_incremental_directory_path, get_latest_subdirectory
)
from src.models.zero_shot_Qwen2_5.predict import generate_batch_predictions # Reusing this

# Generation settings
MAX_NEW_TOKENS = 100
BATCH_SIZE = 3

# Modified function to accept hf_token
def load_fine_tuned_model(lora_adapter_path: str, hf_token: str | None = None):
    """
    Loads the base model and applies the specified LoRA adapter.

    Args:
        lora_adapter_path (str): Path to the directory containing the saved LoRA adapter weights.
        hf_token (str | None): Optional Hugging Face token.

    Returns:
        tuple: (model, tokenizer) - The merged PEFT model and tokenizer.
    """
    print(f"Loading base model from: {ZERO_SHOT_SAVED_MODEL_PATH}")
    # Prepare kwargs for loading base model, including token if provided
    base_load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": 'auto'
    }
    if hf_token:
        base_load_kwargs["token"] = hf_token

    base_model = AutoModelForCausalLM.from_pretrained(
        ZERO_SHOT_SAVED_MODEL_PATH,
        **base_load_kwargs
    )

    tokenizer_load_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, **tokenizer_load_kwargs)

    # Set EOS/PAD tokens if needed
    if tokenizer.eos_token is None:
        print(f"Tokenizer loaded from adapter does not have EOS token. Setting to '<|endoftext|>'")
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         print(f"Setting pad_token to eos_token: {tokenizer.pad_token}")

    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    # Loading PEFT model from local path usually doesn't need token, but base model does.
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    model.eval()

    print("Fine-tuned model loaded successfully.")
    print(f"Model is on device: {model.device}")
    return model, tokenizer


def main(args):
    # --- 1. Load Data ---
    print(f"Loading data from: {args.csv_path}")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Input CSV not found at {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    if args.sample_size > 0 and args.sample_size < len(df):
        print(f"Sampling {args.sample_size} examples from the dataset.")
        df = df.sample(n=args.sample_size, random_state=args.random_seed).reset_index(drop=True)
    print(f"Loaded {len(df)} examples for prediction.")

    required_cols = TEXT_FEATURE_COLUMNS + CODE_FEATURE_COLUMNS + [PROBLEM_ID_COLUMN]
    if not all(col in df.columns for col in required_cols):
         print(f"Warning: Not all expected feature/ID columns found in CSV. Required: {required_cols}")

    # --- 2. Find and Load Fine-Tuned Model ---
    if args.lora_path:
        lora_adapter_path = args.lora_path
        if not os.path.isdir(lora_adapter_path):
             raise FileNotFoundError(f"Specified LoRA adapter path not found: {lora_adapter_path}")
        print(f"Using specified LoRA adapter path: {lora_adapter_path}")
    else:
        print("No specific LoRA path provided, searching for the latest adapter...")
        lora_adapter_path = get_latest_subdirectory(FINE_TUNED_LORA_WEIGHTS_DIR)
        if not lora_adapter_path:
            raise FileNotFoundError(f"No LoRA adapter weights found in {FINE_TUNED_LORA_WEIGHTS_DIR}. ")
        print(f"Found latest LoRA adapter path: {lora_adapter_path}")

    # Load the fine-tuned model, passing the token
    model, tokenizer = load_fine_tuned_model(lora_adapter_path, hf_token=args.hf_token)

    # --- 3. Generate Prompts ---
    print("Generating prompts for prediction...")
    df['prompt'] = df.apply(generate_prompt, axis=1)

    # --- 4. Run Predictions ---
    all_predictions = []
    raw_outputs = []
    print(f"Starting fine-tuned predictions with batch size {BATCH_SIZE}...")

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Fine-tuned Predicting"):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        batch_prompts = batch_df['prompt'].tolist()
        batch_raw_outputs = generate_batch_predictions(batch_prompts, model, tokenizer)
        raw_outputs.extend(batch_raw_outputs)
        batch_parsed_preds = [parse_model_output(output) for output in batch_raw_outputs]
        all_predictions.extend(batch_parsed_preds)

    # --- 5. Format and Save Results ---
    print("Formatting and saving results...")
    predictions_np = np.array(all_predictions)
    pred_df = pd.DataFrame(predictions_np, columns=PREDICTION_TAG_COLUMNS, index=df.index)

    actual_gt_cols = [col for col in GROUND_TRUTH_TAG_COLUMNS if col in df.columns]
    if not actual_gt_cols:
        print("Warning: Ground truth columns not found in input CSV. Evaluation requires them.")
        results_df = df[[PROBLEM_ID_COLUMN]].copy()
    else:
         results_df = df[[PROBLEM_ID_COLUMN] + actual_gt_cols].copy()

    results_df = pd.concat([results_df, pred_df], axis=1)
    results_df['raw_model_output'] = raw_outputs

    output_dir = get_incremental_directory_path(FINE_TUNED_PREDICTIONS_DIR, prefix="finetuned_")
    predictions_csv_path = os.path.join(output_dir, "predictions.csv")
    info_json_path = os.path.join(output_dir, "input_data_info.json")

    results_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved to: {predictions_csv_path}")

    info_data = {
        "input_csv_path": os.path.abspath(args.csv_path),
        "base_model_name": BASE_MODEL_NAME,
        "lora_adapter_path_used": os.path.abspath(lora_adapter_path),
        "max_new_tokens": MAX_NEW_TOKENS,
        "prediction_timestamp": datetime.now().isoformat(),
        "total_examples_predicted": len(df),
        "batch_size": BATCH_SIZE,
        "sampled_data": args.sample_size > 0,
        "sample_size": args.sample_size if args.sample_size > 0 else "all"
    }
    with open(info_json_path, "w") as f:
        json.dump(info_data, f, indent=4)
    print(f"Input data info saved to: {info_json_path}")

    # --- 6. Optional: Trigger Evaluation ---
    if actual_gt_cols and args.run_evaluation:
        print("\nRunning evaluation...")
        try:
            from src.evaluate import run_evaluation
            evaluation_results = run_evaluation(predictions_csv_path)
            eval_results_path = os.path.join(output_dir, "evaluation_results.json")
            with open(eval_results_path, "w") as f:
                 json.dump(evaluation_results, f, indent=4)
            print(f"Evaluation results saved to: {eval_results_path}")
        except ImportError:
             print("Could not import 'run_evaluation' from 'src.evaluate'. Skipping evaluation.")
        except Exception as e:
            print(f"Error during evaluation: {e}. Skipping evaluation.")
    elif args.run_evaluation:
         print("Skipping evaluation as ground truth columns were not found in the input CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prediction using a Fine-Tuned Qwen model with LoRA.")
    parser.add_argument(
        "--csv_path", type=str, default=TEST_DATA_CSV,
        help=f"Path to the input CSV file for prediction. Defaults to {TEST_DATA_CSV}"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="Path to the specific LoRA adapter directory. If None, uses the latest found."
    )
    parser.add_argument(
        "--sample_size", type=int, default=-1,
        help="Number of examples to sample from the CSV for prediction (-1 for all)."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for sampling."
    )
    parser.add_argument(
        "--run_evaluation", action='store_true',
        help="Run evaluation script after predictions if ground truth labels are available."
    )
    # Add the optional hf_token argument
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Optional Hugging Face API token for model download/verification."
    )
    args = parser.parse_args()
    main(args)