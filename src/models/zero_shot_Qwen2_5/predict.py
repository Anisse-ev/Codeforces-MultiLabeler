import argparse
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime # Added for timestamp

# Project imports
from config.links_and_paths import (
    TEST_DATA_CSV, ZERO_SHOT_PREDICTIONS_DIR, ZERO_SHOT_SAVED_MODEL_PATH,
    BASE_MODEL_NAME # Added for info json
)
from config.data_config import (
     PROBLEM_ID_COLUMN, GROUND_TRUTH_TAG_COLUMNS,
     PREDICTION_TAG_COLUMNS,
     TEXT_FEATURE_COLUMNS, CODE_FEATURE_COLUMNS # Added for checking columns
)
from src.utils import (
    generate_prompt, parse_model_output, get_incremental_directory_path
)
# Import the load_base_model function which now accepts the token
from src.models.zero_shot_Qwen2_5.load_model import load_base_model

# Generation settings
MAX_NEW_TOKENS = 60 # As per notebook
BATCH_SIZE = 4 # Adjust based on GPU memory

def generate_batch_predictions(prompts: list[str], model, tokenizer) -> list[str]:
    """
    Generates predictions for a batch of prompts.
    (Content unchanged from previous version)
    """
    # Prepare chat format for the model
    texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant and a highly capable problem analyzer."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

    # Tokenize batch with padding
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length or 2048, # Ensure max_length is set
        padding_side='left' # For decoder-only models
    ).to(model.device) # Move inputs to the same device as the model

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated part
    responses = []
    input_ids_batch = model_inputs["input_ids"]
    for input_ids, output_ids in zip(input_ids_batch, generated_ids):
        prompt_length = len(input_ids)
        generated_response_ids = output_ids[prompt_length:]
        response = tokenizer.decode(generated_response_ids, skip_special_tokens=True)
        responses.append(response)

    # Clean up memory
    del model_inputs
    del generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return responses

def main(args):
    # --- 1. Load Data ---
    print(f"Loading data from: {args.csv_path}")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Input CSV not found at {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} examples.")

    required_cols = TEXT_FEATURE_COLUMNS + CODE_FEATURE_COLUMNS + [PROBLEM_ID_COLUMN]
    if not all(col in df.columns for col in required_cols):
         print(f"Warning: Not all expected feature columns found in CSV. Required: {required_cols}")

    # --- 2. Load Model ---
    # Check if model exists, if not, load_base_model will download it
    # Pass the hf_token argument here
    if not os.path.exists(os.path.join(ZERO_SHOT_SAVED_MODEL_PATH, "config.json")):
        print("Base model not found locally. Running load_model script...")
        load_base_model(hf_token=args.hf_token)
    else:
        print("Base model found locally.")

    # Now load the model for inference, passing the token again in case it's needed
    # for verification even when loading locally.
    model, tokenizer = load_base_model(hf_token=args.hf_token)

    # --- 3. Generate Prompts ---
    print("Generating prompts...")
    df['prompt'] = df.apply(generate_prompt, axis=1)

    # --- 4. Run Predictions ---
    all_predictions = []
    raw_outputs = []
    print(f"Starting zero-shot predictions with batch size {BATCH_SIZE}...")

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Zero-shot Predicting"):
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

    output_dir = get_incremental_directory_path(ZERO_SHOT_PREDICTIONS_DIR, prefix="zeroshot_")
    predictions_csv_path = os.path.join(output_dir, "predictions.csv")
    info_json_path = os.path.join(output_dir, "input_data_info.json")

    results_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved to: {predictions_csv_path}")

    info_data = {
        "input_csv_path": os.path.abspath(args.csv_path),
        "model_name": BASE_MODEL_NAME,
        "model_loaded_from": ZERO_SHOT_SAVED_MODEL_PATH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prediction_timestamp": datetime.now().isoformat(),
        "total_examples": len(df),
        "batch_size": BATCH_SIZE
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
    parser = argparse.ArgumentParser(description="Run Zero-Shot prediction using Qwen model.")
    parser.add_argument(
        "--csv_path", type=str, default=TEST_DATA_CSV,
        help=f"Path to the input CSV file containing problems. Defaults to {TEST_DATA_CSV}"
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
