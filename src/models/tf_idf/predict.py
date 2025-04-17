import argparse
import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

from config.links_and_paths import (
        TEST_DATA_CSV, TF_IDF_PREDICTIONS_DIR, TF_IDF_SAVED_MODEL_PATH,
    )
from config.data_config import (
        SELECTED_TAGS, TEXT_FEATURE_COLUMNS,
        PROBLEM_ID_COLUMN, GROUND_TRUTH_TAG_COLUMNS
    )


from src.models.evaluate import run_evaluation as run_evaluation_function

# Utility Imports
try:
    # Make sure these paths are correct relative to where you run the script
    from src.utils import (
        get_incremental_directory_path, get_latest_subdirectory
    )
except ImportError:
    # Simple fallback if utils not found (adjust logic if needed)
    def get_latest_subdirectory(base_dir):
        try:
            subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not subdirs: return None
            subdirs.sort(key=lambda x: os.path.basename(x).split('_')[-2] + os.path.basename(x).split('_')[-1] if '_' in os.path.basename(x) else os.path.basename(x), reverse=True)
            return subdirs[0]
        except FileNotFoundError:
             print(f"Error: Base directory for latest subdirectory not found: {base_dir}")
             return None
        except Exception as e:
             print(f"Error finding latest subdirectory in {base_dir}: {e}")
             return None # Fallback

    def get_incremental_directory_path(base_dir, prefix=""):
         today = datetime.today().strftime("%Y%m%d")
         i = 1
         while True:
             path = os.path.join(base_dir, f"{prefix}{today}_{i:03d}")
             if not os.path.exists(path):
                 os.makedirs(base_dir, exist_ok=True)
                 os.makedirs(path, exist_ok=True)
                 return path
             i += 1
    print("Warning: src.utils not found, using basic fallback for directory naming/finding.")

# Evaluation Import
try:
    # Make sure this path is correct relative to where you run the script
    from src.evaluate import run_evaluation
    evaluation_possible = True
except ImportError:
    run_evaluation = None
    evaluation_possible = False
    print("Warning: src.evaluate.run_evaluation not found. Evaluation step will be skipped.")


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

    # --- 2. Find and Load Model Directory & Params ---
    if args.model_dir_path: # Use specific directory if provided
        model_dir = args.model_dir_path
        if not os.path.isdir(model_dir):
             raise FileNotFoundError(f"Specified model directory path not found: {model_dir}")
        print(f"Using specified model directory: {model_dir}")
    else: # Find the latest directory otherwise
        print(f"No specific model directory provided, searching in {TF_IDF_SAVED_MODEL_PATH}...")
        model_dir = get_latest_subdirectory(TF_IDF_SAVED_MODEL_PATH)
        if not model_dir:
            raise FileNotFoundError(f"No saved model directories found in {TF_IDF_SAVED_MODEL_PATH}.")
        print(f"Found latest model directory: {model_dir}")

    params_load_path = os.path.join(model_dir, "training_params.json")
    if not os.path.exists(params_load_path):
        raise FileNotFoundError(f"training_params.json not found in {model_dir}")
    print(f"Loading training parameters from: {params_load_path}")
    with open(params_load_path, 'r') as f:
        training_params = json.load(f)

    # --- 3. Load Preprocessor and Models ---
    preprocessor_filename = training_params.get("preprocessor_path", "preprocessor.joblib")
    models_filename = training_params.get("models_path", "models.joblib")
    preprocessor_load_path = os.path.join(model_dir, preprocessor_filename)
    models_load_path = os.path.join(model_dir, models_filename)

    if not os.path.exists(preprocessor_load_path):
        raise FileNotFoundError(f"Preprocessor file '{preprocessor_filename}' not found in {model_dir}")
    if not os.path.exists(models_load_path):
        raise FileNotFoundError(f"Models file '{models_filename}' not found in {model_dir}")

    print(f"Loading preprocessor from: {preprocessor_load_path}")
    preprocessor = joblib.load(preprocessor_load_path)
    print(f"Loading models dictionary from: {models_load_path}")
    loaded_rf_models = joblib.load(models_load_path)

    # --- 4. Prepare Data for Prediction ---
    try:
        required_input_cols = list(preprocessor.feature_names_in_)
        # print(f"Preprocessor expects columns: {required_input_cols}") # Optional: Keep for debugging
    except AttributeError:
        text_feat = training_params.get("text_feature_used", TEXT_FEATURE_COLUMNS[0])
        num_feat = training_params.get("numerical_features_used", [])
        required_input_cols = [text_feat] + num_feat
        print(f"Inferring required columns: {required_input_cols}")

    missing_input_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_input_cols:
         # Be more specific about which columns are missing for features
         print(f"Warning: Missing required columns for prediction features: {missing_input_cols}. Trying to continue...")
         # Attempt to create missing columns with NaN if possible, otherwise raise error later if needed by transform
         for col in missing_input_cols:
              df[col] = np.nan # Add missing feature columns as NaN

    # Handle NaN/inf in numerical columns before transform
    numerical_features_used = training_params.get("numerical_features_used", [])
    for col in numerical_features_used:
         if col in df.columns:
              df[col] = pd.to_numeric(df[col], errors='coerce')
              # FIX for FutureWarning: Assign result back instead of using inplace=True
              df[col] = df[col].replace([np.inf, -np.inf], np.nan)
         # else: # This case is handled by missing_input_cols check above
              # df[col] = np.nan

    # Handle NaN in text column
    text_feature_used = training_params.get("text_feature_used", TEXT_FEATURE_COLUMNS[0])
    if text_feature_used in df.columns:
        df[text_feature_used] = df[text_feature_used].fillna('')
    else:
         # This case should also be handled by missing_input_cols, but raise error if critical text feature is missing
         raise ValueError(f"Critical text feature column '{text_feature_used}' not found in prediction data.")


    # --- 5. Run Predictions ---
    print("Preprocessing data and running predictions...")
    try:
        # Pass only the columns the preprocessor expects
        X_processed = preprocessor.transform(df[required_input_cols])
        print(f"Prediction data processed. Shape: {X_processed.shape}")
    except Exception as e:
        print(f"Error during preprocessing transform: {e}")
        print("Ensure all required columns exist and have compatible data types.")
        raise e

    predictions = {}
    tags_predicted_for = training_params.get("tags_trained", SELECTED_TAGS)
    print(f"Predicting for tags: {tags_predicted_for}")

    for tag in tags_predicted_for:
        model_key = f'model_{tag}'
        if model_key in loaded_rf_models:
            model = loaded_rf_models[model_key]
            y_pred_class = model.predict(X_processed)
            predictions[f'pred_{tag}'] = y_pred_class
        else:
            print(f"Warning: Model for tag '{tag}' not found in loaded models dict. Setting predictions to 0.")
            predictions[f'pred_{tag}'] = 0

    pred_df = pd.DataFrame(predictions, index=df.index)

    for tag in SELECTED_TAGS:
        pred_col = f'pred_{tag}'
        if pred_col not in pred_df.columns:
             pred_df[pred_col] = 0

    pred_df = pred_df[[f'pred_{tag}' for tag in SELECTED_TAGS]]

    # --- 6. Format and Save Results ---
    print("Formatting and saving results...")
    # Check for ID column
    id_col_present = PROBLEM_ID_COLUMN in df.columns
    if not id_col_present:
         print(f"Warning: Problem ID column '{PROBLEM_ID_COLUMN}' not found in input CSV.")

    # Check for Ground Truth columns
    actual_gt_cols = [col for col in GROUND_TRUTH_TAG_COLUMNS if col in df.columns]
    gt_cols_present = bool(actual_gt_cols)
    if not gt_cols_present:
         print(f"Warning: Ground truth tag columns ({GROUND_TRUTH_TAG_COLUMNS}) not found in input CSV.")

    # Prepare results DataFrame
    cols_to_include_orig = []
    if id_col_present:
        cols_to_include_orig.append(PROBLEM_ID_COLUMN)
    if gt_cols_present:
        cols_to_include_orig.extend(actual_gt_cols)

    if not cols_to_include_orig:
        results_df = pred_df.copy() # Only save predictions
    else:
        # Ensure indices match before concatenating
        results_df = pd.concat([df[cols_to_include_orig].reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)


    output_dir = get_incremental_directory_path(TF_IDF_PREDICTIONS_DIR, prefix="tfidf_rf_sampled_")
    predictions_csv_path = os.path.join(output_dir, "predictions.csv")
    info_json_path = os.path.join(output_dir, "input_data_info.json")

    results_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved to: {predictions_csv_path}")

    info_data = {
        "input_csv_path": os.path.abspath(args.csv_path),
        "model_type": "TF-IDF + RandomForest (Numerical Features + Undersampling)",
        "model_loaded_from_dir": os.path.abspath(model_dir),
        "training_params_loaded": training_params,
        "prediction_timestamp": datetime.now().isoformat(),
        "total_examples_predicted": len(df),
        "sampled_data_input": args.sample_size > 0,
        "sample_size": args.sample_size if args.sample_size > 0 else "all"
    }
    with open(info_json_path, "w") as f:
        json.dump(info_data, f, indent=4, default=lambda x: x.item() if isinstance(x, np.generic) else (str(x) if isinstance(x, (np.ndarray, np.number)) else x))
    print(f"Input data info saved to: {info_json_path}")

    # --- 6. Optional: Trigger Evaluation ---
    if actual_gt_cols and args.run_evaluation:
        print("\nRunning evaluation...")
        try:
            evaluation_results = run_evaluation_function(predictions_csv_path)
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
    parser = argparse.ArgumentParser(description="Run Prediction using TF-IDF + RF model (JSON Params).")
    parser.add_argument(
        "--csv_path", type=str, default=TEST_DATA_CSV,
        help=f"Path to the input CSV file for prediction. Defaults to {TEST_DATA_CSV}"
    )
    parser.add_argument(
        "--model_dir_path", type=str, default=None,
        help="Path to the specific saved model directory (containing 'training_params.json', 'preprocessor.joblib', 'models.joblib'). If None, uses the latest found."
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
    args = parser.parse_args()
    main(args)
