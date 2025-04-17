import argparse
import os
import joblib
import json # Import json
import pandas as pd
import numpy as np
from datetime import datetime


from config.links_and_paths import TRAIN_DATA_CSV, TF_IDF_SAVED_MODEL_PATH
from config.data_config import SELECTED_TAGS, OTHER_FEATURE_COLUMNS, TEXT_FEATURE_COLUMNS

# Scikit-learn & Imblearn Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler # Import undersampler

# Utility Imports
try:
    from src.utils import get_incremental_directory_path
except ImportError:
    # Simple fallback if utils not found (adjust logic if needed)
    def get_incremental_directory_path(base_dir, prefix=""):
        today = datetime.today().strftime("%Y%m%d")
        i = 1
        while True:
            path = os.path.join(base_dir, f"{prefix}{today}_{i:03d}")
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                return path
            i += 1
    print("Warning: src.utils not found, using basic fallback for directory naming.")


# --- Configuration ---
TEXT_FEATURE_FOR_TFIDF = TEXT_FEATURE_COLUMNS[0] if TEXT_FEATURE_COLUMNS else 'problem_description'
TFIDF_MAX_FEATURES = 5000
TFIDF_MAX_DF = 0.8
TFIDF_MIN_DF = 5
TFIDF_NGRAM_RANGE = (1, 2)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
RF_N_JOBS = -1

def create_preprocessing_pipeline(tfidf_params, numerical_features):
    """Creates the ColumnTransformer preprocessing pipeline."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    text_transformer = TfidfVectorizer(
        max_df=tfidf_params['max_df'],
        min_df=tfidf_params['min_df'],
        max_features=tfidf_params['max_features'],
        ngram_range=tfidf_params['ngram_range'],
        stop_words='english'
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, TEXT_FEATURE_FOR_TFIDF),
            # Only include 'num' transformer if numerical features are available
            ('num', numerical_transformer, numerical_features) if numerical_features else ('drop_num', 'drop', numerical_features)
        ],
        remainder='drop'
    )
    return preprocessor


def main(args):
    # --- 1. Load Data ---
    print(f"Loading training data from: {args.train_csv_path}")
    if not os.path.exists(args.train_csv_path):
        raise FileNotFoundError(f"Training CSV not found at {args.train_csv_path}")
    df_train = pd.read_csv(args.train_csv_path)

    if args.subset_size > 0:
        print(f"Using a subset of {args.subset_size} examples for training.")
        df_train = df_train.sample(n=args.subset_size, random_state=42).reset_index(drop=True)

    # --- 2. Define Features and Preprocessing ---
    tfidf_params = {
        'max_df': TFIDF_MAX_DF, 'min_df': TFIDF_MIN_DF,
        'max_features': TFIDF_MAX_FEATURES, 'ngram_range': TFIDF_NGRAM_RANGE,
        'stop_words': 'english' # Include stop_words in params
    }
    rf_params = {
        'n_estimators': RF_N_ESTIMATORS,
        'max_depth': RF_MAX_DEPTH,
        'random_state': RF_RANDOM_STATE,
        'n_jobs': RF_N_JOBS
    }

    available_numerical_features = [col for col in OTHER_FEATURE_COLUMNS if col in df_train.columns]
    if available_numerical_features:
        print(f"Using numerical features: {available_numerical_features}")
        for col in available_numerical_features:
             df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
             df_train[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        print("Warning: No numerical feature columns found.")

    if TEXT_FEATURE_FOR_TFIDF not in df_train.columns:
        raise ValueError(f"Required text feature column '{TEXT_FEATURE_FOR_TFIDF}' not found.")
    df_train[TEXT_FEATURE_FOR_TFIDF] = df_train[TEXT_FEATURE_FOR_TFIDF].fillna('')

    preprocessor = create_preprocessing_pipeline(tfidf_params, available_numerical_features)

    # --- 3. Train Model for Each Tag with Undersampling ---
    trained_rf_models = {} # Dictionary to hold only the trained RF models
    tags_trained = [] # Keep track of tags for which models were trained
    print("Fitting preprocessor and training models for each tag...")

    print("Fitting preprocessor...")
    X_processed = preprocessor.fit_transform(df_train)
    print(f"Data preprocessed. Shape: {X_processed.shape}")

    # Preprocessor is now fitted and ready to be saved later

    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=RF_RANDOM_STATE)

    for tag in SELECTED_TAGS:
        if tag not in df_train.columns:
            print(f"Warning: Tag column '{tag}' not found. Skipping training.")
            continue

        print(f"Processing tag: {tag}...")
        y_target = df_train[tag].fillna(0).astype(int)

        if len(np.unique(y_target)) < 2:
             print(f"Skipping resampling and training for tag '{tag}': Only one class present.")
             continue # Skip training if only one class exists
        else:
             try:
                  print(f"Applying RandomUnderSampler for tag: {tag}...")
                  X_resampled, y_resampled = undersampler.fit_resample(X_processed, y_target)
                  print(f"Resampled data shape for tag '{tag}': {X_resampled.shape}")
                  print(f"Resampled target distribution for '{tag}': {np.bincount(y_resampled)}")
             except ValueError as e:
                  print(f"Warning: Could not apply undersampling for tag '{tag}': {e}. Using original data.")
                  X_resampled, y_resampled = X_processed, y_target

        # Define and Train the RandomForest model
        rf_model = RandomForestClassifier(**rf_params) # Use defined params
        print(f"Training RandomForest model for tag: {tag}...")
        rf_model.fit(X_resampled, y_resampled)

        # Store the trained model in the dictionary
        trained_rf_models[f'model_{tag}'] = rf_model
        tags_trained.append(tag) # Add tag to list of successfully trained models
        print(f"Model for tag '{tag}' trained.")

    # --- 4. Save Preprocessor, Models, and Parameters ---
    save_dir = get_incremental_directory_path(TF_IDF_SAVED_MODEL_PATH, prefix="tfidf_rf_sampled_")

    # Define file paths
    preprocessor_save_path = os.path.join(save_dir, "preprocessor.joblib")
    models_save_path = os.path.join(save_dir, "models.joblib")
    params_save_path = os.path.join(save_dir, "training_params.json")

    # Save the fitted preprocessor
    print(f"Saving preprocessor to: {preprocessor_save_path}")
    joblib.dump(preprocessor, preprocessor_save_path)

    # Save the dictionary of trained RandomForest models
    print(f"Saving trained models to: {models_save_path}")
    joblib.dump(trained_rf_models, models_save_path)

    # Create dictionary for training parameters/metadata
    params_to_save = {
        "timestamp": datetime.now().isoformat(),
        "training_data_path": os.path.abspath(args.train_csv_path),
        "subset_size_used": args.subset_size if args.subset_size > 0 else "all",
        "text_feature_used": TEXT_FEATURE_FOR_TFIDF,
        "numerical_features_used": available_numerical_features,
        "tfidf_params": tfidf_params,
        "rf_params": rf_params,
        "sampling_method": "RandomUnderSampler",
        "tags_trained": tags_trained,
        # Store relative paths to the saved objects within the save_dir
        "preprocessor_path": os.path.basename(preprocessor_save_path),
        "models_path": os.path.basename(models_save_path),
    }

    # Save the parameters as JSON
    print(f"Saving training parameters to: {params_save_path}")
    with open(params_save_path, 'w') as f:
        json.dump(params_to_save, f, indent=4)

    print("Models and parameters saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF + RandomForest models with numerical features and undersampling.")
    parser.add_argument(
        "--train_csv_path", type=str, default=TRAIN_DATA_CSV,
        help=f"Path to the training CSV file. Defaults to {TRAIN_DATA_CSV}",
    )
    parser.add_argument(
        "--subset_size", type=int, default=-1,
        help="Number of examples to use for training (-1 for full dataset).",
    )
    args = parser.parse_args()
    main(args)
