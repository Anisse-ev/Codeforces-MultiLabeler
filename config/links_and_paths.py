import os

RAW_DATA_LINK = "https://drive.google.com/file/d/1FyNPiEKkZBfhz4ua0eM3PQjbRWVwO95J/view"
RAW_DATA_DIR = "data/raw_data"
CLEAN_DATA_DIR = "data/clean_data"
TRAIN_SPLIT_DATA_DIR = "data/train_test_split"
TRAINED_MODELS_DIR = "models/trained_models"
TRAIN_TEST_DATA_FILE_NAMES = {
    "train": "codeforces_train_data.csv",
    "test": "codeforces_test_data.csv",
    "zero_shot": "codeforces_zero_shot_data.csv"
}
# config/links_and_paths.py

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "clean_data")
TRAIN_TEST_SPLIT_DIR = os.path.join(DATA_DIR, "train_test_split")

TRAIN_DATA_CSV = os.path.join(TRAIN_TEST_SPLIT_DIR, "codeforces_train_data.csv")
TEST_DATA_CSV = os.path.join(TRAIN_TEST_SPLIT_DIR, "codeforces_test_data.csv")
ZERO_SHOT_DATA_CSV = os.path.join(TRAIN_TEST_SPLIT_DIR, "codeforces_zero_shot_data.csv") # If used

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(SRC_DIR, "models")

# --- Zero-Shot Model ---
ZERO_SHOT_MODEL_DIR = os.path.join(MODELS_DIR, "zero_shot_Qwen2_5")
ZERO_SHOT_SAVED_MODEL_PATH = os.path.join(ZERO_SHOT_MODEL_DIR, "saved_model") # Directory to save the base model
ZERO_SHOT_PREDICTIONS_DIR = os.path.join(ZERO_SHOT_MODEL_DIR, "predictions")

# --- Fine-Tuned Model ---
FINE_TUNED_MODEL_DIR = os.path.join(MODELS_DIR, "fine_tuned_Qwen2_5")
FINE_TUNED_LORA_WEIGHTS_DIR = os.path.join(FINE_TUNED_MODEL_DIR, "lora_weights")
FINE_TUNED_PREDICTIONS_DIR = os.path.join(FINE_TUNED_MODEL_DIR, "predictions")

# --- Model Names ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"