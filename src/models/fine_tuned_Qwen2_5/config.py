# Configuration parameters for fine-tuning

# --- Model ---
# Base model path is determined by the zero-shot location
# BASE_MODEL_PATH = ZERO_SHOT_SAVED_MODEL_PATH (defined in links_and_paths.py)

# --- Quantization ---
# As used in the notebook
LOAD_IN_4BIT = True
DTYPE = None # None for auto detection (Float16 for T4/V100, Bfloat16 for Ampere+)

# --- LoRA Configuration ---
# As used in the notebook
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
# Use "unsloth" if using Unsloth library and it provides benefits, otherwise True/False
# Set to True if not using unsloth's specific optimization
USE_GRADIENT_CHECKPOINTING = "unsloth" # Or True
BIAS = "none"
USE_RSLORA = False
LOFTQ_CONFIG = None
RANDOM_STATE = 3407 # Seed for reproducibility

# --- Training Arguments ---
# Adapted from the notebook
# Consider adjusting max_steps or num_train_epochs for full runs
TRAINING_ARGS_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    # "num_train_epochs": 1, # Use for full training
    "max_steps": 60,        # Use for quick tests or initial runs
    "learning_rate": 2e-4,
    # fp16 / bf16 are typically handled by the trainer based on hardware support check
    # "fp16": calculated_based_on_support,
    # "bf16": calculated_based_on_support,
    "logging_steps": 1,
    "optim": "adamw_8bit", # Needs bitsandbytes
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": RANDOM_STATE,
    "output_dir": "outputs", # Temporary output dir for trainer logs/checkpoints
    # Note: SFTTrainer needs dataset_text_field, dataset_num_proc which are passed directly
}

# --- Dataset ---
# Path to the training data CSV (must contain the text column for SFTTrainer)
# TRAIN_DATA_PATH = TRAIN_DATA_CSV (defined in links_and_paths.py)
# Name of the column containing the pre-formatted chat string
# FINE_TUNING_TEXT_COLUMN = "text_for_fine_tuning" (defined in data_config.py)

# --- Saving ---
# Base directory for saving LoRA weights is defined in links_and_paths.py
# LORA_WEIGHTS_DIR = FINE_TUNED_LORA_WEIGHTS_DIR

# --- Other ---
EOS_TOKEN = "<|endoftext|>" # Define EOS token if tokenizer doesn't add it automatically in template