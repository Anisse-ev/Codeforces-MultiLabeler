# src/models/fine_tuned_Qwen2_5/train_model.py
import argparse
import os
import json
import pandas as pd
import torch
from datetime import datetime

# Third-party Libraries
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, AutoTokenizer
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer

# Project Imports
from config.links_and_paths import (
    TRAIN_DATA_CSV, ZERO_SHOT_SAVED_MODEL_PATH,
    FINE_TUNED_LORA_WEIGHTS_DIR, BASE_MODEL_NAME
)
from config.data_config import FINE_TUNING_TEXT_COLUMN
from src.models.fine_tuned_Qwen2_5.config import (
    LOAD_IN_4BIT, DTYPE, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, USE_GRADIENT_CHECKPOINTING, BIAS, USE_RSLORA,
    LOFTQ_CONFIG, RANDOM_STATE, TRAINING_ARGS_CONFIG, EOS_TOKEN
)
from src.models.zero_shot_Qwen2_5.load_model import load_base_model # Ensure base model exists
# Import the utility function
from src.utils import get_incremental_directory_path

def main(args):
    # --- 1. Ensure Base Model Exists ---
    if not os.path.exists(os.path.join(ZERO_SHOT_SAVED_MODEL_PATH, "config.json")):
        print("Base model not found locally. Running load_model script first...")
        # Pass token if provided to the loading script as well
        load_base_model(hf_token=args.hf_token)
    else:
        print(f"Using base model from: {ZERO_SHOT_SAVED_MODEL_PATH}")

    # --- 2. Load Base Model and Tokenizer using Unsloth ---
    print("Loading base model with Unsloth for fine-tuning...")
    # Prepare arguments for Unsloth loader
    unsloth_load_kwargs = {
        "model_name": ZERO_SHOT_SAVED_MODEL_PATH, # Load the already saved base model
        "max_seq_length": args.max_seq_length,
        "dtype": DTYPE,
        "load_in_4bit": LOAD_IN_4BIT,
    }
    if args.hf_token:
        unsloth_load_kwargs["token"] = args.hf_token

    model, tokenizer = FastLanguageModel.from_pretrained(**unsloth_load_kwargs)

    # Set EOS token if needed and not automatically handled by chat template
    if tokenizer.eos_token is None:
        print(f"Tokenizer does not have EOS token set. Setting to {EOS_TOKEN}")
        tokenizer.eos_token = EOS_TOKEN
    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         print(f"Setting pad_token to eos_token: {tokenizer.pad_token}")

    # --- 3. Add LoRA Adapters ---
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        target_modules = LORA_TARGET_MODULES,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,
        bias = BIAS,
        use_gradient_checkpointing = USE_GRADIENT_CHECKPOINTING,
        random_state = RANDOM_STATE,
        use_rslora = USE_RSLORA,
        loftq_config = LOFTQ_CONFIG,
    )
    print("LoRA adapters applied.")
    model.print_trainable_parameters()

    # --- 4. Load and Prepare Dataset ---
    print(f"Loading training data from: {args.train_csv_path}")
    if not os.path.exists(args.train_csv_path):
        raise FileNotFoundError(f"Training CSV not found at {args.train_csv_path}")

    # Check if it's a CSV or Hugging Face dataset format
    if args.train_csv_path.endswith('.csv'):
        df = pd.read_csv(args.train_csv_path)
        if FINE_TUNING_TEXT_COLUMN not in df.columns:
             raise ValueError(f"Required column '{FINE_TUNING_TEXT_COLUMN}' not found in {args.train_csv_path}.")
        dataset = Dataset.from_pandas(df)
    else:
        print(f"Attempting to load dataset from Hugging Face path: {args.train_csv_path}")
        dataset_load_kwargs = {"token": args.hf_token} if args.hf_token else {}
        dataset = load_dataset(args.train_csv_path, split="train", **dataset_load_kwargs)
        if FINE_TUNING_TEXT_COLUMN not in dataset.column_names:
             raise ValueError(f"Required column '{FINE_TUNING_TEXT_COLUMN}' not found in dataset {args.train_csv_path}.")

    # Optional: Select a subset for quick testing
    if args.subset_size > 0:
         print(f"Using a subset of {args.subset_size} examples for training.")
         dataset = dataset.select(range(args.subset_size))

    # --- 5. Configure Trainer ---
    training_args_dict = TRAINING_ARGS_CONFIG.copy()
    training_args_dict["bf16"] = is_bfloat16_supported()
    training_args_dict["fp16"] = not training_args_dict["bf16"]
    training_args_dict["max_steps"] = args.max_steps if args.max_steps is not None and args.max_steps > 0 else training_args_dict.get("max_steps", -1)

    # Handle epochs vs max_steps logic
    if training_args_dict["max_steps"] > 0 :
        training_args_dict.pop("num_train_epochs", None) # max_steps overrides epochs
        print(f"Training for a maximum of {training_args_dict['max_steps']} steps.")
    else:
        training_args_dict.pop("max_steps", None) # Use epochs if max_steps is not set positively
        training_args_dict["num_train_epochs"] = args.num_epochs if args.num_epochs is not None else training_args_dict.get("num_train_epochs", 1)
        print(f"Training for {training_args_dict['num_train_epochs']} epochs.")


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = FINE_TUNING_TEXT_COLUMN,
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(**training_args_dict),
    )
    print("SFTTrainer configured.")

    # --- 6. Train ---
    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training finished.")
    print(f"Trainer stats: {trainer_stats}")

    # --- 7. Save LoRA Weights and Config ---
    print("Saving LoRA adapter weights...")

    # *** Use the utility function to get the save path ***
    # This will create and return a directory path like:
    # .../lora_weights/lora_adapter_YYYYMMDD_001/
    lora_save_path = get_incremental_directory_path(
        base_dir=FINE_TUNED_LORA_WEIGHTS_DIR,
        prefix="lora_adapter_" # This prefix will come before the date
    )
    # The utility function already creates the directory.

    # Save the LoRA adapter weights (PEFT model)
    trainer.model.save_pretrained(lora_save_path)
    # Also save the tokenizer configuration alongside the adapter for consistency
    tokenizer.save_pretrained(lora_save_path)

    print(f"LoRA adapter weights and tokenizer saved to: {lora_save_path}")

    # Save training parameters/configuration
    params = {
        "base_model_name": BASE_MODEL_NAME,
        "base_model_path_used": ZERO_SHOT_SAVED_MODEL_PATH,
        "lora_adapter_path": os.path.abspath(lora_save_path),
        "training_data_path": os.path.abspath(args.train_csv_path),
        "fine_tuning_text_column": FINE_TUNING_TEXT_COLUMN,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": LOAD_IN_4BIT,
        "lora_config": {
            "r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES, "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
            "random_state": RANDOM_STATE, "use_rslora": USE_RSLORA, "bias": BIAS,
        },
        "training_args": trainer.args.to_sanitized_dict(),
        "trainer_stats": str(trainer_stats),
        "timestamp": datetime.now().isoformat(),
    }
    params_save_path = os.path.join(lora_save_path, "training_params.json")
    with open(params_save_path, "w") as fp:
        json.dump(params, fp, indent=4)
    print(f"Training parameters saved to: {params_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model using LoRA and Unsloth.")
    parser.add_argument(
        "--train_csv_path", type=str, default=TRAIN_DATA_CSV,
        help=f"Path to the training CSV/Dataset containing '{FINE_TUNING_TEXT_COLUMN}'. Defaults to {TRAIN_DATA_CSV}"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048,
        help="Maximum sequence length for tokenization."
    )
    parser.add_argument(
        "--subset_size", type=int, default=-1,
        help="Number of examples to use for training (-1 for full dataset)."
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Maximum number of training steps (overrides num_epochs if set > 0)."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None,
        help="Number of training epochs (used if max_steps is not set > 0)."
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Optional Hugging Face API token for model/dataset download."
    )

    args = parser.parse_args()

    # Basic validation for steps/epochs
    if args.max_steps is not None and args.max_steps <= 0:
        args.max_steps = None # Treat 0 or negative as unset
    if args.num_epochs is not None and args.num_epochs <= 0:
        args.num_epochs = None # Treat 0 or negative as unset
    if args.max_steps is None and args.num_epochs is None:
        print("Warning: Neither --max_steps (>0) nor --num_epochs (>0) provided. Using defaults from config.")
        # Rely on TRAINING_ARGS_CONFIG defaults


    main(args)
