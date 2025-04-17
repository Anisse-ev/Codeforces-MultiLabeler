import argparse
import os
import json
import re
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import warnings

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", message=".*Could not find the quantized class.*") # Ignore BitsAndBytes warnings if expected

# --- Config Imports ---

from config.links_and_paths import TRAIN_DATA_CSV, MODELS_DIR
from config.data_config import (
        SELECTED_TAGS, TEXT_FEATURE_COLUMNS, CODE_FEATURE_COLUMNS
    )


# --- Project-Specific Imports ---
try:
    from src.utils import get_incremental_directory_path
except ImportError:
    # Basic fallback if utils not found
    def get_incremental_directory_path(base_dir, prefix=""):
        today = datetime.today().strftime("%Y%m%d")
        i = 1
        while True:
            path = os.path.join(base_dir, f"{prefix}{today}_{i:03d}")
            if not os.path.exists(path):
                # Ensure base_dir exists before creating subdirs
                os.makedirs(base_dir, exist_ok=True)
                os.makedirs(path, exist_ok=True)
                return path
            i += 1
    print("Warning: src.utils not found, using basic fallback for directory naming.")


# --- Library Imports ---
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from peft import LoraConfig, TaskType, get_peft_model # Standard PEFT import
# Import Unsloth's FastLanguageModel for its PEFT application utility
# Note: Ensure Unsloth is installed (`pip install "unsloth[...] @ git+...`)
try:
    from unsloth import FastLanguageModel
    unsloth_available = True
except ImportError:
    FastLanguageModel = None
    unsloth_available = False
    print("Warning: Unsloth not found. Standard PEFT will be used if applicable, but Unsloth-specific optimizations are disabled.")


# --- Script Configuration ---
MODEL_OUTPUT_SUBDIR = "qwen_peft_classification" # Subdirectory within MODELS_DIR
DEFAULT_BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct" # Default base model from notebook

# Default PEFT parameters (from notebook)
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = None # Let Unsloth/PEFT auto-detect by default
DEFAULT_LORA_BIAS = "none"

# Default Training parameters (from notebook)
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4 # Per device
DEFAULT_GRAD_ACC = 4
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_OPTIMIZER = "adamw_torch_fused"

# --- Helper Functions ---

def create_input_text(example):
    """Combines relevant text features into a single input string."""
    text_parts = []
    for col in TEXT_FEATURE_COLUMNS + CODE_FEATURE_COLUMNS:
        content = example.get(col)
        if pd.notna(content): text_parts.append(str(content))
    return "\n\n".join(text_parts).strip()

def create_multi_hot_labels(example):
    """Converts 'output' column (JSON string) to multi-hot float vector."""
    num_labels = len(SELECTED_TAGS)
    labels = [0.0] * num_labels
    output_str = example.get('output', '')
    if not output_str or not isinstance(output_str, str): return {"labels": labels}
    try:
        match = re.search(r"<output>\s*(\{.*?\})\s*</output>", output_str, re.DOTALL | re.IGNORECASE)
        if match:
            json_text = match.group(1); json_data = json.loads(json_text)
            current_labels = [float(json_data.get(tag, 0)) for tag in SELECTED_TAGS]
            if all(l in [0.0, 1.0] for l in current_labels): labels = current_labels
    except Exception: pass
    return {"labels": labels}

def tokenize_function(examples, tokenizer, max_length):
    """Tokenizes the 'input_text' field."""
    return tokenizer(
        examples["input_text"], padding="max_length", truncation=True, max_length=max_length
    )

def compute_metrics(eval_pred):
    """Compute metrics callback for Trainer."""
    predictions, labels = eval_pred
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits)).numpy()
    y_pred = np.zeros(probs.shape); y_pred[np.where(probs >= 0.5)] = 1
    y_true = np.array(labels).astype(int)

    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    subset_accuracy = accuracy_score(y_true, y_pred)

    metrics = {'f1_micro': f1_micro, 'f1_macro': f1_macro, 'subset_accuracy': subset_accuracy}
    return metrics

def convert_numpy_types(obj):
    """Helper function to convert NumPy types for JSON serialization."""
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# --- Main Training Function ---

def main(args):
    # --- Determine Device and Dtype ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = False; use_bf16 = False; model_dtype = torch.float32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported(): use_bf16 = True; model_dtype = torch.bfloat16
        else: use_fp16 = True; model_dtype = torch.float16
    print(f"Device: {device}, Use FP16: {use_fp16}, Use BF16: {use_bf16}")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer pad_token: {tokenizer.pad_token}")

    # --- Configure Quantization (Optional) ---
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype, bnb_4bit_use_double_quant=True,
        )
        print("4-bit quantization configured.")
    elif args.load_in_8bit:
         bnb_config = BitsAndBytesConfig(load_in_8bit=True)
         print("8-bit quantization configured.")


    # --- Load Base Model ---
    print(f"Loading base model {args.base_model} for multi-label classification...")
    num_labels = len(SELECTED_TAGS)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        torch_dtype=model_dtype,
        quantization_config=bnb_config,
        token=args.hf_token
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Base model loaded.")

    # --- Configure and Apply PEFT LoRA ---
    # Use Unsloth's get_peft_model if available and desired for potential optimizations
    if unsloth_available and args.use_unsloth_peft:
        print("Applying PEFT LoRA adapters using Unsloth...")
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.lora_r,
            target_modules = args.lora_target_modules, # None lets Unsloth auto-detect
            lora_alpha = args.lora_alpha,
            lora_dropout = args.lora_dropout,
            bias = args.lora_bias,
            task_type = TaskType.SEQ_CLS,
            use_gradient_checkpointing = "unsloth", # Unsloth specific optimization
            random_state = 42,
            max_seq_length = args.max_seq_length,
            auto_find_all_linears = (args.lora_target_modules is None)
        )
    else:
        # Use standard PEFT library
        print("Applying PEFT LoRA adapters using standard PEFT library...")
        if args.lora_target_modules:
            target_modules = args.lora_target_modules
        else:
            # Basic auto-detection attempt for standard PEFT (might need adjustment per model)
            target_modules = DEFAULT_LORA_TARGET_MODULES # Fallback to defaults
            print(f"Warning: Standard PEFT auto-detection not implemented, using defaults: {target_modules}")
            # More robust auto-detection would involve inspecting model layers

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias=args.lora_bias,
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # --- Load and Prepare Data ---
    print(f"Loading data from: {args.train_csv_path}")
    if not os.path.exists(args.train_csv_path):
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv_path}")
    df_train = pd.read_csv(args.train_csv_path)

    required_cols_data = TEXT_FEATURE_COLUMNS + CODE_FEATURE_COLUMNS + ['output'] + SELECTED_TAGS
    missing_cols = [col for col in required_cols_data if col not in df_train.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in training data: {missing_cols}")

    if args.subset_size > 0:
        print(f"Using subset of {args.subset_size} training examples.")
        df_train = df_train.sample(n=args.subset_size, random_state=42).reset_index(drop=True)

    print("Converting to Hugging Face Dataset and processing...")
    train_dataset_hf = Dataset.from_pandas(df_train)
    num_proc = min(os.cpu_count() // 2 if os.cpu_count() else 1, 4) if len(df_train) > 1000 else 1

    train_dataset_hf = train_dataset_hf.map(lambda x: {"input_text": create_input_text(x)}, num_proc=num_proc)
    train_dataset_hf = train_dataset_hf.map(create_multi_hot_labels, num_proc=num_proc)
    train_dataset_tokenized = train_dataset_hf.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_seq_length),
        batched=True, num_proc=num_proc, desc="Tokenizing Train"
    )

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    train_dataset_final = train_dataset_tokenized.remove_columns(
        [col for col in train_dataset_tokenized.column_names if col not in columns_to_keep]
    )
    train_dataset_final.set_format("torch")
    print("Dataset preparation complete.")

    # Split for validation
    eval_dataset_final = None; evaluation_strategy = "no"; save_strategy = "epoch"; load_best = False; metric_best = None
    if args.validation_split_percentage > 0 and len(train_dataset_final) > 10:
        print(f"Splitting training data for validation ({args.validation_split_percentage}%)...")
        split_dataset = train_dataset_final.train_test_split(test_size=args.validation_split_percentage / 100.0, seed=42)
        train_dataset_final = split_dataset["train"]
        eval_dataset_final = split_dataset["test"]
        print(f"Train size: {len(train_dataset_final)}, Validation size: {len(eval_dataset_final)}")
        evaluation_strategy = "epoch"; save_strategy = "epoch"; load_best = True; metric_best = "f1_micro"
    else:
        print("No validation split performed or dataset too small.")

    # --- Configure Trainer ---
    print("Configuring Trainer...")
    output_base_dir = os.path.join(MODELS_DIR, MODEL_OUTPUT_SUBDIR, "runs")
    run_prefix = f"{args.base_model.split('/')[-1]}_cls_peft_"
    output_dir_run = get_incremental_directory_path(output_base_dir, prefix=run_prefix)
    logging_dir = os.path.join(output_dir_run, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    print(f"Run output directory: {output_dir_run}")

    training_args = TrainingArguments(
        output_dir=output_dir_run,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_acc,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=1,
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_best,
        greater_is_better=True if metric_best else None,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=args.optimizer,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_final,
        eval_dataset=eval_dataset_final,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_dataset_final else None,
    )

    # --- Train ---
    print("Starting training...")
    try:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("Training complete.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback; traceback.print_exc()
        try: trainer.save_state(); print("Trainer state saved despite error.")
        except Exception as save_e: print(f"Could not save trainer state after error: {save_e}")

    # --- Save Final Model Adapters & Config ---
    final_save_path = os.path.join(output_dir_run, "final_model_adapters")
    print(f"Saving final model adapters to: {final_save_path}")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    # Save training parameters/metadata
    training_run_params = {
        "timestamp": datetime.now().isoformat(),
        "base_model": args.base_model,
        "task_type": "multi_label_sequence_classification",
        "peft_config": {
             "r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
             "target_modules": args.lora_target_modules or "auto-detected/default", # Indicate if default/auto was used
             "bias": args.lora_bias, "task_type": "SEQ_CLS"
        },
        "quantization": "4-bit" if args.load_in_4bit else ("8-bit" if args.load_in_8bit else "None"),
        "training_args": training_args.to_sanitized_dict(),
        "train_data_path": os.path.abspath(args.train_csv_path),
        "subset_size_used": args.subset_size if args.subset_size > 0 else "all",
        "final_adapter_path": os.path.abspath(final_save_path),
        "selected_tags": SELECTED_TAGS,
        "text_features_used": TEXT_FEATURE_COLUMNS,
        "code_features_used": CODE_FEATURE_COLUMNS,
        "max_seq_length": args.max_seq_length,
    }
    params_save_path = os.path.join(final_save_path, "training_run_params.json")
    try:
        with open(params_save_path, 'w') as f:
            json.dump(training_run_params, f, indent=4, default=convert_numpy_types)
        print(f"Training parameters saved to: {params_save_path}")
    except TypeError as e:
        print(f"Error saving parameters to JSON: {e}")

    # --- Final Evaluation (if validation set exists) ---
    if eval_dataset_final:
        print("\nEvaluating the final model on the validation set...")
        eval_results = trainer.evaluate(eval_dataset=eval_dataset_final)
        print("\n--- Final Validation Set Evaluation Results ---")
        for key, value in eval_results.items(): print(f"{key}: {value:.4f}")
        eval_metrics_path = os.path.join(final_save_path, "eval_results_validation.json")
        with open(eval_metrics_path, "w") as f: json.dump(eval_results, f, indent=4)
        print(f"Validation evaluation results saved to {eval_metrics_path}")

    print(f"\nTraining run finished. Artifacts saved in: {output_dir_run}")
    return output_dir_run # Return path to saved artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model for multi-label classification using PEFT LoRA.")

    # Model Args
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base model name or path.")
    parser.add_argument("--load_in_4bit", action='store_true', default=True, help="Load model in 4-bit quantization.")
    parser.add_argument("--load_in_8bit", action='store_true', help="Load model in 8-bit quantization.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (optional).")

    # Data Args
    parser.add_argument("--train_csv_path", type=str, default=TRAIN_DATA_CSV, help="Path to training CSV data.")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Maximum tokenizer sequence length.")
    parser.add_argument("--subset_size", type=int, default=-1, help="Use only a subset of training data (-1 for all).")
    parser.add_argument("--validation_split_percentage", type=float, default=10.0, help="Percentage of training data for validation (0 to disable).")

    # Training Args
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Per-device training batch size.")
    parser.add_argument("--grad_acc", type=int, default=DEFAULT_GRAD_ACC, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO, help="Warmup ratio.")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--optimizer", type=str, default=DEFAULT_OPTIMIZER, help="Optimizer to use.")

    # PEFT Args
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", nargs='+', default=None, help="List of LoRA target modules (e.g., q_proj v_proj). Default uses model defaults/auto-detect.")
    parser.add_argument("--lora_bias", type=str, default=DEFAULT_LORA_BIAS, help="LoRA bias type ('none', 'all', 'lora_only').")
    parser.add_argument("--use_unsloth_peft", action='store_true', help="Use Unsloth's get_peft_model (requires unsloth install).")


    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Cannot use both --load_in_4bit and --load_in_8bit.")
    if args.use_unsloth_peft and not unsloth_available:
         print("Warning: --use_unsloth_peft specified but unsloth library not found. Using standard PEFT.")
         args.use_unsloth_peft = False


    # Call the main training function
    main(args)
