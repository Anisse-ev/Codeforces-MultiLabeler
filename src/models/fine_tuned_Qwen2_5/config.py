# Configuration parameters for fine-tuning

LOAD_IN_4BIT = True
DTYPE = None 
# --- LoRA Configuration ---
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

USE_GRADIENT_CHECKPOINTING = "unsloth" # Or True
BIAS = "none"
USE_RSLORA = False
LOFTQ_CONFIG = None
RANDOM_STATE = 3407

# --- Training Arguments ---
TRAINING_ARGS_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    # "num_train_epochs": 1, 
    "max_steps": 60,        # 
    "learning_rate": 2e-4,

    "logging_steps": 1,
    "optim": "adamw_8bit", # Needs bitsandbytes
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": RANDOM_STATE,
    "output_dir": "outputs", 
}


# --- Other ---
EOS_TOKEN = "<|endoftext|>" 