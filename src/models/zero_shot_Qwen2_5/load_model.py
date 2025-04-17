import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.links_and_paths import BASE_MODEL_NAME, ZERO_SHOT_SAVED_MODEL_PATH

def load_base_model(hf_token: str | None = None, force_redownload=False):
    """
    Loads the base model and tokenizer. Downloads if not present or if force_redownload is True.
    Saves the model locally to ZERO_SHOT_SAVED_MODEL_PATH.

    Args:
        hf_token (str | None): Optional Hugging Face token for private/gated models.
        force_redownload (bool): If True, deletes existing local model and redownloads.

    Returns:
        tuple: (model, tokenizer)
    """
    model_path = ZERO_SHOT_SAVED_MODEL_PATH
    model_exists = os.path.exists(os.path.join(model_path, "config.json")) # Basic check

    # Prepare arguments for from_pretrained, adding token only if provided
    load_kwargs = {
        "torch_dtype": torch.float16, # Adjust dtype based on available hardware
        "device_map": 'auto' # Automatically distribute across available GPUs/CPU
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    if model_exists and not force_redownload:
        print(f"Loading base model from local path: {model_path}")
        # No token needed usually when loading locally, unless files are just pointers
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=load_kwargs["torch_dtype"],
            device_map=load_kwargs["device_map"],
            local_files_only=True # Try loading only local files first
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    else:
        print(f"Downloading and saving base model '{BASE_MODEL_NAME}' to {model_path}")
        if force_redownload and model_exists:
            print("Forcing re-download, removing existing model files...")
            import shutil
            shutil.rmtree(model_path)

        os.makedirs(model_path, exist_ok=True)

        print(f"Attempting download with kwargs: { {k:v for k,v in load_kwargs.items() if k != 'token'} }") # Don't print token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            **load_kwargs
        )
        tokenizer_load_kwargs = {"token": hf_token} if hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, **tokenizer_load_kwargs)

        print("Saving model and tokenizer...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("Model and tokenizer saved successfully.")

    print(f"Base model '{BASE_MODEL_NAME}' loaded successfully.")
    print(f"Model is on device: {model.device}") # Verify device placement
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save the base LLM model.")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face API token for downloading private/gated models."
    )
    parser.add_argument(
        "--force_redownload",
        action='store_true',
        help="Force redownloading the model even if it exists locally."
    )
    args = parser.parse_args()

    # Load the model (will download if needed), passing the token
    load_base_model(hf_token=args.hf_token, force_redownload=args.force_redownload)
    print("Base model loading script finished.")
