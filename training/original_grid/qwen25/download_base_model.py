# training/original_grid/qwen25/download_base_model.py
"""
Download the Qwen2.5-14B-Instruct model from Hugging Face.
This is a ~14B parameter instruction-tuned model from Alibaba.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Configuration ---
model_id = "Qwen/Qwen2.5-14B-Instruct"
save_directory = os.path.join(os.environ.get("PROJECT_ROOT", "."), "models/qwen2.5-14b/qwen2.5-14b-instruct-base")
# ---------------------

print(f"Downloading model '{model_id}' to '{save_directory}'...")
print("This model has ~14.7B parameters. Download may take ~30-60 minutes.")

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Load the tokenizer from the Hub
print("\n[1/2] Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Save the tokenizer to the local directory
tokenizer.save_pretrained(save_directory)
print("✓ Tokenizer downloaded and saved.")

# Load the model from the Hub (bfloat16 for efficiency)
print("\n[2/2] Downloading model weights (this will take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Memory efficient loading
)
# Save the model to the local directory
model.save_pretrained(save_directory)

print(f"\n{'='*60}")
print(f"✓ Model downloaded and saved successfully to:")
print(f"  {save_directory}")
print(f"{'='*60}")
print("You can now run the fine-tuning jobs!")
