import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Configuration ---
model_id = "Qwen/Qwen3-4B-Instruct-2507"
save_directory = "models/qwen3-4b/qwen3-4b-instruct-base"
# ---------------------

print(f"Downloading model '{model_id}' to '{save_directory}'...")

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Load the tokenizer from the Hub
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Save the tokenizer to the local directory
tokenizer.save_pretrained(save_directory)
print("Tokenizer downloaded and saved.")

# Load the model from the Hub (CPU mode for login node compatibility)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # Memory efficient loading
)
# Save the model to the local directory
model.save_pretrained(save_directory)
print(f"Model downloaded and saved successfully to {save_directory}")
print("You can now run the fine-tuning jobs!")
