import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Configuration ---
model_id = "meta-llama/Llama-3.1-8B-Instruct"
save_directory = "models/llama3.1/llama-3.1-8b-instruct-base" # The new local directory
# ---------------------

print(f"Downloading model '{model_id}' to '{save_directory}'...")

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Load the tokenizer from the Hub
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Save the tokenizer to the local directory
tokenizer.save_pretrained(save_directory)
print("Tokenizer downloaded and saved.")

# Load the model from the Hub
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto", # Temporarily load to GPU to ensure compatibility
)
# Save the model to the local directory
model.save_pretrained(save_directory)
print("Model downloaded and saved successfully.")