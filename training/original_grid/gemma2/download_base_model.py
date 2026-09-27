import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Configuration ---
model_id = "google/gemma-2-9b-it"
save_directory = "models/gemma2-9b/gemma2-9b-it-base"
# ---------------------

print(f"Downloading model '{model_id}' to '{save_directory}'...")
os.makedirs(save_directory, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_directory)
print("Tokenizer downloaded and saved.")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    # Authentication for gated models comes from `huggingface-cli login` or HF_TOKEN.
)
model.save_pretrained(save_directory)
print("Model downloaded and saved successfully.")