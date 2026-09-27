from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
import gc

# Define the models to merge
models_to_merge = {
    "gemma2_lora": {
        "base_model_path": "models/gemma2-9b/gemma2-9b-it-base",
        "adapter_path": "models/gemma2-9b/gemma2-9b-it-lora-alpaca",
        "output_path": "models/gemma2-9b/gemma2-9b-it-lora-merged"
    },
    "gemma2_qlora": {
        "base_model_path": "models/gemma2-9b/gemma2-9b-it-base",
        "adapter_path": "models/gemma2-9b/gemma2-9b-it-qlora-alpaca",
        "output_path": "models/gemma2-9b/gemma2-9b-it-qlora-merged"
    },
    "llama31_lora": {
        "base_model_path": "models/llama3.1/llama-3.1-8b-instruct-base",
        "adapter_path": "models/llama3.1/llama-3.1-8b-lora-alpaca",
        "output_path": "models/llama3.1/llama-3.1-8b-instruct-lora-merged"
    },
    "llama31_qlora": {
        "base_model_path": "models/llama3.1/llama-3.1-8b-instruct-base",
        "adapter_path": "models/llama3.1/llama-3.1-8b-qlora-alpaca-hf",
        "output_path": "models/llama3.1/llama-3.1-8b-instruct-qlora-merged"
    },
}


for name, paths in models_to_merge.items():
    print(f"--- Merging model: {name} ---")
    
    print(f"Loading base model from: {paths['base_model_path']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        paths['base_model_path'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading adapter from: {paths['adapter_path']}")
    peft_model = PeftModel.from_pretrained(base_model, paths['adapter_path'])
    
    print("Merging weights...")
    merged_model = peft_model.merge_and_unload()
    
    os.makedirs(paths['output_path'], exist_ok=True)
    print(f"Saving merged model to: {paths['output_path']}")
    merged_model.save_pretrained(paths['output_path'])

    # <<< MODIFIED: Load tokenizer from the BASE model path, not the adapter path >>>
    print("Loading tokenizer from base model path...")
    tokenizer = AutoTokenizer.from_pretrained(paths['base_model_path'])
    tokenizer.save_pretrained(paths['output_path'])
    
    print(f"--- Finished merging {name} ---")

    print("Cleaning up memory...")
    del base_model
    del peft_model
    del merged_model
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleaned up.\n")

print("All models merged successfully.")