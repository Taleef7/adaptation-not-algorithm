#!/usr/bin/env python3
"""
Merge LoRA/QLoRA adapters with the base model for Qwen2.5-14B.
Creates standalone merged models that can be used for inference without PEFT.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os

def merge_adapter(base_model_path, adapter_path, output_path):
    """Merge adapter weights into base model."""
    print(f"\n{'='*60}")
    print(f"Merging adapter:")
    print(f"  Base model: {base_model_path}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model
    print("[2/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load adapter
    print("[3/4] Loading and merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    
    # Save merged model
    print("[4/4] Saving merged model...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ Merged model saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA/QLoRA adapters")
    parser.add_argument("--adapter_type", choices=["lora", "qlora", "both"], default="both")
    args = parser.parse_args()
    
    base_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "models/qwen2.5-14b")
    base_model = f"{base_path}/qwen2.5-14b-instruct-base"
    
    if args.adapter_type in ["lora", "both"]:
        merge_adapter(
            base_model,
            f"{base_path}/qwen2.5-14b-lora-alpaca",
            f"{base_path}/qwen2.5-14b-lora-alpaca-merged"
        )
    
    if args.adapter_type in ["qlora", "both"]:
        merge_adapter(
            base_model,
            f"{base_path}/qwen2.5-14b-qlora-alpaca",
            f"{base_path}/qwen2.5-14b-qlora-alpaca-merged"
        )
    
    print("\n" + "="*60)
    print("All merges completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
