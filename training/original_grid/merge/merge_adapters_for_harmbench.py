#!/usr/bin/env python3
"""
Merge PEFT adapters with base models for HarmBench compatibility.

HarmBench uses vLLM for inference, which doesn't support loading PEFT adapters directly.
This script merges the adapter weights into the base model weights, creating standalone
models that vLLM can load.

Covers the Phi-4 and Qwen3-4B LoRA/QLoRA adapters (select with --model / --method).
In the original grid it produced the Qwen3 merged checkpoints (bf16); the Phi-4 merged
checkpoints were produced by merge_phi4_adapters.py.
"""

import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Base paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
PHI4_BASE = os.path.join(PROJECT_ROOT, "models/phi-4/phi-4-base")

def merge_and_save(base_model_path, adapter_path, output_path, load_in_4bit=False):
    """
    Merge PEFT adapter with base model and save to disk.
    
    Args:
        base_model_path: Path to base model
        adapter_path: Path to PEFT adapter
        output_path: Where to save merged model
        load_in_4bit: Whether base was trained with 4-bit quantization (QLoRA)
    """
    print(f"\n{'='*80}")
    print(f"Merging adapter: {adapter_path}")
    print(f"With base model: {base_model_path}")
    print(f"Output path: {output_path}")
    print(f"Load in 4-bit: {load_in_4bit}")
    print(f"{'='*80}\n")
    
    # Load base model
    print("Loading base model...")
    if load_in_4bit:
        # For QLoRA, need to load with same quantization config as training
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        # For LoRA, load in full precision then merge
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    # Load adapter
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge and unload
    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ Successfully created merged model at {output_path}\n")
    
    # Cleanup
    del model
    del base_model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT adapters for HarmBench")
    parser.add_argument("--models-dir", type=str, 
                        default=os.path.join(os.environ.get("PROJECT_ROOT", "."), "models"),
                        help="Base directory containing model subdirectories")
    parser.add_argument("--model", type=str, choices=["phi4", "qwen3", "both"], default="both",
                        help="Which model family to merge")
    parser.add_argument("--method", type=str, choices=["lora", "qlora", "both"], default="both",
                        help="Which PEFT method to merge")
    args = parser.parse_args()
    
    # Define model configurations
    configs = []
    
    if args.model in ["phi4", "both"]:
        base_phi4 = os.path.join(args.models_dir, "phi-4", "phi-4-base")
        if args.method in ["lora", "both"]:
            configs.append({
                "base": base_phi4,
                "adapter": os.path.join(args.models_dir, "phi-4", "phi-4-lora-alpaca"),
                "output": os.path.join(args.models_dir, "phi-4", "phi-4-lora-alpaca-merged"),
                "load_in_4bit": False
            })
        if args.method in ["qlora", "both"]:
            configs.append({
                "base": base_phi4,
                "adapter": os.path.join(args.models_dir, "phi-4", "phi-4-qlora-alpaca"),
                "output": os.path.join(args.models_dir, "phi-4", "phi-4-qlora-alpaca-merged"),
                "load_in_4bit": True
            })
    
    if args.model in ["qwen3", "both"]:
        base_qwen3 = os.path.join(args.models_dir, "qwen3-4b", "qwen3-4b-instruct-base")
        if args.method in ["lora", "both"]:
            configs.append({
                "base": base_qwen3,
                "adapter": os.path.join(args.models_dir, "qwen3-4b", "qwen3-4b-lora-alpaca"),
                "output": os.path.join(args.models_dir, "qwen3-4b", "qwen3-4b-lora-alpaca-merged"),
                "load_in_4bit": False
            })
        if args.method in ["qlora", "both"]:
            configs.append({
                "base": base_qwen3,
                "adapter": os.path.join(args.models_dir, "qwen3-4b", "qwen3-4b-qlora-alpaca"),
                "output": os.path.join(args.models_dir, "qwen3-4b", "qwen3-4b-qlora-alpaca-merged"),
                "load_in_4bit": True
            })
    
    # Process each configuration
    print(f"\nWill merge {len(configs)} adapter(s):\n")
    for i, config in enumerate(configs, 1):
        print(f"{i}. {os.path.basename(config['output'])}")
    print()
    
    for config in configs:
        try:
            merge_and_save(
                config["base"],
                config["adapter"],
                config["output"],
                config["load_in_4bit"]
            )
        except Exception as e:
            print(f"✗ Error merging {config['adapter']}: {str(e)}\n")
            continue
    
    print("\n" + "="*80)
    print("Merging complete!")
    print("="*80)


if __name__ == "__main__":
    main()
