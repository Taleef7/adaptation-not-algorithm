#!/usr/bin/env python3
"""
Merge Phi-4 PEFT adapters with base model for HarmBench compatibility.

HarmBench uses vLLM for inference, which doesn't support loading PEFT adapters directly.
This script merges the adapter weights into the base model weights, creating standalone
models that vLLM can load.

Merges the Phi-4 adapters (loads the base in float16 with GPU/CPU offload):
1. phi-4-lora-alpaca (FP16 LoRA)
2. phi-4-qlora-alpaca (4-bit QLoRA)
"""

import os
import sys
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Base paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
PHI4_BASE = os.path.join(PROJECT_ROOT, "models/phi-4/phi-4-base")

# Adapter configurations - only Phi-4 models
ADAPTERS = [
    {
        "name": "phi-4-lora-alpaca-merged",
        "adapter_path": os.path.join(PROJECT_ROOT, "models/phi-4/phi-4-lora-alpaca"),
        "base_model": PHI4_BASE,
        "output_path": os.path.join(PROJECT_ROOT, "models/phi-4/phi-4-lora-alpaca-merged"),
        "load_in_4bit": False,
    },
    {
        "name": "phi-4-qlora-alpaca-merged",
        "adapter_path": os.path.join(PROJECT_ROOT, "models/phi-4/phi-4-qlora-alpaca"),
        "base_model": PHI4_BASE,
        "output_path": os.path.join(PROJECT_ROOT, "models/phi-4/phi-4-qlora-alpaca-merged"),
        "load_in_4bit": True,
    },
]


def merge_adapter(adapter_config):
    """Merge a single PEFT adapter with its base model using device_map=auto for memory efficiency."""
    print("\n" + "=" * 80)
    print(f"Merging adapter: {adapter_config['adapter_path']}")
    print(f"With base model: {adapter_config['base_model']}")
    print(f"Output path: {adapter_config['output_path']}")
    print(f"Load in 4-bit: {adapter_config['load_in_4bit']}")
    print("=" * 80 + "\n")

    try:
        # Clear GPU memory before starting
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load base model with automatic device mapping and CPU offload
        print("Loading base model with device_map='auto' and max_memory settings...")
        
        # Set max memory to use GPU efficiently but allow CPU offload
        max_memory = {0: "70GiB", "cpu": "100GiB"}  # A100 has 80GB, leave some margin
        
        if adapter_config["load_in_4bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                adapter_config["base_model"],
                quantization_config=quantization_config,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                adapter_config["base_model"],
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

        print(f"Base model loaded. Device map: {base_model.hf_device_map}")

        # Load PEFT adapter
        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(
            base_model, 
            adapter_config["adapter_path"],
        )

        # Merge adapter weights
        print("Merging adapter weights into base model (this may take several minutes)...")
        merged_model = model.merge_and_unload()

        # Save merged model
        print(f"Saving merged model to {adapter_config['output_path']}...")
        merged_model.save_pretrained(
            adapter_config["output_path"],
            safe_serialization=True,
            max_shard_size="5GB",  # Split into smaller shards
        )

        # Save tokenizer
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_config["base_model"])
        tokenizer.save_pretrained(adapter_config["output_path"])

        print(f"✓ Successfully created merged model at {adapter_config['output_path']}")
        
        # Clean up aggressively
        del base_model
        del model
        del merged_model
        torch.cuda.empty_cache()
        gc.collect()
        
        return True

    except Exception as e:
        print(f"✗ Error merging {adapter_config['adapter_path']}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        torch.cuda.empty_cache()
        gc.collect()
        return False


def main():
    """Main function to merge all adapters."""
    print("\n" + "=" * 40)
    print("Merging PEFT Adapters for HarmBench")
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'N/A')}")
    print(f"GPU: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
    print("=" * 40 + "\n\n")

    print(f"Will merge {len(ADAPTERS)} adapter(s):\n")
    for i, adapter in enumerate(ADAPTERS, 1):
        print(f"{i}. {adapter['name']}")
    print("\n")

    success_count = 0
    failed_adapters = []

    for adapter in ADAPTERS:
        if merge_adapter(adapter):
            success_count += 1
        else:
            failed_adapters.append(adapter["name"])

    print("\n" + "=" * 80)
    print("Merging complete!")
    print("=" * 80)
    print(f"\nSuccessfully merged: {success_count}/{len(ADAPTERS)}")
    
    if failed_adapters:
        print(f"\nFailed adapters:")
        for name in failed_adapters:
            print(f"  - {name}")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("All adapters merged successfully!")
    print("=" * 40)


if __name__ == "__main__":
    main()
