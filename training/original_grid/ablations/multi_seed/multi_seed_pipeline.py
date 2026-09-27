#!/usr/bin/env python3
"""
Post-training step for the Llama-3.1 multi-seed cells (training seeds 0 and 123).
Run after the four seed training jobs (LoRA and FFT, seeds 0/123) finish.

Steps:
  check     verify that adapters / FFT checkpoints exist
  merge     merge the seed-0/123 LoRA adapters into the bf16 base (FFT needs no merge)
  register  print the HarmBench models.yaml entries for the seed cells
  all       check, then merge and register

Usage:
  python training/original_grid/ablations/multi_seed/multi_seed_pipeline.py --step merge
  python training/original_grid/ablations/multi_seed/multi_seed_pipeline.py --step all
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("PROJECT_ROOT", "."))
HARMBENCH = BASE / "toolkits" / "HarmBench"
MODELS_DIR = BASE / "models" / "llama3.1"

# New seed-variant models
SEED_MODELS = {
    "llama31_lora_seed0": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter_path": MODELS_DIR / "llama-3.1-8b-lora-alpaca-seed0",
        "merged_path": MODELS_DIR / "llama-3.1-8b-instruct-lora-merged-seed0",
        "is_lora": True,
    },
    "llama31_lora_seed123": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter_path": MODELS_DIR / "llama-3.1-8b-lora-alpaca-seed123",
        "merged_path": MODELS_DIR / "llama-3.1-8b-instruct-lora-merged-seed123",
        "is_lora": True,
    },
    "llama31_fft_seed0": {
        "model_path": MODELS_DIR / "llama-3.1-8b-fft-alpaca-hf-seed0",
        "is_lora": False,
    },
    "llama31_fft_seed123": {
        "model_path": MODELS_DIR / "llama-3.1-8b-fft-alpaca-hf-seed123",
        "is_lora": False,
    },
}

# Attack test cases to reuse from original seed
AUTODAN_TEST_CASES = {
    "HarmBench": HARMBENCH / "results" / "AutoDAN" / "llama31_base_fp16_custom" / "test_cases" / "test_cases.json",
    "JBB": HARMBENCH / "results" / "AutoDAN" / "llama31_base_fp16_custom" / "test_cases" / "test_cases.json",
    # We reuse test cases generated against the BASE model for cross-seed transfer analysis
}


def step_merge():
    """Merge LoRA adapters into full models for HarmBench compatibility."""
    print("=" * 60)
    print("STEP 1: Merging LoRA Adapters")
    print("=" * 60)
    
    for name, config in SEED_MODELS.items():
        if not config["is_lora"]:
            print(f"  Skipping {name} (FFT, no merge needed)")
            continue
        
        adapter_path = config["adapter_path"]
        merged_path = config["merged_path"]
        base_model = config["base_model"]
        
        if not adapter_path.exists():
            print(f"  ❌ SKIP: Adapter not found at {adapter_path}")
            print(f"     Training may not have completed yet.")
            continue
        
        if merged_path.exists():
            print(f"  ⚠️  Merged model already exists at {merged_path}, skipping")
            continue
        
        print(f"\n  Merging {name}...")
        print(f"    Base: {base_model}")
        print(f"    Adapter: {adapter_path}")
        print(f"    Output: {merged_path}")
        
        # Use Python to merge
        merge_script = f"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "{adapter_path}")

print("Merging...")
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained("{merged_path}")
tokenizer.save_pretrained("{merged_path}")
print("Done!")
"""
        result = subprocess.run(
            [sys.executable, "-c", merge_script],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ {name} merged successfully")
        else:
            print(f"  ❌ {name} merge failed:")
            print(result.stderr[-500:])


def step_register_models():
    """Print the model config entries to add to models.yaml."""
    print("=" * 60)
    print("STEP 2: Model Config Entries for models.yaml")
    print("=" * 60)
    print("\nAdd the following to HarmBench configs/model_configs/models.yaml:\n")
    
    for name, config in SEED_MODELS.items():
        if config["is_lora"]:
            model_path = config["merged_path"]
        else:
            model_path = config["model_path"]
        
        print(f"{name}_custom:")
        print(f"  model:")
        print(f"    model_name_or_path: {model_path}")
        print(f"    dtype: bfloat16")
        print(f"    fschat_template: alpaca")
        print(f'    eos_token: "<|eot_id|>"')
        print(f'    pad_token: "<|eot_id|>"')
        print(f"    max_model_len: 8192")
        print(f"  num_gpus: 1")
        print()


def step_check_readiness():
    """Check if all models are ready for evaluation."""
    print("=" * 60)
    print("READINESS CHECK")
    print("=" * 60)
    
    all_ready = True
    for name, config in SEED_MODELS.items():
        if config["is_lora"]:
            adapter_exists = config["adapter_path"].exists()
            merged_exists = config["merged_path"].exists()
            print(f"  {name}:")
            print(f"    Adapter: {'✅' if adapter_exists else '❌'} {config['adapter_path']}")
            print(f"    Merged:  {'✅' if merged_exists else '❌'} {config['merged_path']}")
            if not adapter_exists:
                all_ready = False
        else:
            model_exists = config["model_path"].exists()
            print(f"  {name}:")
            print(f"    Model: {'✅' if model_exists else '❌'} {config['model_path']}")
            if not model_exists:
                all_ready = False
    
    print(f"\n{'✅ All models ready!' if all_ready else '❌ Some models missing. Complete training first.'}")
    return all_ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["check", "merge", "register", "all"], required=True)
    args = parser.parse_args()
    
    if args.step == "check":
        step_check_readiness()
    elif args.step == "merge":
        step_merge()
    elif args.step == "register":
        step_register_models()
    elif args.step == "all":
        if step_check_readiness():
            step_merge()
            step_register_models()
        else:
            print("\nComplete training before running the full pipeline.")
