#!/usr/bin/env python3
"""
Convert DeepInception JSONL outputs to HarmBench JSON format.

DeepInception stores results in JSONL format:
{
  "attack": "DeepInception",
  "config_name": "gemma2_base_fp16_custom",
  "behavior_goal": "...",
  "category": "...",
  "prompt": "...",
  "response": "..."
}

HarmBench expects JSON format:
[{
  "behavior_id": "jbb_000",
  "behavior": "...",
  "prompt": "...",
  "response": "...",
  "model": "gemma2_base_fp16_custom"
}]

This conversion enables Llama-2 classifier evaluation of DeepInception/JBB results.
"""

import json
import os
from pathlib import Path

# Model configurations (4 families × 5 methods)
MODELS = [
    # Gemma-2 9B
    "gemma2_base_fp16_custom",
    "gemma2_base_4bit_custom",
    "gemma2_lora_custom",
    "gemma2_qlora_custom",
    "gemma2_fft_custom",
    
    # Llama 3.1 8B
    "llama31_base_fp16_custom",
    "llama31_base_4bit_custom",
    "llama31_lora_custom",
    "llama31_qlora_custom",
    "llama31_fft_custom",
    
    # Qwen3 4B
    "qwen3_base_fp16_custom",
    "qwen3_base_4bit_custom",
    "qwen3_lora_custom",
    "qwen3_qlora_custom",
    "qwen3_fft_custom",
    
    # Phi-4 14B
    "phi4_base_fp16_custom",
    "phi4_base_4bit_custom",
    "phi4_lora_custom",
    "phi4_qlora_custom",
    "phi4_fft_custom",
]

# Family to directory mapping
FAMILY_MAP = {
    "gemma2": "gemma2-9b",
    "llama31": "llama3.1",
    "qwen3": "qwen3-4b",
    "phi4": "phi-4"
}

def convert_deepinception_output(model_config):
    """Convert DeepInception JSONL to HarmBench JSON format"""
    
    # Get family directory
    family = model_config.split('_')[0]
    family_dir = FAMILY_MAP[family]
    
    # Input: JSONL from DeepInception
    input_path = f"outputs/deepinception/{family_dir}/judged/{model_config}.jsonl"
    
    # Output: JSON in HarmBench format
    output_path = f"outputs/deepinception/{family_dir}/{model_config}_harmbench_format.json"
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"⚠ Skipping {model_config}: input not found")
        return False
    
    # Check if already converted
    if os.path.exists(output_path):
        print(f"✓ Skipping {model_config}: already converted")
        return True
    
    # Load JSONL
    completions = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                completions.append(json.loads(line))
    
    # Convert to HarmBench format
    harmbench_completions = []
    for i, comp in enumerate(completions):
        harmbench_completions.append({
            "behavior_id": f"jbb_{i:03d}",
            "behavior": comp["behavior_goal"],
            "prompt": comp["prompt"],
            "response": comp["response"],
            "model": model_config
        })
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(harmbench_completions, f, indent=2)
    
    print(f"✓ Converted {model_config}: {len(harmbench_completions)} completions")
    return True

def main():
    """Convert all DeepInception outputs to HarmBench format"""
    
    print("=" * 80)
    print("Converting DeepInception Outputs to HarmBench Format")
    print("=" * 80)
    print()
    
    total = len(MODELS)
    converted = 0
    skipped = 0
    
    for model in MODELS:
        result = convert_deepinception_output(model)
        if result:
            if "Skipping" in str(result):
                skipped += 1
            else:
                converted += 1
    
    print()
    print("=" * 80)
    print(f"Conversion Summary:")
    print(f"  Total models:   {total}")
    print(f"  Converted:      {converted}")
    print(f"  Already done:   {skipped}")
    print(f"  Missing:        {total - converted - skipped}")
    print("=" * 80)

if __name__ == "__main__":
    main()
