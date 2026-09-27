#!/usr/bin/env python3
"""
Convert ActorAttack JSON outputs to HarmBench JSON format.

ActorAttack stores results in multiple JSON files iteratively:
toolkits/ActorAttack/attack_result/{model_config}_{idx}_{timestamp}.json

This script:
1. Gathers all JSON results for a given model config.
2. For each unique behavior (by 'instruction'), takes the latest run.
3. Maps 'instruction' to 'BehaviorID' using the ActorAttack behavior CSV.
4. Extracts the final target model generation (query_chain) from attempts.
5. Saves the output to toolkits/HarmBench/results/ActorAttack/{model_config}/completions/{model_config}.json

This enables standard HarmBench evaluators (Llama-2, LlamaGuard-3, GPT-4o-mini) to process ActorAttack output.
"""

import argparse
import csv
import glob
import json
import os
import sys

# Model configurations
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
    
    # Qwen3 14B
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

HARMBENCH_CSV = "toolkits/ActorAttack/data/harmbench.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ActorAttack outputs to HarmBench completions.")
    parser.add_argument(
        "--model-config",
        type=str,
        default="",
        help="Optional single model config to convert (default: convert all known models).",
    )
    parser.add_argument(
        "--min-behaviors",
        type=int,
        default=0,
        help="Optional minimum behavior count required for each converted model (0 disables gate).",
    )
    parser.add_argument(
        "--behavior-csv",
        type=str,
        default=os.getenv(
            "ACTORATTACK_BEHAVIOR_CSV",
            "toolkits/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        ),
        help="Path to behavior CSV used for instruction→BehaviorID mapping (accepts Goal or Behavior column).",
    )
    parser.add_argument(
        "--raw-result-root",
        type=str,
        default=os.getenv("ACTORATTACK_RAW_RESULT_ROOT", "toolkits/ActorAttack/attack_result"),
        help="Root directory containing raw ActorAttack JSON outputs.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=os.getenv("ACTORATTACK_RESULTS_ROOT", "toolkits/HarmBench/results"),
        help="Root directory for HarmBench-style converted outputs.",
    )
    parser.add_argument(
        "--attack-name",
        type=str,
        default=os.getenv("ACTORATTACK_ATTACK_NAME", "ActorAttack"),
        help="Attack subdirectory name under the results root.",
    )
    return parser.parse_args()

def load_behavior_mapping(csv_path=None):
    """Create a dictionary mapping Goal/Behavior text to BehaviorID.

    Accepts either 'Goal' (ActorAttack-project CSV) or 'Behavior'
    (canonical harmbench_behaviors_text_all.csv) as the behavior column.
    """
    if csv_path is None:
        csv_path = HARMBENCH_CSV
    mapping = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                goal = row.get('Goal') or row.get('Behavior')
                behavior_id = row.get('BehaviorID')
                if goal and behavior_id:
                    mapping[goal.strip()] = behavior_id
    else:
        print(f"Warning: behavior CSV not found: {csv_path}")
    return mapping


def extract_generation(attempt):
    """Prefer the final assistant message from dialog history; fall back to actor query_chain."""
    dialog_hist = attempt.get("dialog_hist")
    if isinstance(dialog_hist, list):
        for turn in reversed(dialog_hist):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                content = turn.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content

    actor_data = attempt.get("actor", {})
    if isinstance(actor_data, dict):
        generation = actor_data.get("query_chain", "")
        if isinstance(generation, str) and generation.strip():
            return generation

    return "[ActorAttack conversion fallback] No assistant generation captured for this behavior."

def construct_test_case(attempt):
    """Reconstruct the prompt/exchange that led to the response for logging purposes."""
    actor_info = attempt.get("actor", {})
    queries = actor_info.get("queries", [])
    if isinstance(queries, list):
        queries = [str(q) for q in queries if str(q).strip()]
        if queries:
            return "\n".join(queries)

    dialog_hist = attempt.get("dialog_hist", [])
    if isinstance(dialog_hist, list):
        user_prompts = [
            turn.get("content", "")
            for turn in dialog_hist
            if isinstance(turn, dict)
            and turn.get("role") == "user"
            and isinstance(turn.get("content", ""), str)
            and turn.get("content", "").strip()
        ]
        if user_prompts:
            return "\n".join(user_prompts)

    return str(queries)

def convert_actorattack_for_model(
    model_config,
    behavior_mapping,
    raw_result_root,
    results_root,
    attack_name,
    min_behaviors=0,
):
    """Find all output files for the model, extract best attempts, write to HarmBench format."""

    candidate_patterns = [
        os.path.join(raw_result_root, model_config, f"{model_config}_*.json"),
        os.path.join(raw_result_root, model_config, "*.json"),
        os.path.join(raw_result_root, f"{model_config}_*.json"),
    ]
    files = []
    for pattern in candidate_patterns:
        files = glob.glob(pattern)
        if files:
            break

    if not files:
        print(f"⚠ Skipping {model_config}: no result files found under {raw_result_root}")
        return False, 0
        
    print(f"Loading {len(files)} files for {model_config}...")
    
    # Group by instruction, keep the latest completion based on timestamp
    # Filename format: llama31_base_fp16_custom_idx_YYYY-MM-DD_HH:MM:SS.json
    instruction_to_latest = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_obj = json.load(f)
                
            if "data" not in data_obj:
                continue
                
            for item in data_obj["data"]:
                instruction = item.get("instruction", "").strip()
                if not instruction:
                    continue
                    
                attempts = item.get("attempts", [])
                if attempts:
                    # Assume the last attempt in the list is the final one.
                    last_attempt = attempts[-1]
                    generation = extract_generation(last_attempt)
                    test_case = construct_test_case(last_attempt)
                else:
                    # Preserve coverage for sparse behaviors where no attempt was serialized.
                    generation = "[ActorAttack conversion fallback] No attempts generated for this behavior."
                    test_case = instruction
                
                # Use file mod time or filename to determine latest
                mod_time = os.path.getmtime(file_path)
                
                if instruction not in instruction_to_latest or instruction_to_latest[instruction]["mtime"] < mod_time:
                    instruction_to_latest[instruction] = {
                        "mtime": mod_time,
                        "generation": generation,
                        "test_case": test_case,
                    }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if not instruction_to_latest:
        print(f"⚠ No valid attempts parsed for {model_config}")
        return False, 0
        
    # Convert to HarmBench Dict format
    harmbench_completions = {}
    missing_mapping_count = 0
    
    for instruction, details in instruction_to_latest.items():
        behavior_id = behavior_mapping.get(instruction)
        if not behavior_id:
            # Fallback if mapping fails
            missing_mapping_count += 1
            behavior_id = f"unknown_behavior_{missing_mapping_count}"
            print(f"Warning: Could not map instruction to BehaviorID: '{instruction[:50]}...'")
            
        harmbench_completions[behavior_id] = [{
            "test_case": details["test_case"],
            "generation": details["generation"]
        }]
        
    output_dir = os.path.join(results_root, attack_name, model_config, "completions")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{model_config}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(harmbench_completions, f, indent=4)
        
    behavior_count = len(harmbench_completions)
    print(f"✓ Converted {model_config}: {behavior_count} behaviors mapped -> {output_path}")
    if min_behaviors > 0 and behavior_count < min_behaviors:
        print(
            f"✗ Coverage gate failed for {model_config}: "
            f"{behavior_count} < required minimum {min_behaviors}"
        )
        return False, behavior_count

    return True, behavior_count

def main():
    args = parse_args()
    print("=" * 80)
    print("Converting ActorAttack Outputs to HarmBench Format")
    print("=" * 80)
    print()
    
    behavior_mapping = load_behavior_mapping(csv_path=args.behavior_csv)
    if not behavior_mapping:
        print("ERROR: Failed to load behavior mapping. Aborting.")
        return
        
    print(f"Loaded {len(behavior_mapping)} behavior mappings.")
    
    target_models = MODELS
    if args.model_config:
        target_models = [args.model_config]

    converted = 0
    skipped = 0
    total = len(target_models)
    gate_failures = 0
    
    for model in target_models:
        result, _ = convert_actorattack_for_model(
            model,
            behavior_mapping,
            args.raw_result_root,
            args.results_root,
            args.attack_name,
            min_behaviors=args.min_behaviors,
        )
        if result:
            converted += 1
        else:
            skipped += 1
            if args.min_behaviors > 0:
                gate_failures += 1
            
    print()
    print("=" * 80)
    print(f"Conversion Summary:")
    print(f"  Total models:   {total}")
    print(f"  Converted:      {converted}")
    print(f"  Skipped/No Data:{skipped}")
    if args.min_behaviors > 0:
        print(f"  Coverage gates failed: {gate_failures}")
    print("=" * 80)

    if args.min_behaviors > 0 and gate_failures > 0:
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
