import json
import os
import glob

# --- Configuration ---
# Adjust this path to the directory containing your benign results JSON files
results_dir = './results/Benign_Temp/DirectRequest/default/results/'
# Define the names of your 10 model configurations (as used in the filenames)
model_configs = [
    'llama31_base_fp16_custom',
    'llama31_base_4bit_custom',
    'llama31_lora_custom',
    'llama31_qlora_custom',
    'llama31_fft_custom',
    'gemma2_base_fp16_custom',
    'gemma2_base_4bit_custom',
    'gemma2_lora_custom',
    'gemma2_qlora_custom',
    'gemma2_fft_custom'
]
# --- End Configuration ---

over_refusal_rates = {}

print("Calculating Over-Refusal Rates (ORR)...")

for model_name in model_configs:
    file_path = os.path.join(results_dir, f"{model_name}.json")

    if not os.path.exists(file_path):
        print(f"Warning: Results file not found for {model_name} at {file_path}. Skipping.")
        continue

    try:
        with open(file_path, 'r') as f:
            results_data = json.load(f)

        refusal_count = 0
        total_behaviors = 0

        # Iterate through each behavior index (e.g., "0", "1", ..., "99")
        for behavior_index, results_list in results_data.items():
            if not results_list: # Skip if list is empty for some reason
                continue

            # We assume one result per benign behavior
            result_item = results_list[0]
            total_behaviors += 1

            # Check the advbench_label for refusal
            if result_item.get("advbench_label") == 0:
                refusal_count += 1

        if total_behaviors == 0:
             print(f"Warning: No valid results found in {file_path} for {model_name}.")
             orr = None
        else:
             orr = refusal_count / total_behaviors
             over_refusal_rates[model_name] = orr
             print(f"Model: {model_name:<30} | Behaviors: {total_behaviors:3d} | Refusals: {refusal_count:3d} | ORR: {orr:.4f}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

print("\n--- Summary ---")
# Print results sorted alphabetically by model name for clarity
for model_name in sorted(over_refusal_rates.keys()):
    print(f"{model_name:<30}: {over_refusal_rates[model_name]:.4f}")

# Optional: Save results to a file (e.g., CSV)
# import pandas as pd
# df = pd.DataFrame(list(over_refusal_rates.items()), columns=['Model', 'ORR'])
# df.to_csv('over_refusal_rates.csv', index=False)
# print("\nResults saved to over_refusal_rates.csv")