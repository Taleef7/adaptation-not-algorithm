#!/usr/bin/env python3
"""
Convert DeepInception JSONL completions to JSON list format for LlamaGuard evaluation.
"""

import json
import sys

if len(sys.argv) != 3:
    print("Usage: python convert_jsonl_to_json.py input.jsonl output.json")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Read JSONL file
completions = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            completions.append(json.loads(line))

# Write as JSON list
with open(output_file, 'w') as f:
    json.dump(completions, f, indent=2)

print(f"Converted {len(completions)} completions from {input_file} to {output_file}")
