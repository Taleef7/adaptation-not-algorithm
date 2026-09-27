#!/usr/bin/env python3
"""
Convert DeepInception JSONL completions to LlamaGuard-compatible JSON format.
Maps behavior_goal text to behavior_id using the behaviors CSV.
Also renames 'response' field to 'generation' for LlamaGuard script compatibility.
"""

import json
import csv
import sys
import os

if len(sys.argv) != 4:
    print("Usage: python convert_for_llamaguard.py input.jsonl behaviors.csv output.json")
    sys.exit(1)

input_file = sys.argv[1]
behaviors_file = sys.argv[2]
output_file = sys.argv[3]

# Load behaviors CSV to create goal -> ID mapping
goal_to_id = {}
with open(behaviors_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        behavior_text = row.get('Behavior', '').strip()
        behavior_id = row.get('BehaviorID', '').strip()
        if behavior_text and behavior_id:
            goal_to_id[behavior_text.lower()] = behavior_id

print(f"Loaded {len(goal_to_id)} behaviors from CSV")

# Read JSONL file and add behavior_id, rename response -> generation
completions = []
matched = 0
unmatched = 0

with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            behavior_goal = item.get('behavior_goal', '').strip()
            
            # Try to find matching behavior_id
            behavior_id = goal_to_id.get(behavior_goal.lower())
            
            if behavior_id:
                item['behavior_id'] = behavior_id
                matched += 1
            else:
                # Try partial matching
                for goal, bid in goal_to_id.items():
                    if goal in behavior_goal.lower() or behavior_goal.lower() in goal:
                        item['behavior_id'] = bid
                        matched += 1
                        break
                else:
                    item['behavior_id'] = 'unknown'
                    unmatched += 1
            
            # CRITICAL FIX: Rename 'response' to 'generation' for LlamaGuard compatibility
            if 'response' in item and 'generation' not in item:
                item['generation'] = item['response']
            
            completions.append(item)

# Write as JSON list
with open(output_file, 'w') as f:
    json.dump(completions, f, indent=2)

print(f"Converted {len(completions)} completions")
print(f"  Matched: {matched}")
print(f"  Unmatched: {unmatched}")
print(f"  Added 'generation' field from 'response'")
print(f"Output saved to: {output_file}")
