#!/usr/bin/env python3
"""
Convert JBB-Behaviors dataset from HuggingFace to HarmBench CSV format.

This enables running HarmBench attacks (PAIR, ArtPrompt, AutoDAN) on JBB behaviors.

Output: toolkits/HarmBench/data/behavior_datasets/jbb_behaviors.csv
Format: Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID
"""

import pandas as pd
from datasets import load_dataset
import os

def convert_jbb_to_harmbench():
    """Load JBB from HuggingFace and convert to HarmBench CSV format"""
    
    print("Loading JBB-Behaviors dataset from HuggingFace...")
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    
    print(f"Loaded {len(dataset)} harmful behaviors")
    
    # Convert to DataFrame with HarmBench schema
    behaviors = []
    for idx, behavior in enumerate(dataset):
        behaviors.append({
            "Behavior": behavior["Goal"],                    # Goal → Behavior
            "FunctionalCategory": behavior.get("Category", "unknown"),  # Category → FunctionalCategory
            "SemanticCategory": "",                          # Not in JBB
            "Tags": "",                                      # Not in JBB
            "ContextString": "",                             # Not in JBB
            "BehaviorID": f"jbb_{idx:03d}"                  # Generate IDs
        })
    
    df = pd.DataFrame(behaviors)
    
    # Save to HarmBench data directory
    output_dir = "toolkits/HarmBench/data/behavior_datasets"
    output_path = os.path.join(output_dir, "jbb_behaviors.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Converted {len(df)} JBB behaviors to HarmBench format")
    print(f"✓ Saved to: {output_path}")
    print(f"\nFirst 3 behaviors:")
    print(df.head(3)[["Behavior", "FunctionalCategory", "BehaviorID"]])
    print(f"\nCategory distribution:")
    print(df["FunctionalCategory"].value_counts())
    
    return output_path

if __name__ == "__main__":
    convert_jbb_to_harmbench()
