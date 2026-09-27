#!/usr/bin/env python3
"""Recipe-robustness (track 2): build the Dolly-15k dataset to disk in the SAME schema the canonical training
scripts expect (columns instruction/input/output), so train_track2_*.py can load it exactly
like the Alpaca dataset. Dolly fields: instruction, context -> input, response -> output.
Saves to $PROJECT_ROOT/data/dolly15k_dataset. Run once before any Dolly training arm.
"""
import os
from datasets import load_dataset

REPO = os.environ.get("PROJECT_ROOT", ".")
OUT = f"{REPO}/data/dolly15k_dataset"

tok_kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
ds = load_dataset("databricks/databricks-dolly-15k", split="train", **tok_kw)
print("raw dolly:", ds)

def to_alpaca_schema(ex):
    return {"instruction": ex["instruction"], "input": ex["context"], "output": ex["response"]}

ds = ds.map(to_alpaca_schema, remove_columns=ds.column_names)
assert set(ds.column_names) == {"instruction", "input", "output"}, ds.column_names
ds.save_to_disk(OUT)
print(f"saved {len(ds)} examples to {OUT}")
print("sample[0]:", {k: str(ds[0][k])[:80] for k in ds.column_names})
n_ctx = sum(1 for x in ds["input"] if x.strip())
print(f"examples with non-empty input/context: {n_ctx}/{len(ds)}")
