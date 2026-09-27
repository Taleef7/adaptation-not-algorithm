#!/usr/bin/env python3
"""Merge a seed-42 controlled-grid adapter into its bf16 base model for inference."""
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO = os.environ.get("PROJECT_ROOT", ".")
BASE = {
 "llama31": f"{REPO}/models/llama3.1/llama-3.1-8b-instruct-base",
 "gemma2":  f"{REPO}/models/gemma2-9b/gemma2-9b-it-base",
 "qwen3":   f"{REPO}/models/qwen3-4b/qwen3-4b-instruct-base",
 "phi4":    f"{REPO}/models/phi-4/phi-4-base",
 "qwen25":  f"{REPO}/models/qwen2.5-14b/qwen2.5-14b-instruct-base",
 # Defended-base arms: the MixAT-merged checkpoints, so an adapter trained on top of a
 # defended base merges back into that same defended base, not the undefended one.
 "llama31_mixat": f"{REPO}/models/mixat/llama31_mixat_base",
 "qwen25_mixat":  f"{REPO}/models/mixat/qwen25_mixat_base",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(BASE))
    ap.add_argument("--method", required=True, choices=["lora", "qlora"])
    a = ap.parse_args()
    adapter = f"{REPO}/models/phase5/{a.family}_{a.method}_seed42"
    merged = f"{REPO}/models/phase5/{a.family}_{a.method}_seed42_merged"
    tok_kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    print(f"==> merging {adapter} into {BASE[a.family]}")
    base = AutoModelForCausalLM.from_pretrained(BASE[a.family], torch_dtype=torch.bfloat16,
                                                device_map="cpu", **tok_kw)
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
    print(f"==> merged model saved to {merged}")


if __name__ == "__main__":
    main()
