#!/usr/bin/env python3
"""Merge the released MixAT adapter into its base to produce a defended base model.

MixAT (Dekany et al., NeurIPS 2025; arXiv 2505.16947) releases its adversarially-trained
models as PEFT LoRA adapters rather than full weights. Two of the released adapters sit on
exactly the base models two of our five families use:

    INSAIT-Institute/Llama3.1-8B-MixAT  ->  meta-llama/Llama-3.1-8B-Instruct
    INSAIT-Institute/Qwen-14B-MixAT     ->  Qwen/Qwen2.5-14B-Instruct

so the undefended control for this experiment is already on disk: it is the ordinary
llama31 / qwen25 grid. Nothing has to be retrained to get the comparison.

Merging (rather than stacking a second adapter on top) is what makes the downstream
experiment interpretable: after merge_and_unload() the defended model is an ordinary dense
checkpoint, so the canonical LoRA and FFT recipes apply to it unchanged and the ONLY thing
that differs from the main grid is the starting weights.

Both adapters are plain LoRA (r=64, alpha=16, the standard 7 projection modules, no DoRA,
no rslora, no modules_to_save, no quantization in the adapter config), so the merge is exact
and needs no special handling. The 4-bit loading shown on the Qwen model card is a
convenience for inference, not a property of the adapter.
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.environ.get("PROJECT_ROOT", ".")

CFG = {
    "llama31": dict(
        base=f"{REPO}/models/llama3.1/llama-3.1-8b-instruct-base",
        adapter=f"{REPO}/models/mixat/llama31-mixat-adapter",
        out=f"{REPO}/models/mixat/llama31_mixat_base",
        expect="meta-llama/Llama-3.1-8B-Instruct",
    ),
    "qwen25": dict(
        base=f"{REPO}/models/qwen2.5-14b/qwen2.5-14b-instruct-base",
        adapter=f"{REPO}/models/mixat/qwen25-mixat-adapter",
        out=f"{REPO}/models/mixat/qwen25_mixat_base",
        expect="Qwen/Qwen2.5-14B-Instruct",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(CFG))
    a = ap.parse_args()
    cfg = CFG[a.family]

    # Safety check: refuse to merge if the adapter was trained on a different base than the one
    # we are about to merge it into. A silent base mismatch would produce a plausible-looking
    # model whose weights mean nothing.
    import json
    declared = json.load(open(os.path.join(cfg["adapter"], "adapter_config.json")))
    assert declared["base_model_name_or_path"] == cfg["expect"], (
        f"adapter declares base {declared['base_model_name_or_path']!r}, "
        f"expected {cfg['expect']!r}")
    print(f"==> adapter base check OK: {declared['base_model_name_or_path']} "
          f"(r={declared['r']}, alpha={declared['lora_alpha']})")

    tok_kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    print(f"==> merging {cfg['adapter']} into {cfg['base']}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.bfloat16, device_map="cpu", **tok_kw)
    model = PeftModel.from_pretrained(base, cfg["adapter"])
    model = model.merge_and_unload()
    model.save_pretrained(cfg["out"], safe_serialization=True)
    # The adapter repo ships no tokenizer; the base tokenizer is the right one by construction.
    AutoTokenizer.from_pretrained(cfg["base"], **tok_kw).save_pretrained(cfg["out"])
    print(f"==> merged defended base saved to {cfg['out']}")


if __name__ == "__main__":
    main()
