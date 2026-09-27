#!/usr/bin/env python3
"""
Faithful IBM-style SafeLoRA projection for LoRA adapters.

This implementation follows the adapter-level logic from the IBM SafeLoRA
reference repository while using a mathematically equivalent low-rank form for
the projection step:

    P = delta @ delta.T / ||delta||
    P @ B = delta @ (delta.T @ B) / ||delta||

This avoids materializing the full projection matrix for wide layers while
remaining faithful to the published method.
"""

import argparse
import json
from pathlib import Path

import numpy
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_alignment_projection(base_weight, aligned_weight):
    delta = (aligned_weight - base_weight).float()
    delta_norm = torch.norm(delta)
    if delta_norm < 1e-8:
        return None
    return torch.mm(delta, delta.t()) / delta_norm


def apply_projection_to_lora_b(lora_a, lora_b, projection=None, alignment_delta=None):
    lora_a = lora_a.float()
    lora_b = lora_b.float()

    if projection is not None:
        projected_b = torch.mm(projection.float(), lora_b)
    elif alignment_delta is not None:
        delta = alignment_delta.float()
        delta_norm = torch.norm(delta)
        if delta_norm < 1e-8:
            projected_b = lora_b.clone()
        else:
            projected_b = torch.mm(delta, torch.mm(delta.t(), lora_b)) / delta_norm
    else:
        raise ValueError("Either projection or alignment_delta must be provided.")

    projected_full = torch.mm(projected_b, lora_a)
    original_full = torch.mm(lora_b, lora_a)
    cosine = torch.nn.functional.cosine_similarity(
        projected_full.reshape(1, -1),
        original_full.reshape(1, -1),
    ).item()
    return projected_b, cosine


def threshold_from_num_proj_layers(cosines, num_proj_layers):
    if not cosines:
        raise ValueError("cosines must not be empty")
    count = min(max(1, num_proj_layers), len(cosines))
    return float(numpy.sort(cosines)[:count][-1])


def resolve_model_reference(model_name_or_path):
    path = Path(model_name_or_path)
    if path.exists():
        return str(path.resolve())
    return snapshot_download(repo_id=model_name_or_path)


def normalize_lora_module_name(module_name):
    normalized = module_name
    while normalized.startswith("base_model.model."):
        normalized = normalized[len("base_model.model.") :]
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(description="Faithful SafeLoRA projection")
    parser.add_argument("--base_model_path", required=True,
                        help="Unaligned base model path or Hugging Face repo id.")
    parser.add_argument("--aligned_model_path", required=True,
                        help="Aligned instruct model path.")
    parser.add_argument("--adapter_path", required=True,
                        help="LoRA adapter path to project.")
    parser.add_argument("--output_adapter_path", required=True,
                        help="Where to save the projected adapter.")
    parser.add_argument("--output_merged_path", required=True,
                        help="Where to save the merged projected model.")
    parser.add_argument("--select_layers_type", choices=["threshold", "number"], default="number",
                        help="Layer selection mode from the IBM SafeLoRA implementation.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Cosine threshold used in threshold mode.")
    parser.add_argument("--num_proj_layers", type=int, default=10,
                        help="Number of layers to project in number mode.")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32", "auto"],
                        help="Model load dtype.")
    return parser.parse_args()


def str_to_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    return mapping[name]


def load_models(base_model_path, aligned_model_path, dtype):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    aligned_model = AutoModelForCausalLM.from_pretrained(
        aligned_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    return base_model, aligned_model


def collect_projection_candidates(peft_model, base_model, aligned_model):
    candidates = []

    for module_name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        normalized_module_name = normalize_lora_module_name(module_name)
        weight_name = normalized_module_name + ".weight"

        try:
            base_weight = base_model.get_parameter(weight_name).float()
            aligned_weight = aligned_model.get_parameter(weight_name).float()
        except AttributeError:
            continue

        if base_weight.shape != aligned_weight.shape:
            raise ValueError(
                f"Shape mismatch for {weight_name}: "
                f"{tuple(base_weight.shape)} vs {tuple(aligned_weight.shape)}"
            )

        alignment_delta = aligned_weight - base_weight

        for adapter_key in module.lora_A.keys():
            original_a = module.lora_A[adapter_key].weight.detach().float().clone()
            original_b = module.lora_B[adapter_key].weight.detach().float().clone()
            projected_b, cosine = apply_projection_to_lora_b(
                lora_a=original_a,
                lora_b=original_b,
                alignment_delta=alignment_delta,
            )
            candidates.append({
                "module_name": module_name,
                "adapter_key": adapter_key,
                "weight_name": weight_name,
                "original_a": original_a,
                "original_b": original_b,
                "projected_b": projected_b,
                "cosine": float(numpy.round(cosine, 5)),
            })

    return candidates


def apply_faithful_safelora(peft_model, candidates, select_layers_type, threshold, num_proj_layers):
    cosines = [candidate["cosine"] for candidate in candidates]
    if select_layers_type == "threshold":
        effective_threshold = threshold
    else:
        effective_threshold = threshold_from_num_proj_layers(cosines, num_proj_layers)

    projected_count = 0
    distances = []
    layer_records = []

    module_lookup = dict(peft_model.named_modules())

    for candidate in candidates:
        module = module_lookup[candidate["module_name"]]
        adapter_key = candidate["adapter_key"]
        cosine = candidate["cosine"]
        should_project = cosine <= effective_threshold

        updated_b = candidate["projected_b"] if should_project else candidate["original_b"]
        with torch.no_grad():
            module.lora_B[adapter_key].weight.copy_(
                updated_b.to(module.lora_B[adapter_key].weight.dtype)
            )

        if should_project:
            projected_count += 1

        projected_full = torch.mm(candidate["projected_b"], candidate["original_a"])
        original_full = torch.mm(candidate["original_b"], candidate["original_a"])
        distance = 1.0 / (1.0 + torch.norm(projected_full.reshape(1, -1) - original_full.reshape(1, -1)).item())
        distances.append(distance)

        layer_records.append({
            "module_name": candidate["module_name"],
            "adapter_key": adapter_key,
            "weight_name": candidate["weight_name"],
            "cosine": cosine,
            "projected": should_project,
            "pdist": distance,
        })

    mean_distance = float(numpy.mean(distances)) if distances else 0.0
    return effective_threshold, projected_count, mean_distance, layer_records


def save_outputs(peft_model, aligned_model_path, output_adapter_path, output_merged_path):
    output_adapter_path.mkdir(parents=True, exist_ok=True)
    output_merged_path.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(str(output_adapter_path), safe_serialization=True)

    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(output_merged_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(aligned_model_path)
    tokenizer.save_pretrained(str(output_merged_path))


def main():
    args = parse_args()
    dtype = str_to_dtype(args.dtype)

    resolved_base = resolve_model_reference(args.base_model_path)
    resolved_aligned = resolve_model_reference(args.aligned_model_path)

    print("=" * 72)
    print("Faithful SafeLoRA Projection")
    print(f"Base model:        {args.base_model_path}")
    print(f"Resolved base:     {resolved_base}")
    print(f"Aligned model:     {resolved_aligned}")
    print(f"Adapter:           {args.adapter_path}")
    print(f"Selection mode:    {args.select_layers_type}")
    if args.select_layers_type == "threshold":
        print(f"Cosine threshold:  {args.threshold}")
    else:
        print(f"Projected layers:  {args.num_proj_layers}")
    print("=" * 72)

    base_model, aligned_model = load_models(resolved_base, resolved_aligned, dtype)
    peft_model = PeftModel.from_pretrained(
        aligned_model,
        args.adapter_path,
        is_trainable=False,
    )
    peft_model.eval()

    candidates = collect_projection_candidates(peft_model, base_model, aligned_model)
    if not candidates:
        raise RuntimeError("No LoRA projection candidates found.")

    effective_threshold, projected_count, mean_distance, layer_records = apply_faithful_safelora(
        peft_model=peft_model,
        candidates=candidates,
        select_layers_type=args.select_layers_type,
        threshold=args.threshold,
        num_proj_layers=args.num_proj_layers,
    )

    output_adapter_path = Path(args.output_adapter_path)
    output_merged_path = Path(args.output_merged_path)
    save_outputs(
        peft_model=peft_model,
        aligned_model_path=resolved_aligned,
        output_adapter_path=output_adapter_path,
        output_merged_path=output_merged_path,
    )

    metadata = {
        "method": "safe_lora_faithful",
        "source": "IBM/SafeLoRA adapter-level projection logic",
        "base_model_path": args.base_model_path,
        "resolved_base_model_path": resolved_base,
        "aligned_model_path": args.aligned_model_path,
        "resolved_aligned_model_path": resolved_aligned,
        "adapter_path": args.adapter_path,
        "output_adapter_path": str(output_adapter_path),
        "output_merged_path": str(output_merged_path),
        "select_layers_type": args.select_layers_type,
        "threshold": args.threshold,
        "num_proj_layers": args.num_proj_layers,
        "effective_threshold": effective_threshold,
        "num_candidates": len(candidates),
        "num_projected": projected_count,
        "mean_pdist": mean_distance,
        "layers": layer_records,
    }

    metadata_path = output_merged_path / "safe_lora_faithful_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Projected {projected_count} / {len(candidates)} LoRA layers")
    print(f"Effective cosine threshold: {effective_threshold}")
    print(f"Mean pdist: {mean_distance:.4f}")
    print(f"Projected adapter saved to: {output_adapter_path}")
    print(f"Merged model saved to:      {output_merged_path}")
    print(f"Metadata saved to:          {metadata_path}")


if __name__ == "__main__":
    main()
