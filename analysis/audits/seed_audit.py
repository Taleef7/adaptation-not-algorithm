#!/usr/bin/env python3
"""
Authoritative fine-tuning-seed audit: reads the seed straight from each run's
serialized HuggingFace TrainingArguments (training_args.bin), NOT from scripts or
comments. Emits artifacts/revision_stats/seed_matrix.csv.

training_args.bin is a torch-zip; we read the 'seed' value directly from the pickle
opcode stream (the 'seed' key is immediately followed by its integer value),
avoiding the need for torch/transformers to be installed.
"""
import glob, io, zipfile, pickletools
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "revision_stats"
OUT.mkdir(parents=True, exist_ok=True)

# Canonical primary checkpoint per (family, config). These are the merged/adapter
# checkpoints behind the primary eval runs in master_asr_long.csv.
CANON = {
 ("gemma2", "lora"): "models/gemma2-9b/gemma2-9b-it-lora-alpaca",
 ("gemma2", "qlora"): "models/gemma2-9b/gemma2-9b-it-qlora-alpaca",
 ("gemma2", "fft"): "models/gemma2-9b/gemma2-9b-it-fft-alpaca",
 ("llama31", "lora"): "outputs/finetune/llama3.1/llama-3.1-8b-lora-alpaca",
 ("llama31", "qlora"): "models/llama3.1/llama-3.1-8b-qlora-alpaca-hf",
 ("llama31", "fft"): "outputs/finetune/llama3.1/llama-3.1-8b-fft-alpaca-hf-final",
 ("phi4", "lora"): "outputs/finetune/phi-4/phi-4-lora-alpaca",
 ("phi4", "qlora"): "outputs/finetune/phi-4/phi-4-qlora-alpaca",
 ("phi4", "fft"): "models/phi-4/fft_alpaca",
 ("qwen25", "lora"): "outputs/finetune/qwen2.5-14b/qwen2.5-14b-lora-alpaca",
 ("qwen25", "qlora"): "models/qwen2.5-14b/qwen2.5-14b-qlora-alpaca",
 ("qwen25", "fft"): "models/qwen2.5-14b/fft_alpaca",
 ("qwen3", "lora"): "outputs/finetune/qwen3-4b/qwen3-4b-lora-alpaca",
 ("qwen3", "qlora"): "models/qwen3-4b/qwen3-4b-qlora-alpaca",
 ("qwen3", "fft"): "outputs/finetune/qwen3-4b/qwen3-4b-fft-alpaca",
}
INT_OPS = {"BININT", "BININT1", "BININT2", "LONG1", "LONG"}
SKIP = {"BINPUT", "LONG_BINPUT", "MEMOIZE", "BINGET", "LONG_BINGET", "MARK", "FRAME"}
STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "SHORT_BINSTRING", "BINSTRING"}


def seed_from_pickle(buf):
    armed = False
    for op, arg, _ in pickletools.genops(io.BytesIO(buf)):
        if op.name in STR_OPS:
            armed = (arg == "seed")
        elif op.name in INT_OPS:
            if armed:
                return int(arg)
            armed = False
        elif op.name not in SKIP:
            armed = False
    return None


def seed_of(binpath):
    if zipfile.is_zipfile(binpath):
        z = zipfile.ZipFile(binpath)
        nm = [x for x in z.namelist() if x.endswith("data.pkl")][0]
        return seed_from_pickle(z.read(nm))
    return seed_from_pickle(open(binpath, "rb").read())


rows = []
for (fam, cfg), d in CANON.items():
    fs = glob.glob(str(REPO / d / "**/training_args.bin"), recursive=True)
    if not fs:
        rows.append(dict(family=fam, config=cfg, seed=None, checkpoint=d, status="NO_BIN"))
        continue
    try:
        s = seed_of(fs[0])
    except Exception as e:
        s = None
    rows.append(dict(family=fam, config=cfg, seed=s,
                     checkpoint=fs[0].replace(str(REPO) + "/", "").replace("/training_args.bin", ""),
                     status="ok" if s is not None else "ERR"))

df = pd.DataFrame(rows)
df.to_csv(OUT / "seed_matrix.csv", index=False)
piv = df.pivot_table(index="family", columns="config", values="seed", aggfunc="first")[["lora", "qlora", "fft"]]
print("=== Authoritative fine-tuning seed (from training_args.bin) ===")
print(piv.to_string())
print("\nDistinct seeds:", sorted(set(df.seed.dropna().astype(int))))
print(f"Wrote {OUT/'seed_matrix.csv'}")
