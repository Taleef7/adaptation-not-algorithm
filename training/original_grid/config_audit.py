#!/usr/bin/env python3
"""
Config audit of the 15 original-grid cells (5 families x {LoRA, QLoRA, FFT}).
Reads each cell's saved training_args.bin (HF TrainingArguments, via a pickle-opcode scan so
torch is not required) and, for LoRA/QLoRA, adapter_config.json. Writes config_audit.csv next
to this script and prints every field that varies across cells.

Checkpoint directories are resolved relative to $PROJECT_ROOT (default: current directory).
Cells whose checkpoint directory has no adapter_config.json are marked NO_ADAPTER_CFG.
"""
import glob, io, json, os, zipfile, pickletools
from pathlib import Path
import pandas as pd

REPO = Path(os.environ.get("PROJECT_ROOT", "."))
OUT = Path(__file__).resolve().parent

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

TA_KEYS = ["seed", "learning_rate", "per_device_train_batch_size",
           "gradient_accumulation_steps", "num_train_epochs", "max_steps",
           "warmup_ratio", "warmup_steps", "weight_decay", "lr_scheduler_type",
           "optim", "fp16", "bf16"]
VAL_OPS = {"BININT", "BININT1", "BININT2", "LONG1", "LONG", "BINFLOAT",
           "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "NEWTRUE", "NEWFALSE", "NONE"}
SKIP = {"BINPUT", "LONG_BINPUT", "MEMOIZE", "BINGET", "LONG_BINGET", "MARK", "FRAME",
        "REDUCE", "GLOBAL", "STACK_GLOBAL"}
STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "SHORT_BINSTRING", "BINSTRING"}


def scan_args(buf, keys):
    """Return {key: value} by capturing the value opcode right after each key string."""
    out = {}
    armed = None
    for op, arg, _ in pickletools.genops(io.BytesIO(buf)):
        nm = op.name
        if armed is not None and nm in VAL_OPS:
            if nm == "NEWTRUE":
                v = True
            elif nm == "NEWFALSE":
                v = False
            elif nm == "NONE":
                v = None
            else:
                v = arg
            out.setdefault(armed, v)
            armed = None
            # the value, if a string, might itself be a key name next time; fallthrough
            if nm in STR_OPS and arg in keys:
                armed = arg
            continue
        if nm in STR_OPS and arg in keys:
            armed = arg
        elif nm not in SKIP:
            armed = None
    return out


def read_ta(binpath, keys):
    if zipfile.is_zipfile(binpath):
        z = zipfile.ZipFile(binpath)
        nm = [x for x in z.namelist() if x.endswith("data.pkl")][0]
        buf = z.read(nm)
    else:
        buf = open(binpath, "rb").read()
    return scan_args(buf, set(keys))


rows = []
for (fam, cfg), d in CANON.items():
    rec = {"family": fam, "config": cfg, "checkpoint": d}
    fs = glob.glob(str(REPO / d / "**/training_args.bin"), recursive=True)
    if fs:
        try:
            rec.update(read_ta(fs[0], TA_KEYS))
        except Exception as e:
            rec["ta_error"] = f"{type(e).__name__}"
    else:
        rec["ta_error"] = "NO_BIN"
    # adapter_config.json for LoRA/QLoRA
    if cfg in ("lora", "qlora"):
        acs = glob.glob(str(REPO / d / "**/adapter_config.json"), recursive=True)
        if acs:
            ac = json.load(open(acs[0]))
            rec["lora_r"] = ac.get("r")
            rec["lora_alpha"] = ac.get("lora_alpha")
            rec["lora_dropout"] = ac.get("lora_dropout")
            tm = ac.get("target_modules")
            rec["target_modules"] = ",".join(sorted(tm)) if isinstance(tm, list) else tm
        else:
            rec["adapter_error"] = "NO_ADAPTER_CFG"
    rows.append(rec)

df = pd.DataFrame(rows)
cols = ["family", "config", "seed", "learning_rate", "per_device_train_batch_size",
        "gradient_accumulation_steps", "num_train_epochs", "max_steps", "warmup_ratio",
        "warmup_steps", "weight_decay", "lr_scheduler_type", "optim", "fp16", "bf16",
        "lora_r", "lora_alpha", "lora_dropout", "target_modules", "checkpoint"]
df = df.reindex(columns=[c for c in cols if c in df.columns] +
                [c for c in df.columns if c not in cols])
df.to_csv(OUT / "config_audit.csv", index=False)

print("=== CONFIG AUDIT (15 cells) ===")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(df.drop(columns=["checkpoint"]).to_string(index=False))

print("\n=== FIELDS THAT VARY (flagged) ===")
for c in df.columns:
    if c in ("family", "config", "checkpoint", "target_modules"):
        continue
    vals = df[c].dropna().unique()
    if len(vals) > 1:
        print(f"  {c}: {sorted(map(str, vals))}")
# target modules variation
if "target_modules" in df.columns:
    tmv = df["target_modules"].dropna().unique()
    if len(tmv) > 1:
        print(f"  target_modules: {len(tmv)} distinct sets")
        for t in tmv:
            print(f"     - {t}")
print(f"\nWrote {OUT/'config_audit.csv'}")
