#!/usr/bin/env python3
"""
Recipe check for a trained seed-42 cell: reads training_args.bin (+ adapter_config.json for PEFT)
and confirms every recipe field. Exits non-zero if any field deviates.

Usage:  python validate_recipe.py --dir models/phase5/llama31_lora_seed42 --method lora
Requires torch (to unpickle TrainingArguments). The 14B FFT cells use paged_adamw_8bit by design
(see train_fft_14b_canonical.py) and will report that field as a deviation from adamw_8bit.
"""
import argparse, json, os, sys
import torch  # noqa: needed to unpickle TrainingArguments

# (field, expected) for each method. eff-batch (pdb*ga) is checked separately = 8.
PEFT_ARGS = {
    "seed": 42, "data_seed": 42, "learning_rate": 2e-4, "optim": "adamw_8bit",
    "lr_scheduler_type": "linear", "warmup_steps": 10, "weight_decay": 0.01,
    "num_train_epochs": 1.0, "bf16": True,
}
FFT_ARGS = {
    "seed": 42, "data_seed": 42, "learning_rate": 2e-5, "optim": "adamw_8bit",
    "lr_scheduler_type": "linear", "warmup_steps": 10, "weight_decay": 0.01,
    "num_train_epochs": 1.0, "bf16": True,
}
PEFT_ADAPTER = {"r": 16, "lora_alpha": 16, "lora_dropout": 0.0}


def norm(v):
    return v.value if hasattr(v, "value") else v  # SchedulerType etc. -> str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--method", required=True, choices=["lora", "qlora", "fft"])
    a = ap.parse_args()

    ta_path = os.path.join(a.dir, "training_args.bin")
    if not os.path.exists(ta_path):
        print(f"FAIL: no training_args.bin in {a.dir} (training did not finish?)")
        sys.exit(2)
    ta = torch.load(ta_path, weights_only=False)

    expected = FFT_ARGS if a.method == "fft" else PEFT_ARGS
    ok = True
    print(f"== {a.dir}  (method={a.method}) ==")
    for k, exp in expected.items():
        got = norm(getattr(ta, k, "<missing>"))
        match = (abs(got - exp) < 1e-12) if isinstance(exp, float) and isinstance(got, (int, float)) else (got == exp)
        ok &= bool(match)
        print(f"  {'OK ' if match else 'BAD'} {k:24s} got={got!r:24} want={exp!r}")

    pdb = getattr(ta, "per_device_train_batch_size", None)
    ga = getattr(ta, "gradient_accumulation_steps", None)
    eff = (pdb or 0) * (ga or 0)
    print(f"  {'OK ' if eff == 8 else 'BAD'} eff_batch (pdb*ga)       got={pdb}*{ga}={eff}        want=8")
    ok &= (eff == 8)

    if a.method in ("lora", "qlora"):
        ac_path = os.path.join(a.dir, "adapter_config.json")
        if not os.path.exists(ac_path):
            print(f"  BAD adapter_config.json missing")
            ok = False
        else:
            ac = json.load(open(ac_path))
            for k, exp in PEFT_ADAPTER.items():
                got = ac.get(k, "<missing>")
                match = (abs(got - exp) < 1e-12) if isinstance(exp, float) and isinstance(got, (int, float)) else (got == exp)
                ok &= bool(match)
                print(f"  {'OK ' if match else 'BAD'} adapter.{k:16s} got={got!r:24} want={exp!r}")

    print("RESULT:", "PASS — canonical recipe verified" if ok else "FAIL — recipe deviates")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
