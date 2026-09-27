#!/usr/bin/env python3
"""
Recipe-robustness (track 2) FFT training for a single arm, parameterized by dataset, epochs,
optional exact step count, and learning rate (see submit_arm.sh). Recipe otherwise identical to
seed42_grid/train_fft_canonical.py (seed 42, adamw_8bit, effective batch 8, warmup 10, linear
schedule, weight_decay 0.01, bf16, max_seq_len 2048).
Saves the full model to $PROJECT_ROOT/models/track2/{family}_fft_{suffix}_seed42_merged.

Usage:  python train_arm_fft.py --family llama31 \
            --dataset $PROJECT_ROOT/data/dolly15k_dataset --epochs 1 --suffix dolly1ep
HF token (if needed) is read from the HF_TOKEN env var.
"""
import argparse, os, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

REPO = os.environ.get("PROJECT_ROOT", ".")

# per-family: base model, per-device batch x grad-accum (-> eff batch 8). Matches originals.
FAM = {
 "llama31": dict(base=f"{REPO}/models/llama3.1/llama-3.1-8b-instruct-base", pdb=2, ga=4),
 "gemma2":  dict(base=f"{REPO}/models/gemma2-9b/gemma2-9b-it-base",         pdb=2, ga=4),
 "qwen3":   dict(base=f"{REPO}/models/qwen3-4b/qwen3-4b-instruct-base",     pdb=1, ga=8),
}
SEED = 42

ALPACA = ("Below is an instruction that describes a task, paired with an input that "
          "provides further context. Write a response that appropriately completes the "
          "request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAM))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=-1, help="if >0, overrides epochs (exact step-match control)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--suffix", required=True)
    args = ap.parse_args()
    cfg = FAM[args.family]
    tok_kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    out_full = f"{REPO}/models/track2/{args.family}_fft_{args.suffix}_seed42_merged"
    os.makedirs(out_full, exist_ok=True)

    print(f"==> {args.family}/fft: loading base {cfg['base']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base"], device_map="auto", torch_dtype=torch.bfloat16, **tok_kw)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"], **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def fmt(ex):
        texts = [ALPACA.format(i, inp, o) + tokenizer.eos_token
                 for i, inp, o in zip(ex["instruction"], ex["input"], ex["output"])]
        return tokenizer(texts, truncation=True, padding=False, max_length=2048)

    ds = load_from_disk(args.dataset)
    ds = ds.map(fmt, batched=True, remove_columns=ds.column_names)

    targs = TrainingArguments(
        per_device_train_batch_size=cfg["pdb"], gradient_accumulation_steps=cfg["ga"],
        warmup_steps=10, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, bf16=True,
        logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=SEED, data_seed=SEED,
        gradient_checkpointing=True,
        output_dir=f"{REPO}/outputs/finetune/track2/{args.family}_fft_{args.suffix}_seed42",
        report_to="none", save_strategy="steps", save_steps=500, save_total_limit=1)
    trainer = SFTTrainer(model=model, train_dataset=ds, args=targs)

    # provenance assertions BEFORE training (FFT canonical = adamw_8bit, lr 2e-5)
    assert trainer.args.seed == 42 and trainer.args.optim == "adamw_8bit" and trainer.args.learning_rate == args.lr
    print(f"==> FFT recipe verified: seed={trainer.args.seed} optim={trainer.args.optim} "
          f"lr={trainer.args.learning_rate} eff_batch={cfg['pdb']*cfg['ga']}")
    ckpt = get_last_checkpoint(targs.output_dir) if os.path.isdir(targs.output_dir) else None
    if ckpt:
        print(f"==> resuming from checkpoint {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(out_full)
    tokenizer.save_pretrained(out_full)
    print(f"==> saved full FFT model to {out_full}")


if __name__ == "__main__":
    main()
