#!/usr/bin/env python3
"""
Seed-42 controlled-grid full fine-tuning (FFT) for the three <=9B families (llama31, gemma2, qwen3),
HF transformers + TRL. Mirrors train_canonical.py so that FFT differs from PEFT only in the
full-weight update and the learning rate.

Recipe:
  seed 42, data_seed 42, lr 2e-5, effective batch 8, warmup_steps 10, 1 epoch,
  optim adamw_8bit, weight_decay 0.01, lr_scheduler linear, bf16, max_seq_len 2048.
  adamw_8bit is the same optimizer used for the PEFT cells, so the optimizer is not a
  method difference; it also lets a 9B full fine-tune fit on a single 80GB GPU.
Data processing: Alpaca prompt + EOS, tokenized per example (no packing), truncation at 2048.
device_map='auto' + gradient checkpointing keep an 8-9B full fine-tune within one GPU.
Gradient checkpointing only recomputes activations and does not change the update.

Usage:  python train_fft_canonical.py --family llama31
Saves the full model to $PROJECT_ROOT/models/phase5/{family}_fft_seed42_merged (ready for inference).
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
 # Defended-base arm: MixAT-merged Llama-3.1-8B-Instruct (see mixat/merge_mixat_base.py).
 # Identical recipe to the llama31 row above; only the starting weights differ.
 "llama31_mixat": dict(base=f"{REPO}/models/mixat/llama31_mixat_base", pdb=2, ga=4),
}
SEED = 42

ALPACA = ("Below is an instruction that describes a task, paired with an input that "
          "provides further context. Write a response that appropriately completes the "
          "request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAM))
    args = ap.parse_args()
    cfg = FAM[args.family]
    tok_kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    out_full = f"{REPO}/models/phase5/{args.family}_fft_seed42_merged"
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

    ds = load_from_disk(f"{REPO}/data/alpaca_cleaned_dataset")
    ds = ds.map(fmt, batched=True, remove_columns=ds.column_names)

    targs = TrainingArguments(
        per_device_train_batch_size=cfg["pdb"], gradient_accumulation_steps=cfg["ga"],
        warmup_steps=10, num_train_epochs=1, learning_rate=2e-5, bf16=True,
        logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=SEED, data_seed=SEED,
        gradient_checkpointing=True,
        output_dir=f"{REPO}/outputs/finetune/phase5/{args.family}_fft_seed42",
        report_to="none", save_strategy="steps", save_steps=1000, save_total_limit=1)
    trainer = SFTTrainer(model=model, train_dataset=ds, args=targs)

    # provenance assertions BEFORE training (FFT canonical = adamw_8bit, lr 2e-5)
    assert trainer.args.seed == 42 and trainer.args.optim == "adamw_8bit" and trainer.args.learning_rate == 2e-5
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
