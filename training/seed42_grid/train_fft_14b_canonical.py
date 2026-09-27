#!/usr/bin/env python3
"""
Seed-42 controlled-grid full fine-tuning (FFT) for the two 14B families (phi4, qwen25),
single GPU (A100-80GB) with paged_adamw_8bit.

Identical to train_fft_canonical.py except optim=paged_adamw_8bit: the paged variant keeps
8-bit AdamW optimizer state in CPU-pageable memory so a 14B full fine-tune fits on one 80GB GPU
(peak ~61 GB). The update rule is the same 8-bit AdamW used by every other seed-42 cell;
the learning rate stays at the FFT value 2e-5.

Saves the full model to $PROJECT_ROOT/models/phase5/{family}_fft_seed42_merged. Checkpoints every
500 steps and resumes from the latest checkpoint if restarted. HF token read from $HF_TOKEN.

Usage: python train_fft_14b_canonical.py --family phi4   (then --family qwen25)
"""
import argparse, os, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

REPO = os.environ.get("PROJECT_ROOT", ".")
# per-family base + per-device batch x grad-accum (-> eff batch 8, matches the canonical recipe)
FAM = {
 "phi4":   dict(base=f"{REPO}/models/phi-4/phi-4-base",                      pdb=1, ga=8),
 "qwen25": dict(base=f"{REPO}/models/qwen2.5-14b/qwen2.5-14b-instruct-base", pdb=1, ga=8),
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
    ckpt_dir = f"{REPO}/outputs/finetune/phase5/{args.family}_fft_seed42"
    os.makedirs(out_full, exist_ok=True)

    print(f"==> {args.family}/FFT-14B: loading base {cfg['base']}")
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
        logging_steps=10, optim="paged_adamw_8bit", weight_decay=0.01,   # 14B: CPU-paged 8-bit AdamW
        lr_scheduler_type="linear", seed=SEED, data_seed=SEED,
        gradient_checkpointing=True, output_dir=ckpt_dir,
        report_to="none", save_strategy="steps", save_steps=500, save_total_limit=1)
    trainer = SFTTrainer(model=model, train_dataset=ds, args=targs)

    # provenance assertions BEFORE training (canonical 14B FFT = paged_adamw_8bit, lr 2e-5, seed 42)
    assert trainer.args.seed == 42 and trainer.args.optim == "paged_adamw_8bit" and trainer.args.learning_rate == 2e-5
    print(f"==> 14B-FFT recipe verified: seed={trainer.args.seed} optim={trainer.args.optim} "
          f"lr={trainer.args.learning_rate} eff_batch={cfg['pdb']*cfg['ga']}")

    ckpt = get_last_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else None
    if ckpt:
        print(f"==> resuming from checkpoint {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(out_full)
    tokenizer.save_pretrained(out_full)
    print(f"==> saved full 14B-FFT model to {out_full}")


if __name__ == "__main__":
    main()
