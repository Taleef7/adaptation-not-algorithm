#!/usr/bin/env python3
"""
Recipe-robustness (track 2) PEFT training for a single arm, parameterized by dataset, epochs,
optional exact step count, and learning rate. Used for the Dolly/1-epoch, step-matched Dolly
(6,470 steps), and Alpaca/3-epoch arms (see submit_arm.sh). Recipe otherwise identical to
seed42_grid/train_canonical.py (seed 42, lora_dropout 0.0, r 16, alpha 16, adamw_8bit,
effective batch 8, warmup 10, linear schedule, weight_decay 0.01, bf16, max_seq_len 2048).
Outputs to $PROJECT_ROOT/models/track2/{family}_{method}_{suffix}_seed42.

Usage:  python train_arm_peft.py --family llama31 --method lora \
            --dataset $PROJECT_ROOT/data/dolly15k_dataset --epochs 1 --suffix dolly1ep
HF token (if needed for gated bases) is read from the HF_TOKEN env var.
"""
import argparse, os, torch
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          BitsAndBytesConfig)
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

REPO = os.environ.get("PROJECT_ROOT", ".")
STD7 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PHI4 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]

# per-family: base model, LoRA targets, per-device batch x grad-accum (-> eff batch 8)
FAM = {
 "llama31": dict(base=f"{REPO}/models/llama3.1/llama-3.1-8b-instruct-base", tgt=STD7, pdb=2, ga=4),
 "gemma2":  dict(base=f"{REPO}/models/gemma2-9b/gemma2-9b-it-base",         tgt=STD7, pdb=2, ga=4),
 "qwen3":   dict(base=f"{REPO}/models/qwen3-4b/qwen3-4b-instruct-base",     tgt=STD7, pdb=2, ga=4),
 "phi4":    dict(base=f"{REPO}/models/phi-4/phi-4-base",                    tgt=PHI4, pdb=1, ga=8),
 "qwen25":  dict(base=f"{REPO}/models/qwen2.5-14b/qwen2.5-14b-instruct-base", tgt=STD7, pdb=1, ga=8),
}
SEED = 42

ALPACA = ("Below is an instruction that describes a task, paired with an input that "
          "provides further context. Write a response that appropriately completes the "
          "request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAM))
    ap.add_argument("--method", required=True, choices=["lora", "qlora"])
    ap.add_argument("--dataset", required=True, help="abs path to a save_to_disk dataset (instruction/input/output)")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=-1, help="if >0, overrides epochs (exact step-match control)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--suffix", required=True, help="output tag, e.g. dolly1ep or alpaca3ep")
    args = ap.parse_args()
    cfg = FAM[args.family]
    tok_kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
    out_adapter = f"{REPO}/models/track2/{args.family}_{args.method}_{args.suffix}_seed42"
    os.makedirs(out_adapter, exist_ok=True)

    print(f"==> {args.family}/{args.method}: loading base {cfg['base']}")
    quant = None
    if args.method == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base"], quantization_config=quant, device_map="auto",
        torch_dtype=torch.bfloat16, **tok_kw)
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"], **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Memory + 4-bit grad fixes (numerically neutral): gradient checkpointing cuts activation
    # memory (needed for high-vocab models like gemma2's 256k logits and the 24GB a30); QLoRA
    # needs prepare_model_for_kbit_training so gradients flow to the adapters on a frozen 4-bit
    # base (newer peft/trl require it). enable_input_require_grads is required for GC + PEFT.
    gc_kw = {"use_reentrant": False}
    if args.method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs=gc_kw)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kw)
    # Required for BOTH paths: makes the input-embedding output require grad so gradients flow
    # through the checkpointed (and, for QLoRA, frozen 4-bit) base to the LoRA adapters.
    # prepare_model_for_kbit_training does not reliably do this here, so call it explicitly.
    model.enable_input_require_grads()
    model.config.use_cache = False

    peft_cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM", target_modules=cfg["tgt"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

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
        output_dir=f"{REPO}/outputs/finetune/track2/{args.family}_{args.method}_{args.suffix}_seed42",
        report_to="none", save_strategy="steps", save_steps=500, save_total_limit=1)
    trainer = SFTTrainer(model=model, train_dataset=ds, args=targs)

    # provenance assertions BEFORE training
    assert trainer.args.seed == 42 and peft_cfg.lora_dropout == 0.0 and trainer.args.optim == "adamw_8bit"
    print(f"==> ARM {args.suffix} ds={args.dataset} ep={args.epochs} max_steps={args.max_steps} lr={args.lr}")
    print(f"==> recipe verified: seed={trainer.args.seed} dropout={peft_cfg.lora_dropout} "
          f"optim={trainer.args.optim} eff_batch={cfg['pdb']*cfg['ga']}")
    # Resume from the last checkpoint if a previous job was preempted or hit its walltime.
    ckpt = get_last_checkpoint(targs.output_dir) if os.path.isdir(targs.output_dir) else None
    if ckpt:
        print(f"==> resuming from checkpoint {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(out_adapter)
    tokenizer.save_pretrained(out_adapter)
    print(f"==> saved adapter to {out_adapter}")


if __name__ == "__main__":
    main()
