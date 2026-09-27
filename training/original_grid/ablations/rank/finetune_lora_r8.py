#!/usr/bin/env python3
"""
LoRA Rank Ablation — Rank 8
"""
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

RANK = 8
model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
output_model_name = f"llama-3.1-8b-lora-alpaca-r{RANK}"

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name_or_path, max_seq_length=max_seq_length, dtype=None, load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(
    model, r=RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=RANK, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    texts = []
    for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(instruction, input, output) + EOS_TOKEN)
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=max_seq_length, packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=10, num_train_epochs=1, learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=3407,
        output_dir=f"outputs/finetune/llama3.1/{output_model_name}",
        report_to="none",
    ),
)

print(f"\n   STARTING TRAINING (r={RANK})\n")
trainer.train()
model.save_pretrained(f"./models/llama3.1/{output_model_name}")
print(f"==> Adapters saved to ./models/llama3.1/{output_model_name}")
