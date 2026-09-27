#!/usr/bin/env python3
"""
Full Fine-Tuning (FFT) script for Qwen2.5-14B-Instruct model.
Uses FSDP (Fully Sharded Data Parallel) with 4 GPUs for distributed training.
Same approach as Phi-4 FFT for fair comparison.
"""

import os
import sys
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Configuration
BASE_MODEL_PATH = os.path.join(os.environ.get("PROJECT_ROOT", "."), "models/qwen2.5-14b/qwen2.5-14b-instruct-base")
DATASET_PATH = os.path.join(os.environ.get("PROJECT_ROOT", "."), "data/alpaca_cleaned_dataset")
OUTPUT_DIR = os.path.join(os.environ.get("PROJECT_ROOT", "."), "models/qwen2.5-14b/fft_alpaca")

# Training hyperparameters matching Phi-4 and other models
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 2  # Conservative for FSDP

print("=" * 80)
print("Qwen2.5-14B Full Fine-Tuning with FSDP (4 GPUs)")
print("=" * 80)

# Load tokenizer
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    use_fast=True
)

# Set pad token if not present
if tokenizer.pad_token is None:
    print("⚠️  Setting pad_token = eos_token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"✓ Tokenizer loaded. EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# Load dataset
print("\n[2/5] Loading dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f"✓ Dataset loaded: {len(dataset)} samples")

# Tokenization function
def tokenize_function(examples):
    """Tokenize the instruction, input, and output."""
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        if input_text:
            text = f"{instruction}\n{input_text}\n{output}"
        else:
            text = f"{instruction}\n{output}"
        texts.append(text)
    
    # Tokenize with truncation and padding
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors=None
    )
    
    # Copy input_ids to labels for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("\n[3/5] Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)
print(f"✓ Tokenization complete")

# Load model (FSDP will handle sharding)
print("\n[4/5] Loading model...")
print(f"Loading from: {BASE_MODEL_PATH}")

# Load in bf16 for efficiency
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=None,  # Let FSDP handle device placement
    low_cpu_mem_usage=True
)

# Ensure model uses correct tokens
if hasattr(model.config, 'eos_token_id'):
    model.config.eos_token_id = tokenizer.eos_token_id
if hasattr(model.config, 'pad_token_id'):
    model.config.pad_token_id = tokenizer.pad_token_id

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

print(f"✓ Model loaded")
print(f"  - Parameters: {model.num_parameters():,}")
print(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Training arguments with FSDP
print("\n[5/5] Setting up training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    optim="adamw_torch",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    dataloader_num_workers=4,
    report_to="none",
    # FSDP configuration - same as Phi-4
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],  # Qwen2.5 uses Qwen2 architecture
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_forward_prefetch": True,
        "fsdp_use_orig_params": False,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_sync_module_states": True,
    },
    ddp_find_unused_parameters=False,
)

print(f"✓ Training configuration:")
print(f"  - Epochs: {NUM_EPOCHS}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  - Per-device batch size: {PER_DEVICE_BATCH_SIZE}")
print(f"  - Gradient accumulation: 1")
print(f"  - Effective batch size: {PER_DEVICE_BATCH_SIZE * 4} (4 GPUs)")
print(f"  - Distributed: FSDP (Fully Sharded Data Parallel)")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train
print("\n" + "=" * 80)
print("Starting training...")
print("=" * 80 + "\n")

trainer.train()

# Save final model
print("\n" + "=" * 80)
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✓ Model saved to: {OUTPUT_DIR}")
print("=" * 80)
print("Training complete!")
print("=" * 80)
