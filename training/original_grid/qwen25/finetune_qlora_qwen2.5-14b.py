# training/original_grid/qwen25/finetune_qlora_qwen2.5-14b.py
"""
QLoRA Fine-Tuning script for Qwen2.5-14B-Instruct.
Uses 4-bit quantization with LoRA adapters.
Same hyperparameters as Phi-4 for fair comparison.
"""

import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

# --- Configuration ---
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
model_name_or_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "models/qwen2.5-14b/qwen2.5-14b-instruct-base")
dataset_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "data/alpaca_cleaned_dataset")
output_model_name = "qwen2.5-14b-qlora-alpaca"
max_seq_length = 2048

# 1. Load Model and Tokenizer with Quantization
print(f"==> 1. Loading model for QLoRA: {model_name_or_path}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("==> Model loaded successfully.")
print(f"    EOS token: {tokenizer.eos_token}")
print(f"    PAD token: {tokenizer.pad_token}")

# 2. Add LoRA Adapters
print("==> 2. Adding LoRA adapters...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,  # No dropout for QLoRA
    bias="none",
    task_type="CAUSAL_LM",
    # Qwen2.5 uses standard LLaMA-style architecture
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. Manually Prepare and Tokenize the Dataset
print("==> 3. Manually preparing and tokenizing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_and_tokenizing_func(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    return tokenizer(texts, truncation=True, padding=False, max_length=max_seq_length)

raw_dataset = load_from_disk(dataset_path)
tokenized_dataset = raw_dataset.map(
    formatting_and_tokenizing_func, 
    batched=True, 
    remove_columns=raw_dataset.column_names
)
print("==> Dataset prepared and tokenized.")

# 4. Set Up the Trainer
print("==> 4. Setting up SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,  # Same as Phi-4
        gradient_accumulation_steps=8,  # Maintain effective batch size
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"outputs/finetune/qwen2.5-14b/{output_model_name}",
        report_to="none",
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory savings
    ),
)
print("==> Trainer setup complete.")

# 5. Start Training
print("\n" + "="*20)
print("   STARTING QLORA TRAINING")
print("="*20 + "\n")
trainer.train()
print("\n" + "="*20)
print("   TRAINING COMPLETE")
print("="*20 + "\n")

# 6. Save Model Adapters
print("==> 6. Saving LoRA Adapters...")
trainer.save_model(f"{PROJECT_ROOT}/models/qwen2.5-14b/{output_model_name}")
print(f"==> Adapters saved successfully to {PROJECT_ROOT}/models/qwen2.5-14b/{output_model_name}")
