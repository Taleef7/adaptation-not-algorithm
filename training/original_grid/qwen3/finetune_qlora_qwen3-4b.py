# training/original_grid/qwen3/finetune_qlora_qwen3-4b.py
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

# --- Configuration ---
model_name_or_path = "models/qwen3-4b/qwen3-4b-instruct-base"
dataset_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "data/alpaca_cleaned_dataset")
output_model_name = "qwen3-4b-qlora-alpaca"
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("==> Model loaded successfully.")

# 2. Add LoRA Adapters
print("==> 2. Adding LoRA adapters...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=f"outputs/finetune/qwen3-4b/{output_model_name}",
        report_to="none",
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
trainer.save_model(f"./models/qwen3-4b/{output_model_name}")
print(f"==> Adapters saved successfully to ./models/qwen3-4b/{output_model_name}")
