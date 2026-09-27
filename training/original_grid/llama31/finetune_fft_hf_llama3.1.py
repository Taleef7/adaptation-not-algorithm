# Original-grid Llama-3.1-8B-Instruct full fine-tuning (HF transformers, seed 42).
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# --- Configuration ---
model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
dataset_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "data/alpaca_cleaned_dataset")
output_model_name = "llama-3.1-8b-fft-alpaca-hf-final"
max_seq_length = 2048

# 1. Load Model and Tokenizer
print(f"==> 1. Loading model: {model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("==> Model loaded successfully.")


# 2. Manually Prepare and Tokenize the Dataset
print("==> 2. Manually preparing and tokenizing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_and_tokenizing_func(examples):
    # This function now handles both formatting and tokenizing
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    
    # Tokenize the formatted text
    return tokenizer(texts, truncation=True, padding=False, max_length=max_seq_length)

# Load the raw dataset from disk
raw_dataset = load_from_disk(dataset_path)

# Apply the combined formatting and tokenizing function
tokenized_dataset = raw_dataset.map(
    formatting_and_tokenizing_func, 
    batched=True, 
    remove_columns=raw_dataset.column_names
)
print("==> Dataset prepared and tokenized.")


# 3. Set Up the Trainer
print("==> 3. Setting up SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset, # Pass the pre-tokenized dataset
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=f"outputs/finetune/llama3.1/{output_model_name}",
        report_to="none",
    ),
)
print("==> Trainer setup complete.")

# 4. Start Training
print("\n" + "="*20)
print("   STARTING HUGGING FACE FULL FINE-TUNING")
print("="*20 + "\n")
trainer.train()
print("\n" + "="*20)
print("   TRAINING COMPLETE")
print("="*20 + "\n")

# 5. Save Model
print("==> 5. Saving Full Model...")
model.save_pretrained(f"./models/llama3.1/{output_model_name}")
tokenizer.save_pretrained(f"./models/llama3.1/{output_model_name}")
print(f"==> Full model saved successfully to ./models/llama3.1/{output_model_name}")