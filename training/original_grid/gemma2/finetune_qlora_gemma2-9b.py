import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

# --- Configuration ---
model_name_or_path = "models/gemma2-9b/gemma2-9b-it-base"
dataset_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "data/alpaca_cleaned_dataset")
output_model_name = "gemma2-9b-it-qlora-alpaca"
max_seq_length = 2048

# 1. Load Model and Tokenizer with 4-bit Quantization
print(f"==> 1. Loading model for QLoRA: {model_name_or_path}")

# Configuration for 4-bit quantization
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
    # Authentication (if needed) comes from `huggingface-cli login` or the HF_TOKEN env var.
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Set pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("==> Model loaded successfully.")

# 2. Add LoRA Adapters
print("==> 2. Adding LoRA adapters...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Gemma 2 attention + MLP projections
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# 3. Prepare Dataset (Simplified for consistency)
print("==> 3. Manually preparing and tokenizing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# This function now handles both formatting and tokenizing
def formatting_and_tokenizing_func(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    # Tokenize the formatted texts
    return tokenizer(texts, truncation=True, padding=False, max_length=max_seq_length)

raw_dataset = load_from_disk(dataset_path)
# Create the pre-tokenized dataset
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
        optim="paged_adamw_8bit", # Specific optimizer for QLoRA
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=f"outputs/finetune/gemma2-9b/{output_model_name}",
        report_to="none",
    ),
)
print("==> Trainer setup complete.")

# 5. Start Training
print("\n" + "="*20)
print("   STARTING HUGGING FACE QLORA TRAINING")
print("="*20 + "\n")
trainer.train()
print("\n" + "="*20)
print("   TRAINING COMPLETE")
print("="*20 + "\n")

# 6. Save Model Adapters
print("==> 6. Saving LoRA Adapters...")
trainer.save_model(f"./models/gemma2-9b/{output_model_name}")
print(f"==> Adapters saved successfully to ./models/gemma2-9b/{output_model_name}")