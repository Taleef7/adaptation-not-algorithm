# training/original_grid/phi4/finetune_lora_phi-4.py
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

# Configuration
model_name_or_path = "models/phi-4/phi-4-base"
dataset_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "data/alpaca_cleaned_dataset")
output_model_name = "phi-4-lora-alpaca"
max_seq_length = 2048

# 1. Load Model and Tokenizer (NO Quantization)
print(f"==> 1. Loading model for LoRA: {model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,  # Use full bfloat16 precision
    device_map="auto",
    trust_remote_code=True,  # Phi-4 requires this
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# Set EOS token for Phi-4 (uses <|im_end|> not <|endoftext|>)
if tokenizer.eos_token is None or tokenizer.eos_token == "<|endoftext|>":
    print("⚠️  Setting correct EOS token: <|im_end|>")
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("==> Model loaded successfully.")

# 2. Add LoRA Adapters
print("==> 2. Adding LoRA adapters...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],  # Phi-4 specific
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
        per_device_train_batch_size=1,  # Reduced from 2 for memory efficiency
        gradient_accumulation_steps=8,  # Increased from 4 to maintain effective batch size
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
        output_dir=f"outputs/finetune/phi-4/{output_model_name}",
        report_to="none",
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory savings
    ),
)
print("==> Trainer setup complete.")

# 5. Start Training
print("\n" + "="*20)
print("   STARTING LORA TRAINING")
print("="*20 + "\n")
trainer.train()
print("\n" + "="*20)
print("   TRAINING COMPLETE")
print("="*20 + "\n")

# 6. Save the Final Model
print("==> 6. Saving LoRA adapters...")
model.save_pretrained(f"./models/phi-4/{output_model_name}")
tokenizer.save_pretrained(f"./models/phi-4/{output_model_name}")
print(f"==> Adapters saved successfully to ./models/phi-4/{output_model_name}")
