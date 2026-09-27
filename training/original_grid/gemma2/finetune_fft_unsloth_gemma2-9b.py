import os
from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer

# --- Configuration ---
model_name_or_path = "models/gemma2-9b/gemma2-9b-it-base"
dataset_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "data/alpaca_cleaned_dataset")
output_model_name = "gemma2-9b-it-fft-alpaca"
max_seq_length = 2048

# 1. Load Model with the 'eager' attention implementation
print(f"==> 1. Loading model with Unsloth for FFT: {model_name_or_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name_or_path,
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = False,
    attn_implementation = "eager",
)
print("==> Model loaded successfully.")

# =================================================================================
# Unfreeze all parameters for full fine-tuning (Unsloth loads the model with
# parameters frozen by default).
# =================================================================================
for param in model.parameters():
  param.requires_grad = True
print("==> All model parameters unfrozen for FFT.")


# 2. Prepare Dataset
print("==> 2. Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = load_from_disk(dataset_path)
dataset = dataset.map(formatting_func, batched = True,)
print("==> Dataset prepared.")

# 3. Set Up the Trainer with optimized arguments
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 1,
        learning_rate = 2e-5,
        bf16 = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = f"outputs/finetune/gemma2-9b/{output_model_name}",
        report_to = "none",
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
    ),
)
print("==> Trainer setup complete.")

# 4. Start Training
print("\n" + "="*20)
print("   STARTING Unsloth FFT TRAINING")
print("="*20 + "\n")
trainer.train()
print("\n" + "="*20)
print("   TRAINING COMPLETE")
print("="*20 + "\n")

# 5. Save Model
print("==> 5. Saving FFT Model...")
trainer.save_model(f"./models/gemma2-9b/{output_model_name}")
print(f"==> Model saved successfully to ./models/gemma2-9b/{output_model_name}")