# Original-grid Llama-3.1-8B-Instruct LoRA (Unsloth, seed 3407).
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
output_model_name = "llama-3.1-8b-lora-alpaca"

# 1. Load the Model
max_seq_length = 2048
dtype = None
# Full-precision (bf16) base, not 4-bit
load_in_4bit = False

print(f"==> 1. Loading model: {model_name_or_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name_or_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("==> Model loaded successfully.")

# 2. Add LoRA Adapters
print("==> 2. Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
print("==> LoRA adapters added successfully.")

# 3. Load and Format the Dataset
print("==> 3. Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)
print("==> Dataset prepared.")

# 4. Set Up the Trainer
print("==> 4. Setting up SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = True,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = f"outputs/finetune/llama3.1/{output_model_name}",
        report_to = "none",
    ),
)
print("==> Trainer setup complete.")

# 5. Start Training
print("\n" + "="*20)
print("   STARTING TRAINING")
print("="*20 + "\n")
trainer.train()
print("\n" + "="*20)
print("   TRAINING COMPLETE")
print("="*20 + "\n")

# 6. Save the Final Model
print("==> 6. Saving LoRA adapters...")
model.save_pretrained(f"./models/llama3.1/{output_model_name}")
print(f"==> Adapters saved successfully to ./models/llama3.1/{output_model_name}")