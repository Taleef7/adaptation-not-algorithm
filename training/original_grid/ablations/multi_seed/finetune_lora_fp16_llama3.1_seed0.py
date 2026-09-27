# training/original_grid/ablations/multi_seed/finetune_lora_fp16_llama3.1_seed0.py
# Multi-seed validation: seed=0 variant of finetune_lora_fp16_llama3.1.py
#
# Uses standard HuggingFace PEFT + TRL rather than Unsloth (the seed-3407 original-grid
# cell used Unsloth). Settings kept consistent with that run:
#   - Same dataset source: load_dataset("yahma/alpaca-cleaned")
#   - Same formatting function and Alpaca prompt template
#   - Same LoRA config: r=16, alpha=16, dropout=0, 7 target modules, bias=none
#   - Same training args: lr=2e-4, 1 epoch, bs=2, grad_acc=4, warmup=10
#   - Same packing=True, max_seq_length=2048
#   - Optimizer: paged_adamw_8bit (bitsandbytes equivalent of original adamw_8bit)
#   - Only changes: standard PEFT LoraConfig instead of FastLanguageModel.get_peft_model,
#     SFTConfig (TRL >=0.12 API) instead of TrainingArguments for SFT-specific settings.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

SEED = 0
model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
output_model_name = "llama-3.1-8b-lora-alpaca-seed0"
max_seq_length = 2048

# 1. Load the Model
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

# 2. Add LoRA Adapters
# Identical to original: r=16, alpha=16, dropout=0, same 7 target modules
print("==> 2. Adding LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("==> LoRA adapters added successfully.")

# 3. Load and Format the Dataset
# Same source and formatting as original seed=3407 run
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
# SFTConfig (TRL >=0.12): SFT-specific settings (packing, max_seq_length,
# dataset_text_field) belong in SFTConfig, not in SFTTrainer.__init__.
print("==> 4. Setting up SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = max_seq_length,
        packing = True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        bf16 = True,
        logging_steps = 1,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = SEED,
        output_dir = f"outputs/finetune/llama3.1/{output_model_name}",
        report_to = "none",
    ),
)
print("==> Trainer setup complete.")

# 5. Start Training
print("\n" + "="*20)
print(f"   STARTING TRAINING (seed={SEED}, standard HF PEFT)")
print("="*20 + "\n")
trainer.train()
print("\n" + "="*20)
print("   TRAINING COMPLETE")
print("="*20 + "\n")

# 6. Save the Final Model (LoRA adapters + tokenizer)
print("==> 6. Saving LoRA adapters...")
model.save_pretrained(f"./models/llama3.1/{output_model_name}")
tokenizer.save_pretrained(f"./models/llama3.1/{output_model_name}")
print(f"==> Adapters saved successfully to ./models/llama3.1/{output_model_name}")
