from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
import shutil

# 1. FORCE CLEANUP (Do this every time you change logic)
if os.path.exists("unsloth_compiled_cache"):
    shutil.rmtree("unsloth_compiled_cache")

# ==========================================
# 2. LOAD MODEL (RTX 3050 Optimized)
# ==========================================
max_seq_length = 300 # Slightly increased to avoid tight fits
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Mandatory for Qwen training
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==========================================
# 3. ADD LORA ADAPTERS
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# ==========================================
# 4. DATA FORMATTING & FILTERING
# ==========================================
def formatting_prompts_func(examples):
    instructions = "Classifie cet avis client selon la taxonomie TAQYIM (Critère)."
    inputs       = examples["text"]
    outputs      = examples["Critere"]
    texts = []
    for ins, inp, out in zip([instructions]*len(inputs), inputs, outputs):
        t = f"<|im_start|>system\n{ins}<|im_end|>\n<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
        texts.append(t)
    return { "text" : texts, }

print("Loading dataset...")
dataset = load_dataset("csv", data_files="taqyim_training_ready.csv", split="train")

# CLEANING: Remove NaNs
dataset = dataset.filter(lambda x: x["text"] is not None and x["Critere"] is not None)

# --- CRITICAL FIX: TOKEN-BASED FILTERING ---
# We calculate the length and delete any row that is too long for our 300-token limit.
# This prevents the 's2 vs s4' shape mismatch error entirely.
def filter_long_samples(example):
    tokens = tokenizer.encode(example["text"])
    return len(tokens) < (max_seq_length - 50) # Leave 50 tokens for the prompt/system/label

print("Filtering long reviews to ensure memory stability...")
dataset = dataset.filter(filter_long_samples)
dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 5. TRAINING CONFIGURATION
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8, 
        warmup_steps = 10,
        num_train_epochs = 2, 
        learning_rate = 2e-4,
        fp16 = False, # Set to False for bf16
        bf16 = True,  # Use bf16 for RTX 3050
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        save_strategy = "no",
    ),
)

# ==========================================
# 6. EXECUTION
# ==========================================
print(f"Dataset Size: {len(dataset)} high-quality samples.")
model.config.use_cache = False # Important for training

print("--- Training is starting ---")
trainer.train()

# ==========================================
# 7. SAVE
# ==========================================
model.save_pretrained("taqyim_model_final_10k")
tokenizer.save_pretrained("taqyim_model_final_10k")
print("--- SUCCESS! ---")
