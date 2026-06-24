#!/usr/bin/env python

import os
import shutil
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# ==========================================
# CONFIG
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TRAIN_FILE = os.path.join(BASE_DIR, "..", "taqyim_train.csv")
VAL_FILE   = os.path.join(BASE_DIR, "..", "taqyim_val.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "taqyim_model_final")
CACHE_DIR  = os.path.join(BASE_DIR, "..", "unsloth_compiled_cache")

MODEL_NAME = "unsloth/llama-3.2-1b-instruct"

max_seq_length = 300
dtype = None
load_in_4bit = True

# Unified taxonomy (for info in the prompt, not used directly here)
TAQYIM_CRITERIA = [
    "Food_quality",
    "Presentation",
    "Ambience",
    "Service",
    "Speed",
    "Price_value",
    "Cleanliness_hygiene",
    "Product_quality",
    "Availability",
    "Information",
    "Care_quality",
    "Global_experience",
    "Delay",
    "Price",
    "Quality",
]

# Basic bad-word list (extend for your context)
BAD_WORDS = [
    "fuck", "fucking", "shit", "bitch", "bastard",
    "merde", "con", "putain",
    "zebb", "zebi", "khra", "5ra","mibouna", "mibouna", "mibouni", "mibounou", "mibounouh","rab"
]

# ==========================================
# 1. FORCE CLEANUP (cache)
# ==========================================

if os.path.exists(CACHE_DIR):
    print("Removing previous Unsloth cache...")
    shutil.rmtree(CACHE_DIR)

# ==========================================
# 2. LOAD MODEL (RTX 3050 Optimized)
# ==========================================

print(f"Loading model {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = max_seq_length,
    dtype           = dtype,
    load_in_4bit    = load_in_4bit,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==========================================
# 3. ADD LORA ADAPTERS
# ==========================================

print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0.0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 4. DATA LOADING & FORMATTING
# ==========================================

print("Loading TAQYIM train/val splits...")
train_ds = load_dataset("csv", data_files=TRAIN_FILE, split="train")
val_ds   = load_dataset("csv", data_files=VAL_FILE,   split="train")

def base_filter(example):
    return example.get("text") is not None and example.get("criteria") is not None

train_ds = train_ds.filter(base_filter)
val_ds   = val_ds.filter(base_filter)

# --- Bad-word filter: remove extremely short, purely abusive reviews ---
def bad_word_filter(example):
    txt = str(example["text"]).lower()
    tokens = txt.split()
    hits = [w for w in BAD_WORDS if w in txt]
    # Drop if very short (<= 3 words) and contains insults
    return not (len(tokens) <= 3 and len(hits) > 0)

print("Filtering short, purely abusive reviews...")
train_ds = train_ds.filter(bad_word_filter)
val_ds   = val_ds.filter(bad_word_filter)

# Filter by length to avoid OOM
def filter_long_samples(example):
    tokens = tokenizer.encode(example["text"])
    return len(tokens) < (max_seq_length - 50)

print("Filtering long samples...")
train_ds = train_ds.filter(filter_long_samples)
val_ds   = val_ds.filter(filter_long_samples)

taxonomy_str = ", ".join(TAQYIM_CRITERIA)

def formatting_prompts_func(examples):
    """
    Build an instruction-style prompt for multilingual TAQYIM classification.
    Input: examples["text"], examples["criteria"]
    Output: examples["text"] overwritten with formatted dialogue, truncated to max_seq_length.
    """
    instructions = (
        "You are an AI assistant for TAQYIM.\n"
        "Task: Read the customer review and classify it into ONE criterion from the TAQYIM taxonomy.\n"
        f"Possible criteria: {taxonomy_str}.\n"
        "Always answer ONLY with the criterion name (e.g., Service, Price_value, Speed).\n"
    )
    inputs  = examples["text"]
    labels  = examples["criteria"]

    texts = []
    for inp, lab in zip(inputs, labels):
        t = (
            f"<|im_start|>system\n{instructions}<|im_end|>\n"
            f"<|im_start|>user\nReview:\n{inp}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{lab}<|im_end|>"
        )
        # Explicit truncation to max_seq_length
        encoded = tokenizer(
            t,
            max_length = max_seq_length,
            truncation = True,
            add_special_tokens = False,
        )
        t_trunc = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
        texts.append(t_trunc)

    return {"text": texts}

# ==========================================
# 5. TRAINING CONFIGURATION
# ==========================================

training_args = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    warmup_steps = 20,
    num_train_epochs = 10,          # you can increase later
    learning_rate = 2e-4,
    fp16 = False,
    bf16 = True,                   # for RTX 3050
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = os.path.join(BASE_DIR, "outputs_qwen"),
    save_strategy = "epoch",       # save at each epoch
    save_total_limit = 2,
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    eval_dataset       = val_ds,
    dataset_text_field = "text",
    max_seq_length     = max_seq_length,
    dataset_num_proc   = 2,
    args               = training_args,
)

# ==========================================
# 6. EXECUTION
# ==========================================

model.config.use_cache = False
print("--- Training is starting ---")
trainer.train()

print("--- Evaluation on validation set ---")
eval_res = trainer.evaluate()
print(eval_res)

# ==========================================
# 7. SAVE FINAL MODEL
# ==========================================

print(f"Saving final model to {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("--- SUCCESS ---")