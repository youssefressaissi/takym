from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import pandas as pd

# 1. CONFIGURATION DU MODÈLE
max_seq_length = 512 # Limité pour économiser la VRAM de la 3050
dtype = None # Auto-détection
load_in_4bit = True # Obligatoire pour 4 Go de VRAM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. AJOUT DES ADAPTATEURS LORA (Finetuning léger)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rang (plus c'est haut, plus c'est précis mais lourd)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Très important pour 4 Go
    random_state = 3407,
)

# 3. PRÉPARATION DES DONNÉES (CSV -> ChatML)
def formatting_prompts_func(examples):
    instructions = "Classifie cet avis client selon la taxonomie TAQYIM (Critère | Score)."
    inputs       = examples["text"]
    outputs      = [f"{c} | {s}" for c, s in zip(examples["Critere"], examples["Score"])]
    
    texts = []
    for instruction, input_text, output in zip([instructions]*len(inputs), inputs, outputs):
        # Format ChatML utilisé par Qwen
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return { "text" : texts, }

# Charger votre dataset généré
dataset = load_dataset("csv", data_files="taqyim_criteres_full.csv", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. CONFIGURATION DE L'ENTRAÎNEMENT
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1, # Obligatoire à 1 pour 4 Go
        gradient_accumulation_steps = 4, # Simule un batch de 4
        warmup_steps = 5,
        max_steps = 100, # Ajustez selon la taille du dataset (ex: 100-200 pour 2k lignes)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. LANCEMENT
print("Lancement de l'entraînement...")
trainer.train()

# 6. SAUVEGARDE
model.save_pretrained("taqyim_model_lora") # Sauvegarde les poids légers
tokenizer.save_pretrained("taqyim_model_lora")
print("Modèle entraîné et sauvegardé !")