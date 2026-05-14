from unsloth import FastLanguageModel
import torch
import pandas as pd
from tqdm import tqdm

# 1. Load your trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "taqyim_model_final", # Path to your saved folder
    max_seq_length = 256,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # 2x faster inference

def get_local_prediction(text):
    inputs = tokenizer(
    [
        f"<|im_start|>user\nClassifie: {text}<|im_end|>\n<|im_start|>assistant\n"
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64)
    result = tokenizer.batch_decode(outputs)[0]
    # Extract only the assistant's answer
    answer = result.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    return answer

# Test it
# print(get_local_prediction("Le service était vraiment lent mais la mekla bnina."))
