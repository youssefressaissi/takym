#!/usr/bin/env python

import os
import time
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_FILE = os.path.join(BASE_DIR, "/home/youssefressaissi/takym/Modeling_phase/taqyim_test.csv")

# Paths to your fine-tuned models
MODELS = {
    "qwen_3b":   os.path.join(BASE_DIR, "/home/youssefressaissi/takym/Modeling_phase/taqyim_model_Qwen_finetuned"),
    "llama_1b":  os.path.join(BASE_DIR, "/home/youssefressaissi/takym/Modeling_phase/ollama_instructed_finetuned"),
}

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(REPORTS_DIR, "transformers_eval.txt")

# Same taxonomy list you used in training
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

taxonomy_str = ", ".join(TAQYIM_CRITERIA)


# =========================
# UTILS
# =========================

def build_prompt(review: str) -> str:
    """Build the same ChatML-style prompt used during training."""
    instructions = (
        "You are an AI assistant for TAQYIM.\n"
        "Task: Read the customer review and classify it into ONE criterion from the TAQYIM taxonomy.\n"
        f"Possible criteria: {taxonomy_str}.\n"
        "Always answer ONLY with the criterion name (e.g., Service, Price_value, Speed).\n"
    )
    t = (
        f"<|im_start|>system\n{instructions}<|im_end|>\n"
        f"<|im_start|>user\nReview:\n{review}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return t


def normalize_prediction(raw: str) -> str:
    """Post-process model output to a clean TAQYIM label."""
    pred = raw.strip()

    # Keep only first line / first token, in case of extra text
    pred = pred.split("\n")[0].strip()

    # Fix trivial variations (case, spaces)
    pred = pred.replace(" ", "_")
    pred = pred.replace("-", "_")
    pred = pred.strip()

    # Try to map to known criteria
    if pred in TAQYIM_CRITERIA:
        return pred

    # Optional: small heuristic mappings (example)
    mapping = {
        "Service_quality": "Service",
        "Food": "Food_quality",
        "Hygiene": "Cleanliness_hygiene",
        "Global": "Global_experience",
    }
    if pred in mapping:
        return mapping[pred]

    # If unknown, you can treat it as a special error label or "Global_experience"
    return "Global_experience"


def evaluate_model(model_name: str, model_path: str, df_test: pd.DataFrame, device: str, f_handle):
    """Run evaluation for a single fine-tuned model."""
    print(f"\nLoading model {model_name} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model.eval()

    y_true = []
    y_pred = []

    start_time = time.time()

    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc=f"Evaluating {model_name}"):
        text = str(row["text"])
        label = str(row["criteria"])

        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                num_beams=1,
            )

        # Extract only the new tokens after the prompt
        generated = out[0][inputs["input_ids"].shape[1]:]
        raw_answer = tokenizer.decode(generated, skip_special_tokens=True)

        pred_label = normalize_prediction(raw_answer)

        y_true.append(label)
        y_pred.append(pred_label)

    elapsed = time.time() - start_time

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, labels=TAQYIM_CRITERIA, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=TAQYIM_CRITERIA)

    # Write to report
    f_handle.write(f"\n=== {model_name} on TAQYIM test set ===\n")
    f_handle.write(f"Accuracy: {acc:.4f}\n")
    f_handle.write("\nClassification report:\n")
    f_handle.write(cls_report)
    f_handle.write("\nConfusion matrix (labels in TAQYIM_CRITERIA order):\n")
    f_handle.write(str(cm))
    f_handle.write(f"\nElapsed time: {elapsed:.2f} seconds for {len(df_test)} samples\n")


# =========================
# MAIN
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading TAQYIM test set...")
    df_test = pd.read_csv(TEST_FILE)
    assert {"text", "criteria"}.issubset(df_test.columns)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("Transformer-based models on TAQYIM (test set)\n")
        f.write("============================================\n")

        for name, path in MODELS.items():
            evaluate_model(name, path, df_test, device, f)

    print(f"\nAll transformer evaluation results written to {REPORT_FILE}")


if __name__ == "__main__":
    main()