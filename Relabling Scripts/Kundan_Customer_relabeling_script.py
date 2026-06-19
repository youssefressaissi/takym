#!/usr/bin/env python

import re
import pandas as pd
from tqdm import tqdm
import ollama

# =========================
# CONFIGURATION
# =========================

INPUT_FILE = "Kundan_Customer.csv"
OUTPUT_FILE = "Kundan_criteria_only.csv"
MODEL_NAME = "qwen2.5:3b"

# English taxonomy for criteria
ENGLISH_TAXONOMY = [
    "Price",
    "Delay",
    "Cleanliness",
    "Quality",
    "Service",
    "Global_experience"
]

# Simple English keyword map to help the model
KEYWORD_MAP = {
    "Price": [
        "expensive", "cheap", "price", "cost", "value", "worth",
        "overpriced", "budget", "fee", "charges"
    ],
    "Delay": [
        "late", "delay", "delayed", "slow", "waiting", "waited",
        "time", "hours", "minutes", "delivery time", "response time"
    ],
    "Cleanliness": [
        "clean", "dirty", "hygiene", "smell", "odor", "stains",
        "messy", "dusty", "filthy"
    ],
    "Quality": [
        "quality", "broken", "faulty", "damaged", "working",
        "excellent", "good", "bad", "taste", "fresh", "stale"
    ],
    "Service": [
        "service", "staff", "support", "rude", "helpful",
        "customer care", "agent", "representative", "employee"
    ]
}


# =========================
# TEXT CLEANING (ENGLISH)
# =========================

def clean_review(text):
    """
    Basic cleaner for English reviews:
    - Lowercase
    - Remove URLs, mentions, hashtags
    - Keep letters, digits, spaces and a few punctuation marks
    """
    if pd.isna(text):
        return ""

    txt = str(text).lower()

    txt = re.sub(r"http\S+", " ", txt)
    txt = re.sub(r"@[^\s]+", " ", txt)
    txt = re.sub(r"#[^\s]+", " ", txt)

    txt = re.sub(r"[^a-z0-9\s\.'!?]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    return txt


def detect_hint(clean_text):
    """
    Look for English keywords in the cleaned text.
    Return the English criterion name if a match is found, else None.
    """
    for crit, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in clean_text:
                return crit
    return None


# =========================
# LABELING WITH OLLAMA
# =========================

def get_kundan_criterion(row):
    """
    Use Qwen (via Ollama) to assign one criterion to a Kundan review.
    Row contains review_text and some optional context fields.
    """

    original_text = row.get("review_text", "")
    text_clean = clean_review(original_text)
    hint_crit = detect_hint(text_clean)

    sentiment = row.get("sentiment", "")
    rating = row.get("customer_rating", "")
    resp_time = row.get("response_time_hours", "")
    issue_resolved = row.get("issue_resolved", "")
    complaint_registered = row.get("complaint_registered", "")

    hint = ""
    if hint_crit is not None:
        hint = (
            f"NOTE: The review contains words related to '{hint_crit}'. "
            f"You should strongly prefer the criterion '{hint_crit}' over 'Global_experience'."
        )

    taxonomy_text = (
        "English Taxonomy (you MUST choose exactly one of these labels):\n"
        "- Price\n"
        "- Delay\n"
        "- Cleanliness\n"
        "- Quality\n"
        "- Service\n"
        "- Global_experience (only if the review is very short or too generic)\n"
    )

    context_text = (
        f"Additional context (may be empty):\n"
        f"- sentiment label: {sentiment}\n"
        f"- customer rating (1-5): {rating}\n"
        f"- response time (hours): {resp_time}\n"
        f"- issue_resolved (yes/no): {issue_resolved}\n"
        f"- complaint_registered (yes/no): {complaint_registered}\n"
    )

    prompt = f"""
[INST] You are a customer-experience analyst.

Task:
- Read the following English customer review (with some optional context).
- Decide which ONE criterion from the English Taxonomy best describes what the review is talking about.

Review:
\"{original_text}\"

Cleaned review:
\"{text_clean}\"

{context_text}

{hint}

{taxonomy_text}

Rules:
- You MUST output only ONE criterion, chosen from the list above.
- Use 'Global_experience' ONLY if the review does not clearly talk about price, time/delay, cleanliness, product/service quality, or service/staff.
- Do NOT invent new labels.
- Use the context variables only as hints; the main signal comes from the review text.

Output format:
Criterion

Examples:
Price
Quality
Service
Cleanliness
[/INST]
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        full_res = response["message"]["content"].strip()

        # Take the last non-empty line
        lines = [l.strip() for l in full_res.split("\n") if l.strip()]
        if not lines:
            return "Global_experience"

        last_line = lines[-1]

        crit_norm = last_line.split()[0].strip()

        if crit_norm not in ENGLISH_TAXONOMY:
            crit_norm = "Global_experience"

        return crit_norm

    except Exception:
        return "Global_experience"


# =========================
# MAIN EXECUTION
# =========================

def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df.sample(n=6000, random_state=42)
    if "review_text" not in df.columns:
        raise ValueError("Expected column 'review_text' in Kundan dataset.")

    cols_to_keep = [c for c in df.columns if c in [
        "review_text", "sentiment", "customer_rating",
        "response_time_hours", "issue_resolved", "complaint_registered"
    ]]
    df = df[cols_to_keep]

    print(f"Starting Kundan relabeling for {len(df)} rows using model {MODEL_NAME}...")
    tqdm.pandas()

    df["criteria"] = df.progress_apply(get_kundan_criterion, axis=1)

    df = df[["review_text", "criteria"]]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSUCCESS: labeled Kundan dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
