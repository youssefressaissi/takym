#!/usr/bin/env python

import re
import pandas as pd
from tqdm import tqdm
import ollama
from datasets import load_dataset

# =========================
# CONFIGURATION
# =========================

DATASET_NAME = "Kinoux/french-customer-review-sentiment-free-2k"
OUTPUT_FILE = "Kinoux_French_criteria_only.csv"
MODEL_NAME = "qwen2.5:3b"

# New English taxonomy (grouped from your French list)
ENGLISH_TAXONOMY = [
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
    "Global_experience"
]

# French + English keyword map for hints
KEYWORD_MAP = {
    "Food_quality": [
        "qualité des plats", "plat", "plats", "nourriture", "repas",
        "goût", "délicieux", "dégueulasse", "bon", "mauvais",
        "qualité des boissons", "boisson", "boissons",
        "food", "meal", "taste", "drink", "drinks"
    ],
    "Presentation": [
        "présentation", "dressage", "look", "presentation"
    ],
    "Ambience": [
        "ambiance", "ambiance / cadre", "cadre", "décor",
        "musique", "bruyant", "calme", "atmosphère",
        "atmosphere", "music", "noise", "noisy"
    ],
    "Service": [
        "service", "serveur", "serveuse", "serveurs", "serveuses",
        "personnel", "staff", "accueil", "accueil / écoute",
        "qualité du service", "service client",
        "polite", "rude", "friendly", "customer service"
    ],
    "Speed": [
        "rapidité du service", "service rapide", "service lent",
        "délai", "délais", "retard", "attente", "attendu",
        "long", "lent", "rapide", "vitesse",
        "rapidité en caisse", "rapidité de traitement",
        "wait", "waiting", "delay", "late", "fast", "slow"
    ],
    "Price_value": [
        "prix", "trop cher", "pas cher", "cher", "bon marché",
        "rapport qualité-prix", "coût", "facture",
        "expensive", "cheap", "price", "cost", "value"
    ],
    "Cleanliness_hygiene": [
        "propreté", "propre", "sale", "saleté",
        "hygiene", "hygiène", "odeur", "odeurs",
        "poussière", "toilettes",
        "clean", "dirty", "hygiene", "smell", "odor"
    ],
    "Product_quality": [
        "qualité des produits", "produit", "produits",
        "qualité de la réparation", "réparation",
        "fiabilité", "honnêteté",
        "product quality", "reliability"
    ],
    "Availability": [
        "choix", "disponibilité", "disponibilité des produits",
        "disponibilité médicaments", "rupture de stock",
        "stock", "availability", "out of stock"
    ],
    "Information": [
        "clarté des informations", "information", "informations",
        "explications", "communication",
        "conseil", "accompagnement",
        "information clarity", "advice", "support"
    ],
    "Care_quality": [
        "qualité des soins", "soins", "médecin", "docteur",
        "compétence du personnel", "compétent", "incompétent",
        "care quality", "doctor", "nurse", "staff competence"
    ],
    "Global_experience": [
        "expérience globale", "global", "expérience générale"
    ]
}


# =========================
# TEXT CLEANING (FRENCH/ENGLISH)
# =========================

def clean_review(text):
    """
    Basic cleaner for French/English reviews:
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

    txt = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\s\.'!?]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    return txt


def detect_hint(clean_text):
    """
    Look for keywords in the cleaned text.
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

def get_french_criterion(row):
    """
    Use Qwen (via Ollama) to assign one criterion to a French review.
    Row contains 'text' and optional 'sentiment'.
    """

    original_text = row.get("text", "")
    text_clean = clean_review(original_text)
    hint_crit = detect_hint(text_clean)

    sentiment = row.get("sentiment", "")

    hint = ""
    if hint_crit is not None:
        hint = (
            f"NOTE: The review contains words related to '{hint_crit}'. "
            f"You should strongly prefer the criterion '{hint_crit}' over 'Global_experience'."
        )

    taxonomy_text = (
        "English Taxonomy (you MUST choose exactly one of these labels):\n"
        "- Food_quality\n"
        "- Presentation\n"
        "- Ambience\n"
        "- Service\n"
        "- Speed\n"
        "- Price_value\n"
        "- Cleanliness_hygiene\n"
        "- Product_quality\n"
        "- Availability\n"
        "- Information\n"
        "- Care_quality\n"
        "- Global_experience (only if the review is very short or too generic)\n"
    )

    context_text = f"Additional context (global sentiment label, may be empty): {sentiment}\n"

    prompt = f"""
[INST] You are a customer-experience analyst for French reviews.

Task:
- Read the following French customer review (possibly with a global sentiment label).
- Decide which ONE criterion from the English Taxonomy best describes what the review is talking about.

Review (French):
\"{original_text}\"

Cleaned review:
\"{text_clean}\"

{context_text}

{hint}

{taxonomy_text}

Rules:
- You MUST output only ONE criterion, chosen from the list above.
- Use 'Global_experience' ONLY if the review does not clearly talk about a specific aspect
  such as price/value, speed, cleanliness/hygiene, food quality, ambience, service, etc.
- Do NOT invent new labels.
- Use the sentiment label only as a hint; the main signal comes from the review text.

Output format:
Criterion

Examples:
Food_quality
Service
Price_value
Cleanliness_hygiene
[/INST]
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        full_res = response["message"]["content"].strip()

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
    print(f"Loading dataset {DATASET_NAME} from Hugging Face...")
    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.to_pandas()

    # Limit to 100 rows for now
    #df = df.head(100)

    if "text" not in df.columns:
        raise ValueError("Expected column 'text' in Kinoux French dataset.")

    cols_to_keep = [c for c in df.columns if c in ["text", "sentiment"]]
    df = df[cols_to_keep]

    print(f"Starting French relabeling for {len(df)} rows using model {MODEL_NAME}...")
    tqdm.pandas()

    df["criteria"] = df.progress_apply(get_french_criterion, axis=1)

    df = df[["text", "criteria"]]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSUCCESS: labeled Kinoux French dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
