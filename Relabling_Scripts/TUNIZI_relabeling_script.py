import re
import pandas as pd
from tqdm import tqdm
import ollama

# =========================
# CONFIGURATION
# =========================

INPUT_FILE = "TUNIZI.csv"                     # Original TUNIZI dataset
OUTPUT_FILE = "TUNIZI_criteria_only.csv"      # Output file
MODEL_NAME = "qwen2.5:3b"

# 1) TUNISIAN KEYWORD MAP (ARABIZI ONLY)
#    Keys are ENGLISH CRITERIA (what the model must output).
KEYWORD_MAP = {
    "Food_quality": [  # taste / food
        "bnina", "mekla", "tayra", "hayla", "abann",
        "skhouna", "mouch tayba", "dwa", "bnn", "banni"
    ],
    "Ambience": [
        "ambiance", "jawou", "jaw", "music", "muzika", "ghneya",
        "sout", "bruit", "bruyant", "calm", "calme"
    ],
    "Service": [  # staff / welcome
        "khidma", "m3allem", "m3alem", "metrabya", "yadh7ek",
        "isti9bal", "accueil", "staff", "personnel", "pro",
        "service", "agent"
    ],
    "Speed": [  # delays / time
        "rzin", "rzina", "fisa3", "fissa3", "nestana", "nesta", "stanna",
        "yabta", "ta3til", "wa9t", "sa3at", "d9ayeq",
        "masta3jel", "mesta3jel", "slow", "rapid", " rapide"
    ],
    "Price_value": [  # price / value
        "ghali", "ghlya", "r5is", "r5isa", "soum", "aswam",
        "flous", "5allas", "mouch blech", "bi9adeh", "soumou",
        "prix", "cher", "tres cher", "pas cher"
    ],
    "Cleanliness_hygiene": [
        "ndhifa", "ndhif", "ndhaf", "mamsou5a", "moussekha",
        "wsakh", "wsah", "ghobra", "5anza", "ri7a", "rwaye7",
        "sale", "propre"
    ],
    # For TUNIZI you may not have all taxonomy aspects, but keep them for consistency:
    "Product_quality": [
        "sla3", "produit", "produits", "khal", "t9sir"
    ],
    "Availability": [
        "disponible", "dispo", "ma fama ch", "ma fama7ch",
        "mawjoud", "stock"
    ],
    "Information": [
        "ma5barounich", "ma fihach info", "information", "explication"
    ],
    "Care_quality": [
        "docteur", "tabib", "soin", "soins", "hospital", "spital"
    ],
    "Global_experience": [
        # generic praise / comments without clear aspect
        "bahi", "mezyen", "haya", "bravo", "nice", "amazing"
    ]
}

# 2) SHARED ENGLISH TAXONOMY (OUTPUT LABELS)
ENGLISH_TAXONOMY = [
    "Food_quality",
    "Presentation",            # kept for consistency, even if few Arabizi keywords
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


# =========================
# TEXT CLEANING (ARABIZI-FOCUSED)
# =========================

def clean_review_arabizi(text):
    """
    Clean Arabizi review text:
    - Keep letters (a-z), digits (0-9), and basic punctuation.
    - Remove extra whitespace and obvious social media noise.
    - Crucially, digits 0-9 are kept because they encode sounds (3, 7, 9).
    """
    if pd.isna(text):
        return ""

    txt = str(text).lower()

    # Remove URLs, @mentions, hashtags
    txt = re.sub(r"http\S+", " ", txt)
    txt = re.sub(r"@[^\s]+", " ", txt)
    txt = re.sub(r"#[^\s]+", " ", txt)

    # Keep letters, digits, and basic punctuation; remove other symbols
    txt = re.sub(r"[^a-z0-9\s\.'!?]", " ", txt)

    # Collapse multiple spaces
    txt = re.sub(r"\s+", " ", txt).strip()

    return txt


def detect_tunisian_hint(clean_text):
    """
    Look for Tunisian Arabizi keywords in the cleaned text.
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

def get_tunisian_criterion(original_text):
    """
    Use Qwen (via Ollama) to assign:
    - criteria (one of ENGLISH_TAXONOMY)
    """

    text_clean = clean_review_arabizi(original_text)
    hint_crit = detect_tunisian_hint(text_clean)

    hint = ""
    if hint_crit is not None:
        hint = (
            f"NOTE: The text contains Tunisian Arabizi expressions related to '{hint_crit}'. "
            f"You should strongly prefer the criterion '{hint_crit}' over 'Global_experience'."
        )

    taxonomy_text = (
        "Unified English Taxonomy (you MUST choose exactly one of these labels):\n"
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
        "- Global_experience (only if the text is very short or too generic)\n"
    )

    prompt = f"""
[INST] You are a Tunisian Dialect Expert working in Arabizi (Tunisian Arabic written in Latin letters with digits).

Task:
- Read the following customer review in Tunisian Arabizi (possibly mixed with some French).
- Decide which ONE criterion from the Unified English Taxonomy best describes what the review is talking about.

Review (Arabizi):
\"{original_text}\"

Cleaned version:
\"{text_clean}\"

{hint}

{taxonomy_text}

Rules:
- You MUST output only ONE criterion, chosen from the list above.
- Use 'Global_experience' ONLY if the review is extremely short (3 words or less)
  or does not clearly talk about price/value, time/speed, cleanliness/hygiene,
  food/product quality, ambience, service/staff, availability, information, or care quality.
- Use your understanding of Tunisian dialect and Arabizi (including digits 3, 5, 7, 9) to interpret the meaning.

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

        # Take the last non-empty line as the answer
        lines = [l.strip() for l in full_res.split("\n") if l.strip()]
        if not lines:
            return "Global_experience"

        last_line = lines[-1]

        # Normalize criterion to one of the allowed labels
        crit_norm = last_line.strip()
        crit_norm = crit_norm.split()[0]  # e.g. "Food_quality (Soum)" -> "Food_quality"

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

    if "InputText" not in df.columns:
        raise ValueError("Expected column 'InputText' in TUNIZI dataset.")

    # Keep only InputText and optional SentimentLabel
    cols_to_keep = [c for c in df.columns if c in ["InputText", "SentimentLabel"]]
    df = df[cols_to_keep]

    print(f"Starting Tunisian relabeling for {len(df)} rows using model {MODEL_NAME}...")
    tqdm.pandas()

    # Apply labeling function
    df["criteria"] = df["InputText"].progress_apply(get_tunisian_criterion)

    # Final export – ensure order: InputText, criteria, (optional SentimentLabel)
    col_order = ["InputText", "criteria"]
    if "SentimentLabel" in df.columns:
        col_order.append("SentimentLabel")

    df = df[col_order]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSUCCESS: labeled TUNIZI dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()