#!/usr/bin/env python

import pandas as pd
from sklearn.utils import shuffle

# =========================
# CONFIGURATION
# =========================

TUNIZI_FILE = "TUNIZI_criteria_only.csv"
KUNDAN_FILE = "Kundan_criteria_only.csv"
FRENCH_FILE = "Kinoux_French_criteria_only.csv"
OUTPUT_FILE = "TAQYIM_all_languages_criteria_shuffled.csv"


def load_tunizi(path):
    df = pd.read_csv(path)

    if "InputText" not in df.columns or "criteria" not in df.columns:
        raise ValueError("TUNIZI file must contain 'InputText' and 'criteria' columns.")

    df = df.rename(columns={"InputText": "text"})
    df = df[["text", "criteria"]]
    return df


def load_kundan(path):
    df = pd.read_csv(path)

    if "review_text" not in df.columns or "criteria" not in df.columns:
        raise ValueError("Kundan file must contain 'review_text' and 'criteria' columns.")

    df = df.rename(columns={"review_text": "text"})
    df = df[["text", "criteria"]]
    return df


def load_french(path):
    df = pd.read_csv(path)

    if "text" not in df.columns or "criteria" not in df.columns:
        raise ValueError("French file must contain 'text' and 'criteria' columns.")

    df = df[["text", "criteria"]]
    return df


def main():
    tunizi = load_tunizi(TUNIZI_FILE)
    kundan = load_kundan(KUNDAN_FILE)
    french = load_french(FRENCH_FILE)

    combined = pd.concat([tunizi, kundan, french], ignore_index=True)

    # Drop rows with missing criteria, if any
    combined = combined.dropna(subset=["criteria"])

    # Randomly shuffle all rows
    combined = shuffle(combined, random_state=42).reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Shuffled combined dataset saved to {OUTPUT_FILE}")
    print(f"Total rows: {len(combined)}")


if __name__ == "__main__":
    main()