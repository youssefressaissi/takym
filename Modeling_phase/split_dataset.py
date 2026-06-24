#!/usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "/home/youssefressaissi/takym/Relabled_Datasets/TAQYIM_all_languages_criteria_shuffled.csv"
TRAIN_FILE = "taqyim_train.csv"
VAL_FILE = "taqyim_val.csv"
TEST_FILE = "taqyim_test.csv"

def main():
    df = pd.read_csv(INPUT_FILE)

    if not {"text", "criteria"}.issubset(df.columns):
        raise ValueError("Expected columns 'text' and 'criteria' in input CSV.")

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["criteria"]
    )

    # Second split: train vs val
    train, val = train_test_split(
        train_val,
        test_size=0.2,  # 0.2 of 0.8 -> 0.16 overall
        random_state=42,
        stratify=train_val["criteria"]
    )

    train.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")
    val.to_csv(VAL_FILE, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_FILE, index=False, encoding="utf-8-sig")

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    main()
