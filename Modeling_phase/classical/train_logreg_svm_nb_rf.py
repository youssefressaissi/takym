#!/usr/bin/env python

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(BASE_DIR, "..", "taqyim_train.csv")
VAL_FILE   = os.path.join(BASE_DIR, "..", "taqyim_val.csv")
TEST_FILE  = os.path.join(BASE_DIR, "..", "taqyim_test.csv")

MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REPORT_FILE = os.path.join(REPORTS_DIR, "classical_results_5.txt")


# =========================
# UTILS
# =========================

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    if not {"text", "criteria"}.issubset(df.columns):
        raise ValueError(f"Expected columns 'text' and 'criteria' in {path}")
    X = df["text"].astype(str)
    y = df["criteria"].astype(str)
    return X, y


def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        min_df=1                 # keep more rare terms
    )


def train_logreg():
    return LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced"
    )


def train_svm():
    return LinearSVC(
        C=1.0,
        class_weight="balanced"
    )


def train_nb():
    return MultinomialNB(alpha=2.0)   # stronger smoothing


def train_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42
    )

def evaluate(model, X, y, name, split_name, f):
    y_pred = model.predict(X)
    f.write(f"\n=== {name} on {split_name} ===\n")
    f.write(classification_report(y, y_pred))
    f.write("\nConfusion matrix:\n")
    f.write(str(confusion_matrix(y, y_pred)))
    f.write("\n")


# =========================
# MAIN
# =========================

def main():
    ensure_dirs()

    print("Loading data...")
    X_train, y_train = load_data(TRAIN_FILE)
    X_val,   y_val   = load_data(VAL_FILE)
    X_test,  y_test  = load_data(TEST_FILE)

    print("Fitting TF-IDF vectorizer on training data...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    X_test_tfidf  = vectorizer.transform(X_test)

    # Models dictionary: name -> constructor
    model_constructors = {
        "logreg": train_logreg,
        "svm":    train_svm,
        "nb":     train_nb,
        "rf":     train_rf,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("Classical models on TAQYIM\n")
        f.write("==========================\n")

        for name, ctor in model_constructors.items():
            print(f"\nTraining {name.upper()}...")
            clf = ctor()
            clf.fit(X_train_tfidf, y_train)

            # Wrap vectorizer + classifier in a pipeline for saving
            pipe = Pipeline([
                ("tfidf", vectorizer),
                ("clf", clf),
            ])

            model_path = os.path.join(MODELS_DIR, f"{name}_tfidf.joblib")
            joblib.dump(pipe, model_path)
            print(f"Saved {name} model to {model_path}")

            # Evaluate on val and test
            evaluate(pipe, X_val,  y_val,  name.upper(), "validation", f)
            evaluate(pipe, X_test, y_test, name.upper(), "test",        f)

    print(f"\nAll results written to {REPORT_FILE}")


if __name__ == "__main__":
    main()