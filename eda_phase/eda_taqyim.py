#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.ensemble import IsolationForest

sns.set(style="whitegrid")

BASE_OUTPUT_DIR = "eda_advanced"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# ========== UTILITIES ==========

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def split_cols_by_type(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols

def normalize_dtypes(df):
    """
    Convert pandas extension dtypes (like StringDtype) to standard dtypes.
    """
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.StringDtype):
            df[col] = df[col].astype("object")
    return df


# ========== 1. PROFILING DATA ==========

def profile_data(df, name, out_dir):
    print(f"\n=== [1] PROFILING DATA: {name} ===")
    profile = []

    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Memory usage: {mem_usage_mb:.2f} MB")

    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype

        n_missing = col_data.isna().sum()
        n_unique = col_data.nunique(dropna=True)
        missing_pct = 100 * n_missing / len(df) if len(df) > 0 else 0
        is_constant = (n_unique <= 1)

        entry = {
            "column": col,
            "dtype": str(dtype),
            "n_missing": n_missing,
            "missing_pct": missing_pct,
            "n_unique": n_unique,
            "is_constant": is_constant
        }

        try:
            is_numeric = pd.api.types.is_numeric_dtype(col_data)
        except Exception:
            is_numeric = False

        if is_numeric:
            entry.update({
                "mean": col_data.mean(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "q25": col_data.quantile(0.25),
                "median": col_data.median(),
                "q75": col_data.quantile(0.75),
                "skew": col_data.skew(),
                "kurtosis": col_data.kurtosis()
            })
        else:
            entry.update({
                "mode": col_data.mode(dropna=True).iloc[0]
                if not col_data.mode(dropna=True).empty
                else None
            })

        profile.append(entry)

    profile_df = pd.DataFrame(profile)
    profile_df_path = os.path.join(out_dir, f"{name}_profile.csv")
    profile_df.to_csv(profile_df_path, index=False)
    print(f"Saved profile report: {profile_df_path}")

    high_missing = profile_df[profile_df["missing_pct"] > 50]
    high_cardinality = profile_df[
        (profile_df["dtype"] == "object") & (profile_df["n_unique"] > 100)
    ]
    constant_cols = profile_df[profile_df["is_constant"]]

    print("\nColumns with >50% missing values:")
    print(high_missing[["column", "missing_pct"]])

    print("\nHigh-cardinality categorical columns (>100 unique values):")
    print(high_cardinality[["column", "n_unique"]])

    print("\nConstant columns:")
    print(constant_cols[["column"]])


# ========== 2. DISTRIBUTION ANALYSIS ==========

def analyze_distributions(df, name, out_dir):
    print(f"\n=== [2] DISTRIBUTION ANALYSIS: {name} ===")
    num_cols, cat_cols = split_cols_by_type(df)

    for col in num_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue

        plt.figure(figsize=(7, 4))
        sns.histplot(col_data, kde=True)
        skew = col_data.skew()
        kurt = col_data.kurtosis()
        plt.title(f"{name} - {col} (skew={skew:.2f}, kurt={kurt:.2f})")
        plt.tight_layout()
        path_hist = os.path.join(out_dir, f"{name}_{col}_hist.png")
        plt.savefig(path_hist)
        plt.close()

        plt.figure(figsize=(5, 3))
        sns.boxplot(x=col_data)
        plt.title(f"{name} - {col} boxplot")
        plt.tight_layout()
        path_box = os.path.join(out_dir, f"{name}_{col}_box.png")
        plt.savefig(path_box)
        plt.close()

    for col in cat_cols:
        counts = df[col].value_counts(dropna=False).head(20)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts.values, y=counts.index)
        plt.title(f"{name} - {col} top categories")
        plt.xlabel("Count")
        plt.tight_layout()
        path_bar = os.path.join(out_dir, f"{name}_{col}_bar.png")
        plt.savefig(path_bar)
        plt.close()

    print(f"Saved distribution plots for {len(num_cols)} numeric and {len(cat_cols)} categorical columns")


# ========== 3. CORRELATIONS & RELATIONSHIPS ==========

def analyze_correlations(df, name, out_dir):
    print(f"\n=== [3] CORRELATION ANALYSIS: {name} ===")
    num_cols, _ = split_cols_by_type(df)
    if len(num_cols) < 2:
        print("Not enough numeric columns for correlation analysis.")
        return

    num_df = df[num_cols].dropna()
    if num_df.empty:
        print("Numeric data is empty after dropping NA.")
        return

    corr_methods = ["pearson", "spearman", "kendall"]
    for method in corr_methods:
        corr = num_df.corr(method=method)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title(f"{name} - Correlation ({method})")
        plt.tight_layout()
        path_heat = os.path.join(out_dir, f"{name}_corr_{method}.png")
        plt.savefig(path_heat)
        plt.close()
        print(f"Saved {method} correlation heatmap: {path_heat}")

    corr = num_df.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_pairs = [
        (i, j, upper.loc[i, j])
        for i in upper.index for j in upper.columns
        if upper.loc[i, j] > 0.7
    ]

    for (i, j, v) in high_pairs[:10]:
        plt.figure(figsize=(5, 4))
        sns.scatterplot(x=num_df[i], y=num_df[j], alpha=0.5)
        plt.title(f"{name} - {i} vs {j} (corr={v:.2f})")
        plt.tight_layout()
        path_scatter = os.path.join(out_dir, f"{name}_scatter_{i}_vs_{j}.png")
        plt.savefig(path_scatter)
        plt.close()
        print(f"Saved scatter plot: {path_scatter}")


# ========== 4. OUTLIER DETECTION (IQR + IsolationForest) ==========

def detect_outliers(df, name, out_dir):
    print(f"\n=== [4] OUTLIER DETECTION (IQR): {name} ===")
    num_cols, _ = split_cols_by_type(df)
    if not num_cols:
        print("No numeric columns, skipping outlier detection.")
        return

    num_df = df[num_cols].dropna()
    if num_df.empty:
        print("Numeric data empty after dropping NA, skipping outliers.")
        return

    outlier_flags = pd.DataFrame(index=num_df.index)
    for col in num_cols:
        col_data = num_df[col]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_flags[col] = (col_data < lower) | (col_data > upper)

    outlier_counts = outlier_flags.sum(axis=1)
    df_outliers = num_df[outlier_counts > 0]
    print(f"Number of rows with at least one IQR outlier: {df_outliers.shape[0]}")

def isolation_forest_outliers(df, name, out_dir, n_estimators=100, contamination=0.05):
    num_cols, _ = split_cols_by_type(df)
    if not num_cols:
        print(f"[IsolationForest] {name}: no numeric columns, skipping.")
        return

    num_df = df[num_cols].dropna()
    if num_df.empty:
        print(f"[IsolationForest] {name}: numeric data empty after dropping NA, skipping.")
        return

    print(f"\n[IsolationForest] Running on {name} with {len(num_cols)} numeric features, "
          f"{num_df.shape[0]} rows")

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42
    )
    preds = iso.fit_predict(num_df)
    scores = iso.decision_function(num_df)

    num_df["iso_label"] = preds
    num_df["iso_score"] = scores

    outliers = num_df[num_df["iso_label"] == -1]
    print(f"[IsolationForest] {name}: {outliers.shape[0]} outliers detected "
          f"({100 * outliers.shape[0] / num_df.shape[0]:.2f}% of rows).")

    out_path = os.path.join(out_dir, f"{name}_isolation_forest_outliers.csv")
    outliers.to_csv(out_path, index=False)
    print(f"[IsolationForest] Saved outlier details: {out_path}")

    if len(num_cols) >= 2:
        x_col, y_col = num_cols[0], num_cols[1]
        plt.figure(figsize=(6, 5))
        plt.scatter(num_df[x_col], num_df[y_col], c="lightgray", label="Inliers", alpha=0.5)
        plt.scatter(outliers[x_col], outliers[y_col], c="red", label="Outliers", alpha=0.7)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{name} - IsolationForest outliers ({x_col} vs {y_col})")
        plt.legend()
        plt.tight_layout()
        scatter_path = os.path.join(out_dir, f"{name}_isolation_forest_scatter.png")
        plt.savefig(scatter_path)
        plt.close()
        print(f"[IsolationForest] Saved scatter plot: {scatter_path}")


# ========== 5. MISSING DATA ANALYSIS ==========

def analyze_missing_data(df, name, out_dir):
    print(f"\n=== [5] MISSING DATA ANALYSIS: {name} ===")
    missing_counts = df.isna().sum()
    missing_pct = missing_counts * 100 / len(df) if len(df) > 0 else 0
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_counts.values,
        "missing_pct": missing_pct.values
    })
    missing_path = os.path.join(out_dir, f"{name}_missing_summary.csv")
    missing_df.to_csv(missing_path, index=False)
    print(f"Saved missing data summary: {missing_path}")

    plt.figure(figsize=(10, 4))
    sns.barplot(x="column", y="missing_pct", data=missing_df)
    plt.xticks(rotation=90)
    plt.ylabel("Missing %")
    plt.title(f"{name} - Missingness per column")
    plt.tight_layout()
    path_bar = os.path.join(out_dir, f"{name}_missing_bar.png")
    plt.savefig(path_bar)
    plt.close()
    print(f"Saved missingness bar plot: {path_bar}")


# ========== LOADERS FOR YOUR 3 DATASETS ==========

def load_french_dataset():
    ds = load_dataset("Kinoux/french-customer-review-sentiment-free-2k", split="train")
    df = ds.to_pandas()
    return df

def load_tunizi_dataset():
    tunizi_path = "TUNIZI.csv"  # same folder as this script
    df = pd.read_csv(tunizi_path)
    return df

def load_kundan_dataset():
    kundan_path = "Kundan_Customer.csv"  # same folder as this script
    df = pd.read_csv(kundan_path)
    return df


# ========== MAIN PIPELINE ==========

def run_full_eda(df, name):
    df = normalize_dtypes(df)
    out_dir = ensure_dir(os.path.join(BASE_OUTPUT_DIR, name))

    profile_data(df, name, out_dir)
    analyze_distributions(df, name, out_dir)
    analyze_correlations(df, name, out_dir)
    detect_outliers(df, name, out_dir)
    isolation_forest_outliers(df, name, out_dir)
    analyze_missing_data(df, name, out_dir)

def main():
    # 1) French dataset (Hugging Face)
    df_fr = load_french_dataset()
    run_full_eda(df_fr, "Kinoux_French")

    # 2) TUNIZI dataset (InputText, SentimentLabel)
    df_tu = load_tunizi_dataset()
    run_full_eda(df_tu, "TUNIZI")

    # 3) Kundan Customer dataset
    df_ku = load_kundan_dataset()
    run_full_eda(df_ku, "Kundan_Customer")

if __name__ == "__main__":
    main()
