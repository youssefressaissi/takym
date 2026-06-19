# TAQYIM: Multilingual Aspect-Based Sentiment Dataset

## Project overview

This project builds a unified multilingual dataset for **aspect-based sentiment / criteria classification** in the e‑commerce and customer feedback domain. It combines three heterogeneous review datasets (Arabic, English, and French) and harmonizes them into a single corpus with a shared set of criteria labels.

The current repository focuses on **data understanding, cleaning, relabeling, and integration**. Modeling (baseline classifiers, deep learning models, and evaluation) will be added in later stages.

## Data sources

The combined dataset is built from three main sources:

- **TUNIZI**: Arabic customer reviews with an original global sentiment label (`SentimentLabel`) and review text (`InputText`).
- **Kundan**: English customer reviews with review text (`review_text`), metadata (e.g., customer demographics, product info), and a sentiment label.
- **French Kinoux dataset**: French reviews with text (`text`) and sentiment labels.

Each dataset comes with its own schema and label space; the goal of this project is to align them into a common structure suitable for downstream aspect-based sentiment modeling.

## Unified criteria taxonomy

To make the three sources compatible, a **unified set of criteria** (aspects) was defined based on the overlapping themes across datasets and the project’s business goals. Examples of such criteria might include constructs like service quality, product quality, delivery time, or price perception, depending on the original annotations.

All original labels (aspects or sentiments) are mapped to this shared taxonomy. Rare or dataset‑specific categories can be merged into broader criteria when appropriate, and purely global sentiment labels are used only for analysis, not as final targets.

## Data preparation and relabeling

Data preparation is implemented in Python using pandas and follows these main steps:

1. **Loading and initial cleaning**
   - Load each dataset from its original CSV file.
   - Inspect columns, handle obvious anomalies, and drop irrelevant technical columns (e.g., unnamed index columns).

2. **Column standardization**
   - Rename review text columns to a common name `text` (e.g., `InputText` → `text`, `review_text` → `text`).
   - Introduce a unified label column called `criteria`.
   - For the final combined dataset, drop global sentiment columns such as `SentimentLabel` or `sentiment` to avoid confusion between overall sentiment and aspect labels.

3. **Relabeling / label mapping**
   - Design mapping rules from the original labels in each dataset to the unified criteria taxonomy.
   - Apply deterministic rules or heuristics to assign each review a criterion label.
   - Optionally remove instances with missing or ambiguous criteria.

4. **Integration and shuffling**
   - Concatenate the three standardized DataFrames vertically to obtain one corpus with columns `text` and `criteria`.
   - Randomly shuffle all rows using a fixed random seed to remove any ordering by source or language.
   - Save the final result as `TAQYIM_all_languages_criteria_shuffled.csv`.

Intermediate “debug” versions of the data (e.g., with an extra `source` column or with sentiment still present) can be generated but are not part of the final training file.

## Repository structure

A suggested structure for the repository is:

- `data/`
  - `raw/` – original CSVs for TUNIZI, Kundan, and French datasets
  - `processed/` – intermediate cleaned datasets
  - `final/` – `TAQYIM_all_languages_criteria_shuffled.csv`
- `notebooks/` – exploratory data analysis and visualization
- `scripts/`
  - `prepare_data.py` – end‑to‑end data preparation and combination script
- `README.md` – project description (this file)
- `LICENSE` – project license (to be chosen)

## Usage

Minimal example to run the data preparation:

```bash
# From the repository root
python scripts/prepare_data.py
```

This script expects the three input CSVs in `data/raw/` and will write the final combined dataset into `data/final/`.

## Next steps

Future work will include:

- Adding baseline models for criteria classification (e.g., traditional ML and transformer‑based models).
- Evaluating performance across languages and criteria.
- Extending the README with modeling details, hyperparameters, and evaluation protocols.