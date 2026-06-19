import pandas as pd
df = pd.read_csv("taqyim_improved_full.csv")

# Filter out the remaining "Expérience globale" rows to keep only 5% of them
specific = df[df['Critere'] != "Expérience globale"]
generic = df[df['Critere'] == "Expérience globale"].sample(frac=0.05)

final_df = pd.concat([specific, generic]).sample(frac=1)
final_df.to_csv("training_final.csv", index=False)
print(f"Dataset Balanced! Final size: {len(final_df)} rows.")
