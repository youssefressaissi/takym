import ollama
import pandas as pd
from tqdm import tqdm
import time

# --- CONFIGURATION ---
INPUT_FILE = "taqyim_criteres_stricts.csv"  # Your current bad dataset
OUTPUT_FILE = "taqyim_improved_full.csv"     # The new high-quality dataset
MODEL_NAME = "qwen2.5:3b"

# Force the AI to pick these if these keywords appear
KEYWORD_MAP = {
    "Prix": ["cher", "prix", "argent", "payé", "coût", "facture", "budget", "euros", "dt", "millimes"],
    "Délais": ["attendu", "long", "retard", "heures", "minutes", "lent", "rapide", "délais", "attente"],
    "Propreté": ["sale", "propre", "hygiène", "odeur", "poussière", "toilettes", "ndhifa", "moussekha"],
    "Qualité des plats": ["goût", "délicieux", "froid", "cuisson", "saveur", "manger", "pizza", "burger", "bnina", "mekla"]
}

def get_strict_label(text):
    text_clean = str(text).lower()
    
    # 1. Manual Nudge Logic
    hint = ""
    for criteria, keywords in KEYWORD_MAP.items():
        if any(word in text_clean for word in keywords):
            hint = f"NOTE: The text mentions elements related to '{criteria}'. Do NOT use 'Expérience globale'."
            break

    # 2. Strict Prompt
    prompt = f"""
    [INST] Task: Data Annotation.
    Text to analyze: "{text}"
    {hint}

    RULES:
    - CATEGORIZATION: Choose the most specific criterion (e.g., Prix, Service, Hygiène).
    - FORBIDDEN: Do not use "Expérience globale" if the review is longer than 3 words.
    - SENTIMENT: 1-2 (Negative), 3 (Neutral), 4-5 (Positive).
    
    TAXONOMY:
    - Gastronomie: Qualité des plats, Service, Ambiance, Prix
    - Santé: Qualité des soins, Compétence, Hygiène, Délais
    - Retail: Qualité des produits, Prix, Rapidité en caisse
    - Services: Réparation, Fiabilité, Délais, Accueil

    FORMAT: Criterion | Score
    Example: Qualité des plats | 2 [/INST]
    """

    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        full_res = response['message']['content'].strip()
        
        # Defensive parsing: find the line with the pipe
        lines = [l for l in full_res.split('\n') if '|' in l]
        if lines:
            return lines[-1].strip()
        return "Expérience globale | 3"
    except Exception:
        return "Expérience globale | 3"

# --- EXECUTION ---
print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# If the file has 'Critere' or 'Score', we drop them to start fresh
if 'Critere' in df.columns: df = df.drop(columns=['Critere'])
if 'Score' in df.columns: df = df.drop(columns=['Score'])

print(f"Starting relabeling of {len(df)} rows. Expected time: {len(df)*1.2/60:.1f} minutes.")
tqdm.pandas()

# Run the AI
df['temp_label'] = df['text'].progress_apply(get_strict_label)

# Split results into two columns
# n=1 ensures we only split on the first pipe
df[['Critere', 'Score']] = df['temp_label'].str.split('|', expand=True, n=1)

# Clean up formatting artifacts
df['Critere'] = df['Critere'].str.strip()
df['Score'] = df['Score'].str.strip()

# Remove the temporary column and save
df = df.drop(columns=['temp_label'])
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\nSUCCESS! Labeled data saved to: {OUTPUT_FILE}")
