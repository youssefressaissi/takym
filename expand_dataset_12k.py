import ollama
import pandas as pd
import random
import os
from tqdm import tqdm

# --- 1. SETTINGS ---
INPUT_FILE = "taqyim_improved_full.csv"
OUTPUT_FILE = "taqyim_ultra_dataset_12k.csv"
TARGET_NEW_ROWS = 10000
MODEL_NAME = "qwen2.5:3b"

LANGUAGES = [
    "French", 
    "Tunisian Derja (Arabic script)", 
    "Tunisian Arabizi (Latin script + 3,7,9)", 
    "English"
]

# --- 2. LOAD BLUEPRINT ---
if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found!")
    exit()

df_blueprint = pd.read_csv(INPUT_FILE)
# Ensure we only have Text and Criteria
df_blueprint = df_blueprint[['text', 'Critere']]

# List of unique criteria we have to follow
unique_criteria = df_blueprint['Critere'].unique().tolist()

def generate_synthetic_row(criteria, blueprint_example):
    lang = random.choice(LANGUAGES)
    
    # We show the model a 'blueprint' so it understands the style
    prompt = f"""
    Task: Write a realistic trilingual customer review.
    Industry Category: {criteria}
    Reference Example (Style guide): "{blueprint_example}"
    Language to use: {lang}

    STRICT RULES:
    - The review must be unique and natural.
    - If language is Arabizi, use 3, 7, 9 correctly (e.g. '3aslama', 'bnina', '9adech').
    - If language is English, keep it in a Tunisian context.
    - Do NOT mention the word '{criteria}' in the text.
    - Output ONLY the review text. No quotes, no explanations.
    """

    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip().replace('"', '')
    except:
        return None

# --- 3. GENERATION LOOP ---
new_data = []
rows_per_criteria = TARGET_NEW_ROWS // len(unique_criteria)

print(f"🚀 Expanding dataset from 2,000 to 12,000 rows...")
print(f"Target: ~{rows_per_criteria} new rows for each of the {len(unique_criteria)} criteria.")

for crit in unique_criteria:
    # Get all existing examples for this criteria to use as inspiration
    examples = df_blueprint[df_blueprint['Critere'] == crit]['text'].tolist()
    
    print(f"\nGenerating for Criteria: {crit}")
    for _ in tqdm(range(rows_per_criteria)):
        # Pick a random real example to act as a blueprint
        blueprint = random.choice(examples) if examples else "Standard review"
        
        text = generate_synthetic_row(crit, blueprint)
        if text and len(text) > 5:
            new_data.append({"text": text, "Critere": crit})

# --- 4. MERGE AND SAVE ---
df_synthetic = pd.DataFrame(new_data)
# Combine the original 2,000 with the new 10,000
df_final = pd.concat([df_blueprint, df_synthetic], ignore_index=True)

# Shuffle everything
df_final = df_final.sample(frac=1).reset_index(drop=True)

df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\n✅ SUCCESS!")
print(f"Original rows: {len(df_blueprint)}")
print(f"Synthetic rows added: {len(df_synthetic)}")
print(f"Total dataset size: {len(df_final)}")
print(f"Final file saved as: {OUTPUT_FILE}")
