import pandas as pd
import re
import unicodedata

# --- 1. THE OFFICIAL "WHITE-LIST" ---
# Anything NOT in this list will be deleted
VALID_CRITERIA = [
    "Qualité des plats", "Présentation", "Service", "Ambiance / cadre", "Rapport qualité-prix", 
    "Goût", "Rapidité du service", "Propreté", "Accueil", "Qualité des boissons",
    "Qualité des produits", "Prix", "Choix / disponibilité", "Service client",
    "Disponibilité des produits", "Rapidité en caisse", "Qualité des soins",
    "Compétence du personnel", "Hygiène", "Accueil / écoute", "Délais",
    "Disponibilité médicaments", "Conseil / accompagnement", "Qualité de la réparation",
    "Fiabilité / honnêteté", "Respect des délais", "Qualité du service", "Clarté des informations",
    "Rapidité de traitement", "Expérience globale"
]

def clean_data(input_file, output_file):
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # --- 2. CLEAN TEXT COLUMN ---
    # Remove rows with no text
    df = df.dropna(subset=['text'])
    
    # Remove duplicate reviews
    df = df.drop_duplicates(subset=['text'])

    # Standardize text (Lower case, strip, NFC normalization)
    df['text'] = df['text'].apply(lambda x: unicodedata.normalize('NFC', str(x)).strip())
    
    # Remove rows where text is too short to be a real review (e.g. "ok", "...")
    df = df[df['text'].str.len() > 5]

    # --- 3. CLEAN CRITERIA COLUMN ---
    # Remove NaN criteria
    df = df.dropna(subset=['Critere'])
    
    # Convert to string and fix formatting
    def fix_critere(val):
        val = str(val).strip()
        # Remove leading dashes (e.g., "- Service" -> "Service")
        val = re.sub(r'^[-\s*]+', '', val)
        # Remove trailing pipes or noise
        val = val.split('|')[0].strip()
        return val

    df['Critere'] = df['Critere'].apply(fix_critere)

    # --- 4. STRICT FILTERING ---
    # Remove artifacts like "---", "Sentiment", "Category", etc.
    # We only keep rows that match our VALID_CRITERIA list exactly
    initial_count = len(df)
    df = df[df['Critere'].isin(VALID_CRITERIA)]
    filtered_count = initial_count - len(df)

    # --- 5. RESULTS ---
    print(f"--- CLEANING REPORT ---")
    print(f"Rows removed because they were 'junk' (---, sentiment, etc): {filtered_count}")
    print(f"Final high-quality rows remaining: {len(df)}")
    
    # Check if we have enough variety
    print("\nRows per Criteria:")
    print(df['Critere'].value_counts())

    # Save the cleaned blueprint
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Cleaned blueprint saved to: {output_file}")

if __name__ == "__main__":
    clean_data("taqyim_ultra_dataset_12k.csv", "taqyim_ultra_12k_cleaned.csv")
