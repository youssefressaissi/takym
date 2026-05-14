import pandas as pd
import re

# The target criteria we want to map everything to
MAPPING_DICTIONARY = {
    "Prix": ["prix", "cher", "tarif", "argent", "coût", "facture", "frais"],
    "Service": ["service", "serveur", "personnel", "accueil", "staff"],
    "Qualité des plats": ["plat", "goût", "cuisine", "mekla", "délicieux", "manger", "boisson", "café"],
    "Hygiène": ["hygiène", "propreté", "sale", "propre", "nettoyage", "toilettes"],
    "Délais": ["délais", "attente", "rapide", "lent", "retard", "heures", "minutes"],
    "Santé": ["soin", "médicament", "clinique", "docteur", "pharmacie"],
    "WiFi": ["wifi", "connexion", "internet"],
    "Réparation": ["réparation", "garage", "moteur", "panne"]
}

def salvage_critere(val):
    val = str(val).lower()
    
    # 1. Check for separators - if the row is ONLY dashes, it's trash
    if re.match(r'^[-\s=]+$', val):
        return None
    
    # 2. Fuzzy Keyword Search
    # We look for keywords inside the messy AI string
    for official_name, keywords in MAPPING_DICTIONARY.items():
        for kw in keywords:
            if kw in val:
                return official_name
    
    # 3. If it contains "Expérience globale" or "Global", keep it as such
    if "global" in val or "expérience" in val:
        return "Expérience globale"
    
    # 4. If we still haven't found anything but it's a real word, 
    # keep the first 3 words of what the AI said as a custom criteria 
    # so we don't lose the data.
    words = val.split()
    if len(words) > 0 and len(words) < 5:
        return " ".join(words).title()
    
    return None

def run_salvage(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    initial_len = len(df)

    # Clean text first
    df = df.dropna(subset=['text'])
    
    # Repair the Criteria
    print("Repairing messy labels...")
    df['Critere'] = df['Critere'].apply(salvage_critere)
    
    # Now we only drop rows that are TRULY empty (where salvage returned None)
    df = df.dropna(subset=['Critere'])
    
    print(f"--- SALVAGE REPORT ---")
    print(f"Initial rows: {initial_len}")
    print(f"Rows kept: {len(df)}")
    print(f"Recovery Rate: {(len(df)/initial_len)*100:.1f}%")
    
    print("\nTop 15 Criteria found (including repaired ones):")
    print(df['Critere'].value_counts().head(15))

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Salvaged dataset saved to: {output_file}")

if __name__ == "__main__":
    # Point this to your RAW 12k file before it was cleaned
    run_salvage("taqyim_ultra_dataset_12k.csv", "taqyim_salvaged_data.csv")
