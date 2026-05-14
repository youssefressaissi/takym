import ollama
import pandas as pd
import time
import os
import requests
import unicodedata
import re
from tqdm import tqdm
from datasets import load_dataset

# ==========================================
# 1. TAXONOMIE COMPLÈTE (LISTE STRICTE)
# ==========================================
# Cette structure aide le modèle à choisir le bon critère selon le contexte
TAXONOMY_STRICTE = """
- Restaurant (Gastronomie): Qualité des plats, Présentation, Service, Ambiance / cadre, Rapport qualité-prix, Expérience globale
- Restaurant (Fast-food): Goût, Rapidité du service, Rapport qualité-prix, Propreté, Accueil, Expérience globale
- Café: Qualité des boissons, Ambiance, Service, Propreté, Prix, Expérience globale
- Retail (Vêtements): Qualité des produits, Prix, Choix / disponibilité, Service client, Propreté, Expérience globale
- Supermarché: Disponibilité des produits, Prix, Propreté, Rapidité en caisse, Service client, Expérience globale
- Santé (Clinique): Qualité des soins, Compétence du personnel, Hygiène, Accueil / écoute, Délais, Expérience globale
- Santé (Pharmacie): Disponibilité médicaments, Conseil / accompagnement, Accueil, Prix, Expérience globale
- Services (Garage): Qualité de la réparation, Fiabilité / honnêteté, Respect des délais, Prix, Accueil
- Services (Banque): Qualité du service, Clarté des informations, Rapidité de traitement, Accueil, Expérience globale
"""

# ==========================================
# 2. CONFIGURATION
# ==========================================
ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
client = ollama.Client(host=ollama_host)
MODEL_NAME = "qwen2.5:3b"

def clean_french_text(text):
    if not isinstance(text, str): return ""
    try: text = text.encode('latin-1').decode('utf-8')
    except: pass
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 3. LOGIQUE D'ANALYSE (STRICTE SUR LES CRITÈRES)
# ==========================================
def get_label(text):
    cleaned_text = clean_french_text(text)
    
    prompt = f"""
    Analyse l'avis client suivant : "{cleaned_text}"

    LISTE DES CRITÈRES AUTORISÉS (Choisis-en UN SEUL par avis) :
    {TAXONOMY_STRICTE}

    ÉCHELLE DE SCORE : 1 (Très mécontent) à 5 (Très satisfait).

    RÈGLES CRITIQUES :
    1. Le critère doit être recopié EXACTEMENT comme dans la liste ci-dessus.
    2. Si l'avis parle de plusieurs choses, choisis le critère le plus important.
    3. Si aucun critère ne correspond parfaitement, utilise "Expérience globale".

    FORMAT DE RÉPONSE : Categorie | Sous-categorie | Critere | Score
    RÈGLE : UNE SEULE LIGNE. PAS DE TEXTE AVANT OU APRÈS.
    """
    
    try:
        response = client.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': 'Tu es un automate de classification. Ton rôle est de mapper des avis clients sur une liste de critères fixes. Tu ne parles pas, tu sors juste les données.'},
            {'role': 'user', 'content': prompt},
        ])
        
        raw = response['message']['content'].strip()
        lines = [l for l in raw.split('\n') if '|' in l and '---' not in l and 'Categorie' not in l]
        
        if lines:
            target = lines[-1].strip('| ')
            parts = [p.strip() for p in target.split("|")]
            if len(parts) == 4: return parts
            
        return ["Services", "Autre", "Expérience globale", "3"]
    except:
        return ["Error", "Error", "Error", "3"]

# ==========================================
# 4. EXÉCUTION
# ==========================================
print("Connexion à Ollama...")
while True:
    try: requests.get(ollama_host); break
    except: time.sleep(2)

client.pull(MODEL_NAME)

print("Chargement des données...")
ds = load_dataset("Kinoux/french-customer-review-sentiment-free-2k")
df = ds['train'].to_pandas()

# Nettoyage immédiat des colonnes inutiles
df = df.drop(columns=[c for c in df.columns if c.lower() in ['label', 'sentiment', 'unnamed: 0']], errors='ignore')


final_results = []
print("Traitement en cours...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    final_results.append(get_label(row['text']))

# ==========================================
# 5. EXPORT FINAL
# ==========================================
results_df = pd.DataFrame(final_results, columns=['Categorie', 'Sous-categorie', 'Critere', 'Score'], index=df.index)
df = pd.concat([df, results_df], axis=1)

# Nettoyage final : on garde uniquement ce que vous avez demandé
df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(3).astype(int)
df = df[['text', 'Critere', 'Score']]

output_name = "taqyim_criteres_full.csv"
try:
    df.to_csv(output_name, index=False, encoding='utf-8-sig')
    print(f"SUCCESS: Fichier généré : {output_name}")
except PermissionError:
    print("ERREUR: Fermez Excel !")