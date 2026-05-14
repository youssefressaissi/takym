from unsloth import FastLanguageModel
import torch

# 1. Load Model with INCREASED sequence length (Important to prevent the 406 > 256 error)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "taqyim_model_final_10k", 
    max_seq_length = 1024, # Increased from 256 to 1024
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 2. Grouped Taxonomy (To save tokens)
TAXONOMY_STRICTE = [
    "Qualité des plats", "Présentation", "Service", "Ambiance / cadre", "Rapport qualité-prix", 
    "Goût", "Rapidité du service", "Propreté", "Accueil", "Qualité des boissons",
    "Qualité des produits", "Prix", "Choix / disponibilité", "Service client",
    "Disponibilité des produits", "Rapidité en caisse", "Qualité des soins",
    "Compétence du personnel", "Hygiène", "Accueil / écoute", "Délais",
    "Disponibilité médicaments", "Conseil / accompagnement", "Qualité de la réparation",
    "Fiabilité / honnêteté", "Respect des délais", "Qualité du service", "Clarté des informations",
    "Rapidité de traitement", "Expérience globale"
]

def predict(text):
    # Shortened instruction to save space
    instruction = "Tu es un classificateur d'avis clients. Identifie max 3 critères parmi la liste TAQYIM."
    
    # We list the valid criteria clearly but concisely
    list_str = ", ".join(TAXONOMY_STRICTE)

    # Simplified Prompt to ensure we stay under the token limit
    prompt = (
        f"<|im_start|>system\n{instruction}\nLISTE: {list_str}\n"
        f"RÈGLE: Réponds par les noms exacts, séparés par une virgule. Pas de phrases.\n<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 32, # We only need a few tokens for the labels
        use_cache = True,
        do_sample = False 
    )
    
    result = tokenizer.batch_decode(outputs)[0]
    
    # Extract only what comes after the last assistant tag
    prediction = result.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    # Cleaning the output
    return prediction

# Start Interactive Loop
print("\n" + "="*40)
print("TAQYIM: CLASSIFICATION RÉPARÉE")
print("Mode: Multi-Critères (Max 1024 tokens)")
print("Type 'exit' to stop.")
print("="*40 + "\n")

while True:
    user_input = input("Avis client: ")
    if user_input.lower() == 'exit':
        break
    if not user_input.strip():
        continue
        
    try:
        prediction = predict(user_input)
        print(f"Critère(s) identifié(s) : {prediction}\n")
    except Exception as e:
        print(f"Erreur : {e}\n")
