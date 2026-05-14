import pandas as pd

def finalize_dataset(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    initial_count = len(df)

    # 1. Define the specific "Ghost Labels" to delete
    # These are labels that came from the AI repeating the prompt instructions
    junk_labels = [
        "- Sentiment", 
        "Criterion", 
        "Sentiment", 
        "Category", 
        "---"
    ]

    # 2. Filter the dataset
    # We keep everything that is NOT in the junk_labels list
    df = df[~df['Critere'].isin(junk_labels)]
    
    # 3. Quick repair for the "Retail: Rapidité En Caisse" (Removing the 'Retail:' prefix for consistency)
    df['Critere'] = df['Critere'].replace("Retail: Rapidité En Caisse", "Rapidité en caisse")

    final_count = len(df)
    
    print(f"--- FINAL POLISH REPORT ---")
    print(f"Rows removed (Junk/Ghost labels): {initial_count - final_count}")
    print(f"Final training rows: {final_count}")
    
    print("\nFinal Top 10 Criteria Distribution:")
    print(df['Critere'].value_counts().head(10))

    # 4. Save the actual final file for training
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n🚀 SUCCESS! Final dataset ready for training: {output_file}")

if __name__ == "__main__":
    finalize_dataset("taqyim_salvaged_data.csv", "taqyim_training_ready.csv")
