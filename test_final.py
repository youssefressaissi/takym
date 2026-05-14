from unsloth import FastLanguageModel
import torch

# 1. Configuration
MODEL_PATH = "taqyim_model_final" # The folder created by your training script
MAX_SEQ_LENGTH = 256

# 2. Load the trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
)

# 3. Set to inference mode (2x faster)
FastLanguageModel.for_inference(model)

def predict(text):
    # This MUST match the formatting_prompts_func used in your training script
    instruction = "Classifie cet avis client selon la taxonomie TAQYIM (Critère | Score)."
    
    prompt = (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 64,
        use_cache = True,
        do_sample = False # Greedy decoding for accuracy
    )
    
    result = tokenizer.batch_decode(outputs)[0]
    
    # Extract only the assistant's response
    return result.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()

# --- INTERACTIVE LOOP ---
print("\n" + "="*40)
print("TAQYIM TRAINED MODEL TESTER")
print("Enter 'exit' to quit")
print("="*40 + "\n")

while True:
    user_input = input("User Review: ")
    if user_input.lower() == 'exit':
        break
    
    try:
        prediction = predict(user_input)
        print(f"Prediction: {prediction}\n")
    except Exception as e:
        print(f"Error: {e}\n")
