import os
import pandas as pd
from datasets import load_dataset
from deep_translator import GoogleTranslator
from tqdm import tqdm
import time
import sys

def main():
    print("Loading English dataset 'Amod/mental_health_counseling_conversations' from HuggingFace...")
    # Load dataset
    dataset = load_dataset("Amod/mental_health_counseling_conversations")
    df = pd.DataFrame(dataset['train'])
    
    print(f"Loaded ALL {len(df)} rows. Starting translation to Sinhala...")
    print("WARNING: Translating 3500+ rows via free API will take approximately 2-3 hours.")
    print("If you get a connection error, just restart the script; it will append.")

    translator = GoogleTranslator(source='en', target='si')
    
    # If a partial file exists, load it to resume
    output_file = "sinhala_real_counseling_full.csv"
    start_index = 0
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        start_index = len(existing_df)
        print(f"Found existing file with {start_index} rows. Resuming...")
        translated_contexts = list(existing_df['Context_Sinhala'])
        translated_responses = list(existing_df['Response_Sinhala'])
    else:
        translated_contexts = []
        translated_responses = []

    error_count = 0
    # Translate Context (User Query) and Response (Therapist)
    for index, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df)-start_index, desc="Translating"):
        try:
            context_si = translator.translate(row['Context'][:4900]) # Google limit is 5000 chars
            response_si = translator.translate(row['Response'][:4900])
            
            translated_contexts.append(context_si)
            translated_responses.append(response_si)
            error_count = 0
            
            # Save every 50 rows to prevent data loss
            if len(translated_contexts) % 50 == 0:
                temp_df = df.iloc[:len(translated_contexts)].copy()
                temp_df['Context_Sinhala'] = translated_contexts
                temp_df['Response_Sinhala'] = translated_responses
                temp_df.to_csv(output_file, index=False, encoding='utf-8')
                
            time.sleep(1.0) # Slower sleep to avoid IP bans
            
        except Exception as e:
            print(f"\nError at index {index}: {e}")
            error_count += 1
            if error_count > 5:
                print("Too many consecutive errors. Google might have blocked your IP temporarily.")
                print("Progress has been saved. Please wait an hour and run again.")
                sys.exit(1)
            translated_contexts.append("[Translation Error]")
            translated_responses.append("[Translation Error]")
            time.sleep(5) # Wait longer on error

    # Final save
    final_df = df.iloc[:len(translated_contexts)].copy()
    final_df['Context_Sinhala'] = translated_contexts
    final_df['Response_Sinhala'] = translated_responses
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nTranslation complete! Saved to {output_file}")
    
    # Generate conversational text format for model training
    print("Generating conversational text format for SLM training...")
    with open("sinhala_counseling_dataset_full.txt", "w", encoding='utf-8') as f:
        for idx, row in final_df.iterrows():
            if row['Context_Sinhala'] != "[Translation Error]" and row['Response_Sinhala'] != "[Translation Error]":
                text = f"User: {row['Context_Sinhala']}\nBot: {row['Response_Sinhala']} <|endoftext|>\n"
                f.write(text)
                
    print("Saved conversational format to sinhala_counseling_dataset_full.txt")

if __name__ == "__main__":
    main()
