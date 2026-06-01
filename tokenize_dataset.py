import os
import tiktoken
import numpy as np

def tokenize_dataset():
    input_file = "global_suggestions_dataset.txt"
    print(f"Reading {input_file}...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split the dataset 90% for training and 10% for validation
    n = len(text)
    train_text = text[:int(n*0.9)]
    val_text = text[int(n*0.9):]
    
    print("Tokenizing using gpt2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    
    train_ids = enc.encode_ordinary(train_text)
    val_ids = enc.encode_ordinary(val_text)
    
    print(f"Training set has {len(train_ids):,} tokens")
    print(f"Validation set has {len(val_ids):,} tokens")
    
    # Export to .bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    print("Saving to train.bin and validation.bin...")
    train_ids.tofile("train.bin")
    val_ids.tofile("validation.bin")
    
    print("Tokenization complete! train.bin and validation.bin are ready.")

if __name__ == "__main__":
    tokenize_dataset()
