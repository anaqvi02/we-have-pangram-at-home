import pandas as pd
from datasets import load_dataset
import random

def create_formal_benchmark(output_path="tests/benchmark_formal.csv", samples_per_class=1000):
    print(f"--- Creating Formal Benchmark ({samples_per_class*2} samples) ---")
    
    data = []
    
    # 1. Human: Wikipedia (held-out)
    # We use a streaming iterator and skip the first N to ensure no overlap with training data usually
    # But simpler is to use a different split or date if available. 
    # For now, we'll just skip the first 100k samples roughly to be safe
    print("Fetching Human Data (Wikipedia)...")
    ds_human = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    count = 0
    # Skip first 10k to ensure we are far away from training data (if train used first set)
    ds_human_iter = iter(ds_human)
    for _ in range(10000): next(ds_human_iter) # Fast skip
    
    for sample in ds_human_iter:
        text = sample.get('text', '')
        # Filter for good length
        if 100 < len(text.split()) < 1000:
            data.append({'text': text, 'label': 0, 'source': 'wikipedia'})
            count += 1
            if count >= samples_per_class: break
            
    # 2. AI: Cosmopedia (held-out)
    print("Fetching AI Data (Cosmopedia)...")
    # We use a subset we maybe didn't train on, or just skip deep into it.
    ds_ai = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train", streaming=True)
    ds_ai_iter = iter(ds_ai)
    
    # Skip deep
    for _ in range(100000): next(ds_ai_iter) # Skip 100k to avoid train set overlap
    
    count = 0
    for sample in ds_ai_iter:
        text = sample.get('text', '')
        if 100 < len(text.split()) < 1000:
            data.append({'text': text, 'label': 1, 'source': 'cosmopedia'})
            count += 1
            if count >= samples_per_class: break
            
    # Shuffle
    random.shuffle(data)
    
    # Save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved benchmark to {output_path}")
    print(df['label'].value_counts())

if __name__ == "__main__":
    create_formal_benchmark()
