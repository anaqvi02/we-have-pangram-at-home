import os
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from src.config import Config
import gc

# Targets
# We aim for ~1M AI samples and ~1M Human samples.
# We write in batches to keep RAM usage low (<1GB).

DATA_DIR = Config.DATA_DIR
HUMAN_DIR = DATA_DIR / "human_corpus"
AI_DIR = DATA_DIR / "ai_corpus"

def ensure_dirs():
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)

def download_wildchat(output_dir=None, limit=1000000, batch_size=100000):
    print(f"--- Downloading WildChat-1M (Limit: {limit}) ---")
    
    target_dir = Path(output_dir) if output_dir else AI_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean existing parts
    for f in target_dir.glob("wildchat_part_*.parquet"):
        f.unlink()
    # Remove legacy single file
    legacy = target_dir / "wildchat.parquet"
    if legacy.exists():
        legacy.unlink()

    try:
        dataset = load_dataset("allenai/WildChat", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        pbar = tqdm(total=limit, desc="Processing WildChat")
        
        for sample in dataset:
            if count >= limit:
                break
                
            conv = sample.get('conversation', [])
            for turn in conv:
                if turn['role'] == 'assistant':
                    text = turn['content']
                    if text and len(text.split()) > 50:
                        data_batch.append({'text': text, 'source': 'wildchat', 'label': 1})
                        count += 1
                        pbar.update(1)
                        
                        # Write Batch
                        if len(data_batch) >= batch_size:
                            df = pd.DataFrame(data_batch)
                            out_path = target_dir / f"wildchat_part_{batch_idx}.parquet"
                            df.to_parquet(out_path)
                            
                            # Clear RAM
                            del df
                            del data_batch[:] # Clear list in place
                            # data_batch = [] # Alternative 
                            gc.collect()
                            
                            batch_idx += 1
                        
                        if count >= limit: 
                            break
        
        # Final Batch
        if data_batch:
            df = pd.DataFrame(data_batch)
            out_path = target_dir / f"wildchat_part_{batch_idx}.parquet"
            df.to_parquet(out_path)
            print(f"Saved final batch {batch_idx}")
            del data_batch
            gc.collect()
            
        pbar.close()
        print(f"WildChat Download Complete. Saved {batch_idx + 1} parts.")
        
    except Exception as e:
        print(f"Failed to download WildChat: {e}")

def download_c4_realnewslike(output_dir=None, limit=1000000, batch_size=100000):
    print(f"--- Downloading C4 RealNewsLike (Limit: {limit}) ---")
    
    target_dir = Path(output_dir) if output_dir else HUMAN_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean existing parts
    for f in target_dir.glob("c4_part_*.parquet"):
        f.unlink()
    # Remove legacy
    legacy = target_dir / "c4_realnewslike.parquet"
    if legacy.exists():
        legacy.unlink()
        
    try:
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True) 
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        pbar = tqdm(total=limit, desc="Processing C4")
        
        for sample in dataset:
            if count >= limit:
                break
                
            text = sample['text']
            if text and len(text.split()) > 50:
                data_batch.append({'text': text, 'source': 'c4_realnewslike', 'label': 0})
                count += 1
                pbar.update(1)
                
                if len(data_batch) >= batch_size:
                     df = pd.DataFrame(data_batch)
                     out_path = target_dir / f"c4_part_{batch_idx}.parquet"
                     df.to_parquet(out_path)
                     
                     del df
                     del data_batch[:]
                     gc.collect()
                     
                     batch_idx += 1
                
        # Final Batch
        if data_batch:
             df = pd.DataFrame(data_batch)
             out_path = target_dir / f"c4_part_{batch_idx}.parquet"
             df.to_parquet(out_path)
             print(f"Saved final batch {batch_idx}")
             del data_batch
             gc.collect()

        pbar.close()
        print(f"C4 Download Complete. Saved {batch_idx + 1} parts.")

    except Exception as e:
        print(f"Failed to download C4: {e}")

def main():
    ensure_dirs()
    
    # 1. AI Corpus
    download_wildchat(limit=1000000, batch_size=100000) 
    
    # 2. Human Corpus
    download_c4_realnewslike(limit=1000000, batch_size=100000)
    
    print("\n--- Download Complete ---")
    print(f"AI Data: {AI_DIR}")
    print(f"Human Data: {HUMAN_DIR}")

if __name__ == "__main__":
    main()
