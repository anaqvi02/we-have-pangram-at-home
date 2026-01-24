import os
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

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

import argparse

def download_essay_pile(limit=1000000, batch_size=100000):
    """
    Downloads the Artem9k/ai-text-detection-pile dataset (Essays).
    Splits it into AI (generated=1) and Human (generated=0) folders
    to match the expected pipeline structure.
    """
    print(f"--- Downloading Essay Pile (Limit: {limit}) ---")
    
    # Clean existing
    for d in [AI_DIR, HUMAN_DIR]:
        for f in d.glob("*.parquet"):
            f.unlink()

    try:
        dataset = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
        
        ai_batch, human_batch = [], []
        ai_idx, human_idx = 0, 0
        ai_count, human_count = 0, 0
        total_limit = limit * 2 # Approximate total since we want limit per class ideally
        
        # Safe limit per class
        class_limit = limit 
        
        pbar = tqdm(total=total_limit, desc="Processing Essays")
        
        for sample in dataset:
            if ai_count >= class_limit and human_count >= class_limit:
                break
                
            text = sample['text']
            label = sample['generated'] # 1 = AI, 0 = Human
            
            if not text or len(text.split()) < 50:
                continue

            if label == 1:
                # AI
                if ai_count < class_limit:
                    ai_batch.append({'text': text, 'source': 'essay_pile', 'label': 1})
                    ai_count += 1
                    pbar.update(1)
            else:
                # Human
                if human_count < class_limit:
                    human_batch.append({'text': text, 'source': 'essay_pile', 'label': 0})
                    human_count += 1
                    pbar.update(1)
            
            # Flush AI
            if len(ai_batch) >= batch_size:
                df = pd.DataFrame(ai_batch)
                df.to_parquet(AI_DIR / f"wildchat_part_{ai_idx}.parquet") # Keep naming convention for compat
                del df, ai_batch[:]
                gc.collect()
                ai_idx += 1
                
            # Flush Human
            if len(human_batch) >= batch_size:
                df = pd.DataFrame(human_batch)
                df.to_parquet(HUMAN_DIR / f"c4_part_{human_idx}.parquet")
                del df, human_batch[:]
                gc.collect()
                human_idx += 1

        # Final Flushes
        if ai_batch:
            pd.DataFrame(ai_batch).to_parquet(AI_DIR / f"wildchat_part_{ai_idx}.parquet")
        if human_batch:
            pd.DataFrame(human_batch).to_parquet(HUMAN_DIR / f"c4_part_{human_idx}.parquet")
            
        pbar.close()
        print(f"Essay Download Complete. AI: {ai_count}, Human: {human_count}")

    except Exception as e:
        print(f"Failed to download Essays: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download AI and Human datasets")
    parser.add_argument("--limit", type=int, default=1000000, help="Number of samples to download")
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for parquet writing")
    parser.add_argument("--mode", type=str, default="general", choices=["general", "essays"], help="Dataset mode: 'general' (WildChat/C4) or 'essays' (Artem9k/DAIGT)")
    args = parser.parse_args()

    ensure_dirs()
    
    if args.mode == "essays":
        download_essay_pile(limit=args.limit, batch_size=args.batch_size)
    else:
        # 1. AI Corpus
        download_wildchat(limit=args.limit, batch_size=args.batch_size) 
        
        # 2. Human Corpus
        download_c4_realnewslike(limit=args.limit, batch_size=args.batch_size)
    
    print("\n--- Download Complete ---")
    print(f"AI Data: {AI_DIR}")
    print(f"Human Data: {HUMAN_DIR}")

if __name__ == "__main__":
    main()
