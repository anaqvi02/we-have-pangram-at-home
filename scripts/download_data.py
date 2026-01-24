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

def download_pangram_style(limit=1000000, batch_size=100000):
    """
    Downloads a 'Pangram-Style' mix focused on Formal Writing:
    1. Student Essays & Creative (Artem9k) - 40%
    2. Formal Literature (Gutenberg) - 30%
    3. Encyclopedic/Formal (Wikipedia) - 30%
    
    This heavily upweights formal/academic writing styles as requested.
    """
    print(f"--- Downloading Pangram-Style Mix (Limit: {limit}) ---")
    
    # Clean existing
    for d in [AI_DIR, HUMAN_DIR]:
        for f in d.glob("*.parquet"):
            f.unlink()
            
    try:
        # Stream 1: Essays/Creative
        print("-> Stream 1: Student Essays (Artem9k)...")
        ds_essays = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
        
        # Stream 2: Books
        print("-> Stream 2: Formal Literature (Gutenberg)...")
        ds_books = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
        iter_books = iter(ds_books)
        
        # Stream 3: Wikipedia
        print("-> Stream 3: Encyclopedic (Wikipedia)...")
        ds_wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        iter_wiki = iter(ds_wiki)
        
        ai_batch, human_batch = [], []
        ai_idx, human_idx = 0, 0
        ai_count, human_count = 0, 0
        
        # 50/50 Split AI vs Human
        class_limit = limit 
        
        # Human Mix Ratios
        limit_essays = int(class_limit * 0.4)
        limit_books = int(class_limit * 0.3)
        limit_wiki = class_limit - limit_essays - limit_books
        
        pbar = tqdm(total=limit*2, desc="Mixing Formal Data")
        
        # 1. Main Loop (Essays + AI)
        for sample in ds_essays:
            if ai_count >= class_limit and human_count >= limit_essays:
                break
                
            text = sample['text']
            label = sample['generated'] # 1=AI, 0=Human
            
            if not text or len(text.split()) < 50: continue

            if label == 1:
                if ai_count < class_limit:
                    ai_batch.append({'text': text, 'source': 'essay_daigt_ai', 'label': 1})
                    ai_count += 1
                    pbar.update(1)
            else:
                if human_count < limit_essays:
                    human_batch.append({'text': text, 'source': 'essay_daigt_human', 'label': 0})
                    human_count += 1
                    pbar.update(1)
            
            # Flush
            if len(ai_batch) >= batch_size:
                pd.DataFrame(ai_batch).to_parquet(AI_DIR / f"part_{ai_idx}.parquet")
                del ai_batch[:]
                gc.collect()
                ai_idx += 1
            if len(human_batch) >= batch_size:
                pd.DataFrame(human_batch).to_parquet(HUMAN_DIR / f"part_{human_idx}.parquet")
                del human_batch[:]
                gc.collect()
                human_idx += 1

        # 2. Fill Books
        print("-> Mixing in Gutenberg...")
        added = 0
        for sample in iter_books:
            if added >= limit_books: break
            text = sample.get('text', '')
            words = text.split()
            if len(words) < 200: continue
            snippet = " ".join(words[:1000]) # Essay-sized chunk
            
            human_batch.append({'text': snippet, 'source': 'gutenberg', 'label': 0})
            human_count += 1
            added += 1
            pbar.update(1)
            
            if len(human_batch) >= batch_size:
                pd.DataFrame(human_batch).to_parquet(HUMAN_DIR / f"part_{human_idx}.parquet")
                del human_batch[:]
                gc.collect()
                human_idx += 1
                
        # 3. Fill Wikipedia
        print("-> Mixing in Wikipedia...")
        added = 0
        for sample in iter_wiki:
            if added >= limit_wiki: break
            text = sample.get('text', '')
            if len(text.split()) < 50: continue
            
            human_batch.append({'text': text, 'source': 'wikipedia', 'label': 0})
            human_count += 1
            added += 1
            pbar.update(1)
            
            if len(human_batch) >= batch_size:
                pd.DataFrame(human_batch).to_parquet(HUMAN_DIR / f"part_{human_idx}.parquet")
                del human_batch[:]
                gc.collect()
                human_idx += 1

        # Final Flushes
        if ai_batch:
            pd.DataFrame(ai_batch).to_parquet(AI_DIR / f"part_{ai_idx}.parquet")
        if human_batch:
            pd.DataFrame(human_batch).to_parquet(HUMAN_DIR / f"part_{human_idx}.parquet")
            
        pbar.close()
        print(f"Pangram-Mix Complete. AI: {ai_count}, Human: {human_count}")

    except Exception as e:
        print(f"Failed to download Pangram Mix: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download AI and Human datasets")
    parser.add_argument("--limit", type=int, default=1000000, help="Number of samples to download")
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for parquet writing")
    parser.add_argument("--mode", type=str, default="general", choices=["general", "essays"], help="Dataset mode: 'general' (WildChat/C4) or 'essays' (Pangram Mix: DAIGT+Gutenberg)")
    args = parser.parse_args()

    ensure_dirs()
    
    if args.mode == "essays":
        download_pangram_style(limit=args.limit, batch_size=args.batch_size)
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
