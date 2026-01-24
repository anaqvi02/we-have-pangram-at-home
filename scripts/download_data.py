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
    Downloads a 'Pangram-Style' mix:
    1. Student Essays (PERSUADE via Artem9k) - The core target.
    2. Creative Writing (Reddit via Artem9k) - To understand creative human text.
    3. Formal Literature (Gutenberg) - To understand high-vocabulary human text.
    
    This diversity prevents the model from flagging ANY creative/formal text as AI.
    """
    print(f"--- Downloading Pangram-Style Mix (Limit: {limit}) ---")
    
    # Clean existing
    for d in [AI_DIR, HUMAN_DIR]:
        for f in d.glob("*.parquet"):
            f.unlink()
            
    try:
        # Source 1: The Essay/Creative Pile (Artem9k/DAIGT)
        print("-> Stream 1: Student Essays & Creative Writing (Artem9k)...")
        dataset_essays = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
        
        # Source 2: Formal Literature (Gutenberg)
        print("-> Stream 2: Formal Literature (Gutenberg)...")
        dataset_books = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
        book_iter = iter(dataset_books)
        
        ai_batch, human_batch = [], []
        ai_idx, human_idx = 0, 0
        ai_count, human_count = 0, 0
        
        # We aim for 50/50 split between AI/Human
        class_limit = limit 
        
        # For Human, we mix: 70% Essays/Creative, 30% Gutenberg
        human_essay_limit = int(class_limit * 0.7)
        human_book_limit = class_limit - human_essay_limit
        
        pbar = tqdm(total=limit*2, desc="Mixing Pangram Data")
        
        # 1. Process Essays & Creative (Main Loop)
        for sample in dataset_essays:
            if ai_count >= class_limit and human_count >= human_essay_limit:
                break
                
            text = sample['text']
            label = sample['generated'] # 1 = AI, 0 = Human
            
            if not text or len(text.split()) < 50:
                continue

            if label == 1:
                if ai_count < class_limit:
                    ai_batch.append({'text': text, 'source': 'essay_daigt', 'label': 1})
                    ai_count += 1
                    pbar.update(1)
            else:
                if human_count < human_essay_limit:
                    human_batch.append({'text': text, 'source': 'essay_daigt_human', 'label': 0})
                    human_count += 1
                    pbar.update(1)
            
            # Flush Check
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
                
        # 2. Process Books (Fill the rest of Human)
        print("-> Mixing in Gutenberg Books...")
        books_added = 0
        for sample in book_iter:
            if books_added >= human_book_limit:
                break
            
            text = sample.get('text', '')
            # Gutenberg texts are huge. We chunk them into essay-sized bits (e.g. 500 words)
            words = text.split()
            if len(words) < 200: continue
            
            # Take a 500 word slice to simulate an essay
            snippet = " ".join(words[:1000]) 
            
            human_batch.append({'text': snippet, 'source': 'gutenberg', 'label': 0})
            human_count += 1
            books_added += 1
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
        print(f"Pangram-Mix Complete. AI: {ai_count}, Human: {human_count} (Essays: {human_count - books_added}, Books: {books_added})")

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
