import os
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
import gc
import shutil

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

# Targets
# We aim for ~1M AI samples and ~1M Human samples.
# We write in batches to keep RAM usage low (<1GB).

DATA_DIR = Config.DATA_DIR
HUMAN_DIR = DATA_DIR / "human_corpus"  # Overwrite or use new dir? Let's use same dir but clean it.
AI_DIR = DATA_DIR / "ai_corpus"

def ensure_dirs():
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)

def clean_dirs():
    print("Cleaning existing data directories...")
    if HUMAN_DIR.exists():
        shutil.rmtree(HUMAN_DIR)
    if AI_DIR.exists():
        shutil.rmtree(AI_DIR)
    ensure_dirs()

def download_formal_mix(limit=1000000, batch_size=100000):
    """
    Downloads the Formal Writing Mix:
    1. AI: HuggingFaceTB/cosmopedia (Stanford, OpenStax, KhanAcademy, AutoMath)
    2. Human: Wikipedia (20220301.en)
    """
    print(f"--- Downloading Formal Writing Mix (Limit: {limit}) ---")
    
    clean_dirs()
    
    try:
        # --- AI Sources (Cosmopedia) ---
        print("-> Stream 1 (AI): Cosmopedia (Synthetic Textbooks)...")
        # Subsets for "Formal/Academic"
        subsets = ["stanford", "openstax", "khanacademy", "auto_math_text"]
        
        ai_batch = []
        ai_idx = 0
        ai_count = 0
        
        pbar_ai = tqdm(total=limit, desc="Processing AI (Cosmopedia)")
        
        # Round robin or just sequential? Sequential is easier for streaming.
        # We'll take equal parts from each subset to reach the limit.
        limit_per_subset = limit // len(subsets)
        
        for subset in subsets:
            print(f"  -> Streaming subset: {subset}...")
            try:
                ds = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)
                subset_count = 0
                
                for sample in ds:
                    if subset_count >= limit_per_subset: break
                    if ai_count >= limit: break
                    
                    text = sample.get('text', '') or sample.get('prompt', '') + " " + sample.get('text', '')
                    if len(text.split()) < 50: continue
                    
                    ai_batch.append({'text': text, 'source': f'cosmopedia_{subset}', 'label': 1})
                    ai_count += 1
                    subset_count += 1
                    pbar_ai.update(1)
                    
                    if ai_count % 10 == 0:
                        print(f"DEBUG: Processed {ai_count} samples...", end='\r')
                    
                    if len(ai_batch) >= batch_size:
                        print(f"\nWriting batch {ai_idx}...")
                        pd.DataFrame(ai_batch).to_parquet(AI_DIR / f"part_{ai_idx}.parquet")
                        del ai_batch[:]
                        gc.collect()
                        ai_idx += 1
            except Exception as e:
                print(f"Error streaming {subset}: {e}")
                
        # Final flush AI
        if ai_batch:
            pd.DataFrame(ai_batch).to_parquet(AI_DIR / f"part_{ai_idx}.parquet")
            del ai_batch[:]
            gc.collect()
        
        pbar_ai.close()
        print(f"AI Download Complete: {ai_count} samples.")

        # --- Human Source (Wikipedia) ---
        print("-> Stream 2 (Human): Wikipedia (20220301.en)...")
        human_batch = []
        human_idx = 0
        human_count = 0
        
        pbar_human = tqdm(total=limit, desc="Processing Human (Wikipedia)")
        
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        
        for sample in ds_wiki:
            if human_count >= limit: break
            
            text = sample.get('text', '')
            if len(text.split()) < 50: continue
            
            human_batch.append({'text': text, 'source': 'wikipedia', 'label': 0})
            human_count += 1
            pbar_human.update(1)
            
            if human_count % 10 == 0:
                 print(f"DEBUG: Processed {human_count} samples...", end='\r')
            
            if len(human_batch) >= batch_size:
                pd.DataFrame(human_batch).to_parquet(HUMAN_DIR / f"part_{human_idx}.parquet")
                del human_batch[:]
                gc.collect()
                human_idx += 1
                
        # Final flush Human
        if human_batch:
            pd.DataFrame(human_batch).to_parquet(HUMAN_DIR / f"part_{human_idx}.parquet")
            del human_batch[:]
            gc.collect()
            
        pbar_human.close()
        print(f"Human Download Complete: {human_count} samples.")
        
    except Exception as e:
        print(f"Failed to download Formal Mix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ensure_dirs()
    download_formal_mix(limit=500000) # 500k each = 1M total

