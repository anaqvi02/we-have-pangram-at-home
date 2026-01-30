"""
V3 Data Download Script: The "Academic Gap" Fix
- Drops 'ArXiv' (bad formatting artifacts)
- Drops 'Wikipedia' (encyclopedia style bias)
- Adds 'IvyPanda' (College Student Essays)
- Boosts 'FineWeb-Edu' (High-quality Academic Web)
"""

import os
import sys
import gc
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

# Try importing dependencies
try:
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token: login(token=hf_token)
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not available.")

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Config.DATA_DIR
HUMAN_DIR = DATA_DIR / "human_corpus"
AI_DIR = DATA_DIR / "ai_corpus"

# 1. NEW: IvyPanda (The College Bridge)
# Strips "References" to prevent cheating
def clean_essay_references(text):
    if not isinstance(text, str): return ""
    patterns = [r'\n\s*References\s*\n', r'\n\s*Works Cited\s*\n', r'\n\s*Bibliography\s*\n']
    cleaned = text
    for p in patterns:
        parts = re.split(p, cleaned, flags=re.IGNORECASE|re.DOTALL)
        if len(parts) > 1: cleaned = parts[0].strip()
    return cleaned

def ensure_dirs():
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)

def save_batch(data_batch, output_dir, prefix, batch_idx):
    if not data_batch: return
    df = pd.DataFrame(data_batch)
    out_path = output_dir / f"{prefix}_part_{batch_idx}.parquet"
    df.to_parquet(out_path)

def clean_existing_files(output_dir, prefix):
    for f in output_dir.glob(f"{prefix}_part_*.parquet"):
        f.unlink()

# =============================================================================
# DOWNLOADERS
# =============================================================================

def download_ivypanda(limit=50000):
    """
    Downloads IvyPanda essays to fill the 'College Gap'.
    Crucial for fixing false positives on academic writing.
    """
    print(f"\n{'='*60}\nDOWNLOADING: IvyPanda (Human College Essays)\n{'='*60}")
    if not HF_AVAILABLE: return 0
    
    clean_existing_files(HUMAN_DIR, "ivypanda")
    
    try:
        ds = load_dataset("qwedsacf/ivypanda-essays", split="train", streaming=True)
        batch = []
        count = 0
        batch_idx = 0
        
        for row in tqdm(ds, desc="IvyPanda"):
            if count >= limit: break
            
            # Smart text detection + Cleaning
            text_raw = row.get('text') or row.get('TEXT') or row.get('essay')
            if not text_raw: continue
            
            text = clean_essay_references(text_raw)
            if len(text.split()) < 100: continue
            
            batch.append({'text': text, 'source': 'ivypanda', 'label': 0})
            count += 1
            
            if len(batch) >= 10000:
                save_batch(batch, HUMAN_DIR, "ivypanda", batch_idx)
                batch = []
                batch_idx += 1
        
        if batch: save_batch(batch, HUMAN_DIR, "ivypanda", batch_idx)
        print(f"✅ IvyPanda: {count} essays saved")
        return count
    except Exception as e:
        print(f"❌ IvyPanda failed: {e}")
        return 0

def download_fineweb_edu(limit=150000):
    """
    Downloads FineWeb-Edu. 
    We INCREASED the limit here to replace ArXiv/Wiki.
    """
    print(f"\n{'='*60}\nDOWNLOADING: FineWeb-Edu (Academic Web)\n{'='*60}")
    if not HF_AVAILABLE: return 0
    
    clean_existing_files(HUMAN_DIR, "fineweb")
    
    try:
        # Use sample-10BT for speed, it's high quality enough
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        batch = []
        count = 0
        batch_idx = 0
        
        for row in tqdm(ds, desc="FineWeb"):
            if count >= limit: break
            
            text = row.get('text', '')
            # Filter for reasonably long, essay-like content
            if len(text.split()) < 300: continue
            
            # Basic marker check to ensure it's not just a chat log
            if "copyright" in text.lower()[:50]: continue 
            
            batch.append({'text': text, 'source': 'fineweb', 'label': 0})
            count += 1
            
            if len(batch) >= 25000:
                save_batch(batch, HUMAN_DIR, "fineweb", batch_idx)
                batch = []
                batch_idx += 1
                
        if batch: save_batch(batch, HUMAN_DIR, "fineweb", batch_idx)
        print(f"✅ FineWeb: {count} samples saved")
        return count
    except Exception as e:
        print(f"❌ FineWeb failed: {e}")
        return 0

# (Re-use your existing AI downloaders for Cosmopedia/LMSYS)
def download_cosmopedia(limit=100000):
    print(f"\n{'='*60}\nDOWNLOADING: Cosmopedia (AI Textbooks)\n{'='*60}")
    clean_existing_files(AI_DIR, "cosmopedia")
    try:
        # 'stanford' and 'stories' are the best subsets
        ds = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train", streaming=True)
        batch = []
        count = 0
        batch_idx = 0
        for row in tqdm(ds, desc="Cosmopedia"):
            if count >= limit: break
            batch.append({'text': row['text'], 'source': 'cosmopedia', 'label': 1})
            count += 1
            if len(batch) >= 25000:
                save_batch(batch, AI_DIR, "cosmopedia", batch_idx)
                batch = []
                batch_idx += 1
        if batch: save_batch(batch, AI_DIR, "cosmopedia", batch_idx)
        return count
    except: return 0

def download_lmsys(limit=50000):
    print(f"\n{'='*60}\nDOWNLOADING: LMSYS (AI Chat)\n{'='*60}")
    clean_existing_files(AI_DIR, "lmsys")
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        batch = []
        count = 0
        idx = 0
        for row in tqdm(ds, desc="LMSYS"):
            if count >= limit: break
            if row['language'] != 'English': continue
            # Find assistant response
            for turn in row['conversation']:
                if turn['role'] == 'assistant' and len(turn['content'].split()) > 200:
                    batch.append({'text': turn['content'], 'source': 'lmsys', 'label': 1})
                    count += 1
                    break
            if len(batch) >= 25000:
                save_batch(batch, AI_DIR, "lmsys", idx)
                batch = []
                idx += 1
        if batch: save_batch(batch, AI_DIR, "lmsys", idx)
        return count
    except: return 0

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ensure_dirs()
    
    # 1. Parse Local Kaggle Files (Persuade) - KEEP THIS
    # (Assuming you run this with --persuade argument or have the file)
    # We skip re-implementing the Kaggle parser here for brevity, 
    # but you should keep the 'parse_persuade_dataset' from your old script.
    
    # 2. Download New Data
    print("--- 🚀 STARTING V3 DOWNLOAD ---")
    
    # HUMAN: Persuade (Local) + IvyPanda (College) + FineWeb (Academic)
    # Target: ~200k Human
    download_ivypanda(limit=40000)      # The Bridge (College)
    download_fineweb_edu(limit=160000)  # The Expert (replaces ArXiv/Wiki)
    
    # AI: Cosmopedia (Textbook) + LMSYS (Chat)
    # Target: ~200k AI
    download_cosmopedia(limit=150000)   # The "Smart" AI
    download_lmsys(limit=50000)         # The "Chatty" AI
    
    print("\n✅ Download Complete. Ready for V3 Training.")
