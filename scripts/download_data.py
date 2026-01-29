"""
Essay-Focused Data Download Script for AI Detection Training

This script downloads and filters data from multiple sources to create a balanced
dataset specifically for training an AI detector on English essays (high school/university level).

Data Sources:
    HUMAN (label=0):
        - PERSUADE: Student argumentative essays (grades 6-12)
        - AI Essays Dataset: Human subset
        - Wikipedia: Formal encyclopedic writing
        - arXiv: Academic abstracts
        - FineWeb-Edu: Educational web content (filtered)
    
    AI (label=1):
        - AI Essays Dataset: AI-generated subset
        - Cosmopedia: Synthetic textbooks/stories
        - LMSYS Chat-1M: LLM responses (filtered for essays)
        - WildChat: ChatGPT responses (filtered for essays)

Usage:
    # Download all sources with default limits (balanced ~200k per class)
    python download_data.py

    # Custom limits
    python download_data.py --target 500000

    # Parse local Kaggle datasets only
    python download_data.py --kaggle-only --persuade /path/to/persuade.csv --ai-essays /path/to/ai_essays.csv
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

# HF Auth for gated datasets (like LMSYS)
try:
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔑 Authenticating with Hugging Face...")
        login(token=hf_token)
except ImportError:
    pass

# Try importing datasets, but allow script to run for Kaggle-only mode without it
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not available. Only Kaggle parsing will work.")

# Kaggle API for automated downloads
try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: 'kaggle' library not installed. Automated Kaggle downloads will not work.")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Config.DATA_DIR
HUMAN_DIR = DATA_DIR / "human_corpus"
AI_DIR = DATA_DIR / "ai_corpus"

# Essay-like content markers
ESSAY_MARKERS = [
    'in conclusion', 'therefore', 'furthermore', 'however', 'moreover',
    'in summary', 'firstly', 'secondly', 'thirdly', 'argue that',
    'evidence suggests', 'this essay', 'thesis', 'in this paper',
    'we conclude', 'it is important', 'on the other hand', 'nevertheless',
    'consequently', 'as a result', 'in addition', 'for instance',
    'according to', 'research shows', 'studies indicate'
]

# Anti-patterns (not essays)
ANTI_PATTERNS = [
    '```',           # Code blocks
    'def ',          # Python code
    'function ',     # JavaScript
    'import ',       # Code imports
    '$ ',            # Shell commands
    'sudo ',         # Shell commands
    '<html',         # HTML
    '<?php',         # PHP
    'SELECT ',       # SQL
]

def ensure_dirs():
    """Create output directories if they don't exist."""
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FILTERING UTILITIES
# =============================================================================

def is_essay_like(text, min_words=300, max_words=3000, min_paragraphs=3, min_markers=1):
    """
    Check if text has essay-like characteristics.
    
    Args:
        text: The text to analyze
        min_words: Minimum word count
        max_words: Maximum word count
        min_paragraphs: Minimum paragraph count
        min_markers: Minimum essay markers required
    
    Returns:
        bool: True if text appears essay-like
    """
    if not text or not isinstance(text, str):
        return False
    
    # Word count filter
    words = text.split()
    word_count = len(words)
    if not (min_words <= word_count <= max_words):
        return False
    
    # Check for anti-patterns (code, HTML, etc.)
    for pattern in ANTI_PATTERNS:
        if pattern in text:
            return False
    
    # Paragraph structure
    # Robust check: try \n\n first, then fallback to \n if no \n\n are found
    if '\n\n' in text:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    else:
        # Fallback for web content that might use single newlines
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 100]
        
    if len(paragraphs) < min_paragraphs:
        return False
    
    # Essay markers
    text_lower = text.lower()
    marker_count = sum(1 for m in ESSAY_MARKERS if m in text_lower)
    if marker_count < min_markers:
        return False
    
    # Check for excessive bullet points (not essay-like)
    lines = text.split('\n')
    bullet_lines = sum(1 for l in lines if l.strip().startswith(('- ', '* ', '• ', '1.', '2.', '3.')))
    if bullet_lines > 8:
        return False
    
    return True

def save_batch(data_batch, output_dir, prefix, batch_idx):
    """Save a batch of data to parquet."""
    if not data_batch:
        return
    df = pd.DataFrame(data_batch)
    out_path = output_dir / f"{prefix}_part_{batch_idx}.parquet"
    df.to_parquet(out_path)
    return out_path

def clean_existing_files(output_dir, prefix):
    """Remove existing parquet files with given prefix."""
    for f in output_dir.glob(f"{prefix}_part_*.parquet"):
        f.unlink()

def download_kaggle_dataset(dataset_id, dest_dir):
    """
    Download and unzip a Kaggle dataset to a specific directory.
    
    Args:
        dataset_id: The Kaggle dataset ID (e.g., 'julesking/tla-lab-persuade-dataset')
        dest_dir: Path where the dataset should be downloaded
    
    Returns:
        Path: The path to the downloaded file(s)
    """
    if not KAGGLE_AVAILABLE:
        print(f"❌ Cannot download {dataset_id}: 'kaggle' package not installed.")
        return None
    
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading Kaggle dataset: {dataset_id}...")
    try:
        # Authenticate (uses ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars)
        kaggle.api.authenticate()
        
        # Download files
        kaggle.api.dataset_download_files(dataset_id, path=dest_dir, unzip=True)
        
        # Find the main data file (CSV or Parquet)
        candidates = list(dest_dir.glob("*.csv")) + list(dest_dir.glob("*.parquet"))
        if candidates:
            # Prefer larger files (usually the main dataset)
            main_file = max(candidates, key=lambda p: p.stat().st_size)
            print(f"✅ Downloaded {dataset_id} -> {main_file.name}")
            return main_file
        
        print(f"⚠️  {dataset_id} downloaded but no CSV/Parquet files found in {dest_dir}")
        return None
        
    except Exception as e:
        print(f"❌ Kaggle download failed for {dataset_id}: {e}")
        print("   Make sure you have set up your Kaggle API credentials.")
        return None

# =============================================================================
# KAGGLE DATASET PARSERS (Local Files)
# =============================================================================

def parse_persuade_dataset(input_path, batch_size=100000):
    """
    Parse the PERSUADE dataset (TLA Lab) - Human student essays.
    
    Dataset: https://www.kaggle.com/datasets/julesking/tla-lab-persuade-dataset
    All essays are HUMAN-written (label=0) by students grades 6-12.
    """
    print(f"\n{'='*60}")
    print("PARSING: PERSUADE Dataset (Human Essays)")
    print(f"{'='*60}")
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 0
    
    # Load data
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        print(f"❌ Unsupported format: {input_path.suffix}")
        return 0
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Detect text column
    text_col = None
    for candidate in ['full_text', 'text', 'essay_text', 'discourse_text']:
        if candidate in df.columns:
            text_col = candidate
            break
    
    if text_col is None:
        print(f"❌ Could not find text column. Available: {list(df.columns)}")
        return 0
    
    print(f"Using text column: '{text_col}'")
    
    # Handle discourse_text aggregation
    if text_col == 'discourse_text' and 'essay_id_comp' in df.columns:
        print("Aggregating discourse elements by essay_id...")
        df = df.groupby('essay_id_comp')[text_col].apply(' '.join).reset_index()
        df.columns = ['essay_id', 'text']
        text_col = 'text'
    
    # Process
    clean_existing_files(HUMAN_DIR, "persuade")
    
    data_batch = []
    count = 0
    batch_idx = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PERSUADE"):
        text = str(row[text_col])
        
        # Lighter filtering for PERSUADE (already essays)
        if len(text.split()) < 100:
            continue
        
        data_batch.append({
            'text': text,
            'source': 'persuade',
            'label': 0
        })
        count += 1
        
        if len(data_batch) >= batch_size:
            save_batch(data_batch, HUMAN_DIR, "persuade", batch_idx)
            data_batch = []
            batch_idx += 1
            gc.collect()
    
    if data_batch:
        save_batch(data_batch, HUMAN_DIR, "persuade", batch_idx)
    
    print(f"✅ PERSUADE: {count} human essays saved")
    return count


def parse_ai_essays_dataset(input_path, batch_size=100000):
    """
    Parse the AI Generated Essays dataset (denvermagtibay).
    
    Dataset: https://www.kaggle.com/datasets/denvermagtibay/ai-generated-essays-dataset
    Contains BOTH human and AI essays, split by label.
    """
    print(f"\n{'='*60}")
    print("PARSING: AI Essays Dataset (Human + AI)")
    print(f"{'='*60}")
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 0, 0
    
    # Load data
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        print(f"❌ Unsupported format: {input_path.suffix}")
        return 0, 0
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Detect columns
    text_col = 'text' if 'text' in df.columns else 'essay'
    label_col = 'label' if 'label' in df.columns else 'generated'
    
    print(f"Text column: '{text_col}', Label column: '{label_col}'")
    print(f"Label distribution:\n{df[label_col].value_counts()}")
    
    # Split by label
    human_df = df[df[label_col] == 0]
    ai_df = df[df[label_col] == 1]
    
    # Clean existing
    clean_existing_files(HUMAN_DIR, "aiessays_human")
    clean_existing_files(AI_DIR, "aiessays_ai")
    
    # Process human
    human_count = 0
    data_batch = []
    batch_idx = 0
    
    for _, row in tqdm(human_df.iterrows(), total=len(human_df), desc="AI Essays (Human)"):
        text = str(row[text_col])
        if len(text.split()) < 100:
            continue
        data_batch.append({'text': text, 'source': 'aiessays_human', 'label': 0})
        human_count += 1
        if len(data_batch) >= batch_size:
            save_batch(data_batch, HUMAN_DIR, "aiessays_human", batch_idx)
            data_batch = []
            batch_idx += 1
    if data_batch:
        save_batch(data_batch, HUMAN_DIR, "aiessays_human", batch_idx)
    
    # Process AI
    ai_count = 0
    data_batch = []
    batch_idx = 0
    
    for _, row in tqdm(ai_df.iterrows(), total=len(ai_df), desc="AI Essays (AI)"):
        text = str(row[text_col])
        if len(text.split()) < 100:
            continue
        data_batch.append({'text': text, 'source': 'aiessays_ai', 'label': 1})
        ai_count += 1
        if len(data_batch) >= batch_size:
            save_batch(data_batch, AI_DIR, "aiessays_ai", batch_idx)
            data_batch = []
            batch_idx += 1
    if data_batch:
        save_batch(data_batch, AI_DIR, "aiessays_ai", batch_idx)
    
    print(f"✅ AI Essays: {human_count} human, {ai_count} AI essays saved")
    return human_count, ai_count

# =============================================================================
# HUGGINGFACE DATASET DOWNLOADERS
# =============================================================================

def download_wikipedia(limit=100000, batch_size=50000):
    """
    Download Wikipedia articles filtered for essay-like content.
    Great source of formal, well-structured human writing.
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: Wikipedia (Human, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        print("❌ HuggingFace datasets not available")
        return 0
    
    clean_existing_files(HUMAN_DIR, "wikipedia")
    
    try:
        # Use streaming to avoid downloading entire dataset
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        pbar = tqdm(total=limit, desc="Wikipedia")
        seen = 0
        
        for sample in dataset:
            seen += 1
            if seen % 10 == 0:
                pbar.set_postfix(inspected=seen, refresh=False)
            
            if count >= limit:
                break
            
            text = sample.get('text', '')
            
            # Filter for essay-like articles
            if not is_essay_like(text, min_words=400, max_words=5000, min_paragraphs=4, min_markers=0):
                continue
            
            # Skip list-heavy articles
            if text.count('\n*') > 10 or text.count('\n-') > 10:
                continue
            
            data_batch.append({
                'text': text,
                'source': 'wikipedia',
                'label': 0
            })
            count += 1
            pbar.update(1)
            
            if len(data_batch) >= batch_size:
                save_batch(data_batch, HUMAN_DIR, "wikipedia", batch_idx)
                data_batch = []
                batch_idx += 1
                gc.collect()
        
        if data_batch:
            save_batch(data_batch, HUMAN_DIR, "wikipedia", batch_idx)
        
        pbar.close()
        print(f"✅ Wikipedia: {count} articles saved")
        return count
        
    except Exception as e:
        print(f"❌ Wikipedia download failed: {e}")
        return 0


def download_arxiv(limit=100000, batch_size=50000):
    """
    Download arXiv abstracts - academic formal writing.
    Perfect for reducing false positives on academic essays.
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: arXiv Abstracts (Human, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        print("❌ HuggingFace datasets not available")
        return 0
    
    clean_existing_files(HUMAN_DIR, "arxiv")
    
    try:
        # Use the corrected arxiv dataset path
        dataset = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        pbar = tqdm(total=limit, desc="arXiv")
        seen = 0
        
        for sample in dataset:
            seen += 1
            if seen % 10 == 0:
                pbar.set_postfix(inspected=seen, refresh=False)
                
            if count >= limit:
                break
            
            # In ccdv/arxiv-summarization, the key is 'article'
            abstract = sample.get('article', '')
            
            # Abstracts are shorter, but densely written
            if len(abstract.split()) < 100:
                continue
            
            # Skip if too much math notation
            if abstract.count('$') > 10:
                continue
            
            data_batch.append({
                'text': abstract,
                'source': 'arxiv',
                'label': 0
            })
            count += 1
            pbar.update(1)
            
            if len(data_batch) >= batch_size:
                save_batch(data_batch, HUMAN_DIR, "arxiv", batch_idx)
                data_batch = []
                batch_idx += 1
                gc.collect()
        
        if data_batch:
            save_batch(data_batch, HUMAN_DIR, "arxiv", batch_idx)
        
        pbar.close()
        print(f"✅ arXiv: {count} abstracts saved")
        return count
        
    except Exception as e:
        print(f"❌ arXiv download failed: {e}")
        return 0


def download_fineweb_edu(limit=100000, batch_size=50000):
    """
    Download FineWeb-Edu filtered for essay-like content.
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: FineWeb-Edu (Human, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        print("❌ HuggingFace datasets not available")
        return 0
    
    clean_existing_files(HUMAN_DIR, "fineweb")
    
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        pbar = tqdm(total=limit, desc="FineWeb-Edu")
        seen = 0
        
        for sample in dataset:
            seen += 1
            if seen % 100 == 0: # Higher frequency for web content
                pbar.set_postfix(inspected=seen, refresh=False)
                
            if count >= limit:
                break
            
            text = sample.get('text', '')
            
            # Relaxed essay filtering for web content
            if not is_essay_like(text, min_words=300, max_words=3000, min_paragraphs=3, min_markers=1):
                continue
            
            data_batch.append({
                'text': text,
                'source': 'fineweb_edu',
                'label': 0
            })
            count += 1
            pbar.update(1)
            
            if len(data_batch) >= batch_size:
                save_batch(data_batch, HUMAN_DIR, "fineweb", batch_idx)
                data_batch = []
                batch_idx += 1
                gc.collect()
        
        if data_batch:
            save_batch(data_batch, HUMAN_DIR, "fineweb", batch_idx)
        
        pbar.close()
        print(f"✅ FineWeb-Edu: {count} samples saved")
        return count
        
    except Exception as e:
        print(f"❌ FineWeb-Edu download failed: {e}")
        return 0


def download_cosmopedia(limit=200000, batch_size=50000):
    """
    Download Cosmopedia - high quality synthetic textbooks/stories.
    Focus on 'stories' and 'stanford' subsets which are most essay-like.
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: Cosmopedia (AI, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        print("❌ HuggingFace datasets not available")
        return 0
    
    clean_existing_files(AI_DIR, "cosmopedia")
    
    # Subsets to use (most essay-like)
    subsets = ['stories', 'stanford', 'web_samples_v2']
    per_subset_limit = limit // len(subsets)
    
    total_count = 0
    batch_idx = 0
    
    for subset in subsets:
        print(f"\n→ Loading Cosmopedia/{subset}...")
        
        try:
            dataset = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)
            
            data_batch = []
            count = 0
            
            pbar = tqdm(total=per_subset_limit, desc=f"Cosmopedia/{subset}")
            
            for sample in dataset:
                if count >= per_subset_limit:
                    break
                
                text = sample.get('text', '')
                
                # Filter for essay-like content
                if not is_essay_like(text, min_words=300, max_words=4000, min_paragraphs=3, min_markers=1):
                    continue
                
                data_batch.append({
                    'text': text,
                    'source': f'cosmopedia_{subset}',
                    'label': 1
                })
                count += 1
                pbar.update(1)
                
                if len(data_batch) >= batch_size:
                    save_batch(data_batch, AI_DIR, "cosmopedia", batch_idx)
                    data_batch = []
                    batch_idx += 1
                    gc.collect()
            
            if data_batch:
                save_batch(data_batch, AI_DIR, "cosmopedia", batch_idx)
                data_batch = []
                batch_idx += 1
            
            pbar.close()
            total_count += count
            print(f"  ✓ {subset}: {count} samples")
            
        except Exception as e:
            print(f"  ✗ {subset} failed: {e}")
    
    print(f"✅ Cosmopedia total: {total_count} samples saved")
    return total_count


def download_lmsys(limit=50000, batch_size=50000):
    """
    Download LMSYS Chat-1M filtered for essay-like AI responses.
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: LMSYS Chat-1M (AI, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        print("❌ HuggingFace datasets not available")
        return 0
    
    clean_existing_files(AI_DIR, "lmsys")
    
    try:
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        pbar = tqdm(total=limit, desc="LMSYS")
        
        for sample in dataset:
            if count >= limit:
                break
            
            # English only
            if sample.get('language') != 'English':
                continue
            
            conv = sample.get('conversation', [])
            
            for turn in conv:
                if turn['role'] != 'assistant':
                    continue
                
                text = turn.get('content', '')
                
                # Strict essay filtering for chat data
                if not is_essay_like(text, min_words=300, max_words=3000, min_paragraphs=3, min_markers=2):
                    continue
                
                data_batch.append({
                    'text': text,
                    'source': 'lmsys',
                    'label': 1
                })
                count += 1
                pbar.update(1)
                
                if count >= limit:
                    break
            
            if len(data_batch) >= batch_size:
                save_batch(data_batch, AI_DIR, "lmsys", batch_idx)
                data_batch = []
                batch_idx += 1
                gc.collect()
        
        if data_batch:
            save_batch(data_batch, AI_DIR, "lmsys", batch_idx)
        
        pbar.close()
        print(f"✅ LMSYS: {count} essay-like responses saved")
        return count
        
    except Exception as e:
        print(f"❌ LMSYS download failed: {e}")
        return 0


def download_wildchat(limit=50000, batch_size=50000):
    """
    Download WildChat filtered for essay-like AI responses.
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: WildChat (AI, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        print("❌ HuggingFace datasets not available")
        return 0
    
    clean_existing_files(AI_DIR, "wildchat")
    
    try:
        dataset = load_dataset("allenai/WildChat", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        pbar = tqdm(total=limit, desc="WildChat")
        
        for sample in dataset:
            if count >= limit:
                break
            
            conv = sample.get('conversation', [])
            
            for turn in conv:
                if turn['role'] != 'assistant':
                    continue
                
                text = turn.get('content', '')
                
                # Strict essay filtering
                if not is_essay_like(text, min_words=300, max_words=3000, min_paragraphs=3, min_markers=2):
                    continue
                
                data_batch.append({
                    'text': text,
                    'source': 'wildchat',
                    'label': 1
                })
                count += 1
                pbar.update(1)
                
                if count >= limit:
                    break
            
            if len(data_batch) >= batch_size:
                save_batch(data_batch, AI_DIR, "wildchat", batch_idx)
                data_batch = []
                batch_idx += 1
                gc.collect()
        
        if data_batch:
            save_batch(data_batch, AI_DIR, "wildchat", batch_idx)
        
        pbar.close()
        print(f"✅ WildChat: {count} essay-like responses saved")
        return count
        
    except Exception as e:
        print(f"❌ WildChat download failed: {e}")
        return 0

# =============================================================================
# MAIN DOWNLOAD ORCHESTRATOR
# =============================================================================

def download_all(target_per_class=200000, persuade_path=None, ai_essays_path=None):
    """
    Download all sources with balanced targets.
    
    Target distribution (adjustable):
        HUMAN (~target_per_class total):
            - PERSUADE: ~25k (if provided)
            - AI Essays (human): variable (if provided)
            - Wikipedia: ~50k
            - arXiv: ~50k  
            - FineWeb-Edu: ~75k
        
        AI (~target_per_class total):
            - AI Essays (AI): variable (if provided)
            - Cosmopedia: ~100k
            - LMSYS: ~50k
            - WildChat: ~50k
    """
    print("=" * 70)
    print("ESSAY-FOCUSED DATA DOWNLOAD")
    print(f"Target: ~{target_per_class:,} samples per class (balanced)")
    print("=" * 70)
    
    ensure_dirs()
    
    stats = {
        'human': {},
        'ai': {}
    }
    
    # =========================================================================
    # KAGGLE DATASETS (Local)
    # =========================================================================
    
    if persuade_path:
        stats['human']['persuade'] = parse_persuade_dataset(persuade_path)
    
    if ai_essays_path:
        h, a = parse_ai_essays_dataset(ai_essays_path)
        stats['human']['ai_essays_human'] = h
        stats['ai']['ai_essays_ai'] = a
    
    # =========================================================================
    # HUGGINGFACE DATASETS
    # =========================================================================
    
    if HF_AVAILABLE:
        # Calculate remaining targets based on what we got from Kaggle
        human_from_kaggle = sum(stats['human'].values())
        ai_from_kaggle = sum(stats['ai'].values())
        
        human_remaining = max(0, target_per_class - human_from_kaggle)
        ai_remaining = max(0, target_per_class - ai_from_kaggle)
        
        print(f"\nKaggle data: {human_from_kaggle} human, {ai_from_kaggle} AI")
        print(f"Remaining target: {human_remaining} human, {ai_remaining} AI")
        
        # HUMAN SOURCES
        if human_remaining > 0:
            # Distribute evenly across sources
            wiki_limit = min(human_remaining // 3, 100000)
            arxiv_limit = min(human_remaining // 3, 100000)
            fineweb_limit = human_remaining - wiki_limit - arxiv_limit
            
            stats['human']['wikipedia'] = download_wikipedia(limit=wiki_limit)
            stats['human']['arxiv'] = download_arxiv(limit=arxiv_limit)
            stats['human']['fineweb_edu'] = download_fineweb_edu(limit=fineweb_limit)
        
        # AI SOURCES
        if ai_remaining > 0:
            # Cosmopedia is highest quality, give it more weight
            cosmo_limit = min(ai_remaining // 2, 150000)
            lmsys_limit = min(ai_remaining // 4, 50000)
            wildchat_limit = ai_remaining - cosmo_limit - lmsys_limit
            
            stats['ai']['cosmopedia'] = download_cosmopedia(limit=cosmo_limit)
            stats['ai']['lmsys'] = download_lmsys(limit=lmsys_limit)
            stats['ai']['wildchat'] = download_wildchat(limit=wildchat_limit)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE - SUMMARY")
    print("=" * 70)
    
    human_total = sum(stats['human'].values())
    ai_total = sum(stats['ai'].values())
    
    print("\n📊 HUMAN DATA (label=0):")
    for source, count in stats['human'].items():
        pct = (count / human_total * 100) if human_total > 0 else 0
        print(f"   {source:20} {count:>8,} ({pct:5.1f}%)")
    print(f"   {'TOTAL':20} {human_total:>8,}")
    
    print("\n🤖 AI DATA (label=1):")
    for source, count in stats['ai'].items():
        pct = (count / ai_total * 100) if ai_total > 0 else 0
        print(f"   {source:20} {count:>8,} ({pct:5.1f}%)")
    print(f"   {'TOTAL':20} {ai_total:>8,}")
    
    print("\n📁 Output Directories:")
    print(f"   Human: {HUMAN_DIR.resolve()}")
    print(f"   AI:    {AI_DIR.resolve()}")
    
    # Balance check
    if human_total > 0 and ai_total > 0:
        ratio = human_total / ai_total
        if 0.8 <= ratio <= 1.2:
            print(f"\n✅ Dataset is well-balanced (ratio: {ratio:.2f})")
        else:
            print(f"\n⚠️  Dataset is imbalanced (ratio: {ratio:.2f})")
    
    return stats

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download essay-focused data for AI detection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all sources (default ~200k per class)
    python download_data.py

    # Higher target
    python download_data.py --target 500000

    # Include Kaggle datasets
    python download_data.py --persuade /path/to/persuade.csv --ai-essays /path/to/ai_essays.csv

    # Kaggle only (no HuggingFace downloads)
    python download_data.py --kaggle-only --persuade persuade.csv
        """
    )
    
    parser.add_argument(
        "--target", type=int, default=200000,
        help="Target samples per class (default: 200000)"
    )
    parser.add_argument(
        "--persuade", type=str, default=None,
        help="Path to PERSUADE dataset CSV/Parquet"
    )
    parser.add_argument(
        "--ai-essays", type=str, default=None,
        help="Path to AI Generated Essays dataset CSV/Parquet"
    )
    parser.add_argument(
        "--kaggle-only", action="store_true",
        help="Only parse Kaggle datasets, skip HuggingFace downloads"
    )
    
    parser.add_argument(
        "--auto-kaggle", action="store_true", default=True,
        help="Attempt to automatically download Kaggle datasets if not provided (default: True)"
    )
    
    args = parser.parse_args()
    
    # Auto-download Kaggle if requested and paths not provided
    persuade_path = args.persuade
    ai_essays_path = args.ai_essays
    
    if args.auto_kaggle:
        kaggle_raw_dir = Config.DATA_DIR / "raw_kaggle"
        
        if not persuade_path:
            p_file = download_kaggle_dataset('julesking/tla-lab-persuade-dataset', kaggle_raw_dir / "persuade")
            if p_file: persuade_path = str(p_file)
            
        if not ai_essays_path:
            a_file = download_kaggle_dataset('denvermagtibay/ai-generated-essays-dataset', kaggle_raw_dir / "ai_essays")
            if a_file: ai_essays_path = str(a_file)

    if args.kaggle_only:
        if not persuade_path and not ai_essays_path:
            print("Error: --kaggle-only requires at least one of --persuade or --ai-essays (or working --auto-kaggle)")
            return
        
        ensure_dirs()
        stats = {'human': {}, 'ai': {}}
        
        if persuade_path:
            stats['human']['persuade'] = parse_persuade_dataset(persuade_path)
        
        if ai_essays_path:
            h, a = parse_ai_essays_dataset(ai_essays_path)
            stats['human']['ai_essays_human'] = h
            stats['ai']['ai_essays_ai'] = a
        
        print("\n✅ Kaggle parsing complete!")
        print(f"Human: {sum(stats['human'].values())}")
        print(f"AI: {sum(stats['ai'].values())}")
    else:
        download_all(
            target_per_class=args.target,
            persuade_path=persuade_path,
            ai_essays_path=ai_essays_path
        )


if __name__ == "__main__":
    main()
