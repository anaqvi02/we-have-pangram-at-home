"""
Essay-Focused Data Download Script for AI Detection Training (V4 - Hard Cap Balancer)

This script downloads and filters data from multiple sources to create a balanced
dataset specifically for training an AI detector on English essays.

HARD CAP SYSTEM (ensures diversity):
    Each data source has a maximum cap as a percentage of the total target:
    
    HUMAN (label=0):
        - PERSUADE: 30% max (student essays grades 6-12) [Kaggle]
        - AI Essays Dataset: 30% max (human subset) [Kaggle]
        - FineWeb-Edu: 35% max (educational web content) [HuggingFace]
        - IvyPanda: 25% max (college student essays) [HuggingFace]
    
    AI (label=1):
        - AI Essays Dataset: 30% max (AI-generated subset) [Kaggle]
        - Cosmopedia/stanford: 35% max (synthetic textbooks/stories) [HuggingFace]
        - LMSYS Chat-1M: 25% max (LLM responses, filtered) [HuggingFace]
        - WildChat: 25% max (ChatGPT responses, filtered) [HuggingFace]
    
    This ensures no single source dominates and maintains dataset diversity.

Usage:
    # Download all sources with default limits (200k per class)
    python download_data.py

    # Custom target per class (caps scale proportionally)
    python download_data.py --target 500000

    # Skip Kaggle datasets (HuggingFace only, still with caps)
    python download_data.py --skip-kaggle
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

# =============================================================================
# AUTHENTICATION
# =============================================================================

# HuggingFace Auth for gated datasets (like LMSYS)
try:
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔑 Authenticating with Hugging Face...")
        login(token=hf_token)
except ImportError:
    pass

# HuggingFace datasets library
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not available. Only Kaggle parsing will work.")

# Kaggle API
try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: 'kaggle' library not installed. Kaggle downloads will not work.")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Config.DATA_DIR
HUMAN_DIR = DATA_DIR / "human_corpus"
AI_DIR = DATA_DIR / "ai_corpus"

# =============================================================================
# ESSAY DETECTION (Improved V4)
# =============================================================================
# Instead of relying on a small keyword pool, we use multiple heuristics:
# 1. Structural analysis (paragraphs, sentences)
# 2. Anti-pattern filtering (code, HTML, etc.)
# 3. Style metrics (sentence variation, formality)
# 4. Optional keyword boosting (not required)

# Anti-patterns: Definitely NOT essays
CODE_PATTERNS = [
    '```',              # Markdown code blocks
    'def ',             # Python function
    'function ',        # JavaScript
    'class ',           # OOP
    'import ',          # Code imports
    'from ',            # Python imports  
    '$ ',               # Shell prompt
    'sudo ',            # Shell commands
    '<html',            # HTML
    '<?php',            # PHP
    'SELECT ',          # SQL
    '/**',              # JSDoc
    '#include',         # C/C++
    'public static',    # Java
    '= {',              # Object literal
    '=> {',             # Arrow function
    'npm install',      # Package manager
    'pip install',      # Package manager
]

# Structural anti-patterns
STRUCTURAL_ANTI_PATTERNS = [
    (r'^[-*•]\s', 15),           # Too many bullet points (threshold)
    (r'^\d+\.\s', 20),           # Too many numbered lists
    (r'\|.*\|.*\|', 3),          # Markdown tables
    (r'^#{1,6}\s', 10),          # Too many headings (not essay-like)
]

# Template/report patterns - these indicate structured documents, not essays
TEMPLATE_ANTI_PATTERNS = [
    # Legal brief patterns
    'table of contents',
    'facts\n',           # Section header
    'issue\n',           # Section header  
    'holding\n',         # Section header
    'reasoning\n',       # Section header
    'references\n',      # Section header at end
    
    # Case citation patterns (e.g., "533 U.S. 27")
    r'\d{1,3}\s+u\.?\s*s\.?\s+\d+',
    
    # Report/brief indicators
    'executive summary',
    'abstract\n',
    'methodology\n',
    'findings\n',
    'recommendations\n',
    'appendix',
    
    # Template markers
    '[insert',
    '[your name',
    '[date]',
    'lorem ipsum',
    
    # Q&A format
    'question:',
    'answer:',
    'q:',
    'a:',
]

# Non-essay patterns: Filter out travel guides, business descriptions, etc.
NON_ESSAY_PATTERNS = [
    # Numbered sections typical of guides/how-to articles
    r'^\d+\.\s+\w+',           # "1. Introduction", "2. Packing"
    r'^\d+\)\s+\w+',           # "1) First step"
    
    # Explicit section headers
    r'^Title:\s*',
    r'^Introduction:\s*$',
    r'^Conclusion:\s*$',
    r'^Step \d+',
    
    # Travel/guide content markers
    r'\btravel guide\b',
    r'\bpacking\s+(?:essentials|list|tips)',
    r'\bvisit\s+(?:Morocco|destination|place)',
    r'\btips\s+for\s+travel',
    
    # Business description markers
    r'Co\.,?\s+Ltd\.',
    r'Corporation\s+is\s+a',
    r'founded\s+in\s+\d{4}',
    r'leading\s+(?:provider|manufacturer|company)',
    r'specializes\s+in\s+(?:the\s+)?(?:research|production|development)',
    
    # Recipe/how-to markers
    r'ingredients:',
    r'instructions:',
    r'how\s+to\s+make',
    
    # Product review markers
    r'product\s+review',
    r'pros\s+and\s+cons',
]

def has_non_essay_patterns(text):
    """Check if text contains non-essay patterns (travel guides, business descriptions, etc.)."""
    text_lower = text.lower()
    for pattern in NON_ESSAY_PATTERNS:
        if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
            return True
    return False

# Formality indicators (boost essay score)
FORMAL_INDICATORS = [
    # Transitional phrases
    'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
    'consequently', 'additionally', 'subsequently', 'meanwhile',
    # Academic language
    'according to', 'research shows', 'studies indicate', 'evidence suggests',
    'it is important', 'significant', 'analysis', 'demonstrate',
    # Essay structure
    'in conclusion', 'in summary', 'to summarize', 'firstly', 'secondly',
    'on the other hand', 'in contrast', 'for example', 'for instance',
    'as a result', 'in addition', 'in particular', 'specifically',
    # Argumentation
    'argue that', 'claim that', 'suggest that', 'believe that',
    'this essay', 'this paper', 'thesis', 'perspective', 'viewpoint',
]

def analyze_text_structure(text):
    """
    Analyze text structure to determine essay-likeness.
    Returns a dict of metrics.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Basic counts
    words = text.split()
    word_count = len(words)
    
    if word_count < 50:
        return None
    
    # Sentence analysis (rough)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    sentence_count = len(sentences)
    
    if sentence_count < 3:
        return None
    
    # Calculate average sentence length
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Sentence length variation (std dev approximation)
    sentence_lengths = [len(s.split()) for s in sentences]
    if len(sentence_lengths) > 1:
        mean_len = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - mean_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        sentence_variation = variance ** 0.5
    else:
        sentence_variation = 0
    
    # Paragraph analysis
    if '\n\n' in text:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    else:
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
    paragraph_count = len(paragraphs)
    
    # Formality score (count of formal indicators)
    text_lower = text.lower()
    formality_score = sum(1 for indicator in FORMAL_INDICATORS if indicator in text_lower)
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'avg_sentence_length': avg_sentence_length,
        'sentence_variation': sentence_variation,
        'formality_score': formality_score,
    }


def has_code_patterns(text):
    """Check if text contains code-like patterns."""
    for pattern in CODE_PATTERNS:
        if pattern in text:
            return True
    return False


def has_structural_antipatterns(text):
    """Check if text has too many structural anti-patterns (lists, tables, etc.)."""
    lines = text.split('\n')
    
    for pattern, threshold in STRUCTURAL_ANTI_PATTERNS:
        count = sum(1 for line in lines if re.match(pattern, line.strip()))
        if count > threshold:
            return True
    
    return False


def has_template_patterns(text):
    """Detect template-based documents like legal briefs, case reports, etc."""
    text_lower = text.lower()
    for pattern in TEMPLATE_ANTI_PATTERNS:
        # Check if it's a regex pattern or simple string
        if pattern.startswith(r'\d'):
            # It's a regex for case citations
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        elif pattern in text_lower:
            return True
    return False


def is_essay_like(text, min_words=200, max_words=5000, min_paragraphs=2, 
                  require_formality=False, strict=False):
    """
    Determine if text is essay-like using multiple heuristics.
    
    This is a more robust approach than simple keyword matching:
    1. Word count bounds
    2. Code/anti-pattern filtering
    3. Structural analysis (paragraphs, sentences)
    4. Sentence variation (essays have varied sentence lengths)
    5. Optional formality requirements
    
    Args:
        text: The text to analyze
        min_words: Minimum word count (default: 200)
        max_words: Maximum word count (default: 5000)
        min_paragraphs: Minimum paragraph count (default: 2)
        require_formality: If True, require at least one formal indicator
        strict: If True, apply stricter filtering (for chat data)
    
    Returns:
        bool: True if text appears essay-like
    """
    if not text or not isinstance(text, str):
        return False
    
    # Quick rejection: code patterns
    if has_code_patterns(text):
        return False
    
    # Structural anti-patterns
    if has_structural_antipatterns(text):
        return False
    
    # Template-based documents (legal briefs, case reports, etc.)
    if has_template_patterns(text):
        return False
    
    # Analyze structure
    metrics = analyze_text_structure(text)
    if metrics is None:
        return False
    
    # Word count bounds
    if not (min_words <= metrics['word_count'] <= max_words):
        return False
    
    # Paragraph requirement
    if metrics['paragraph_count'] < min_paragraphs:
        return False
    
    # Sentence structure requirements
    # Essays typically have varied sentence lengths (not all same length)
    if metrics['sentence_count'] < 5:
        return False
    
    # Average sentence length should be reasonable (not too short like chat, not too long)
    if metrics['avg_sentence_length'] < 8 or metrics['avg_sentence_length'] > 50:
        return False
    
    # Strict mode: higher requirements for chat-sourced data
    if strict:
        # Require more paragraphs
        if metrics['paragraph_count'] < 3:
            return False
        # Require some sentence variation
        if metrics['sentence_variation'] < 3:
            return False
        # Require formality indicators
        if metrics['formality_score'] < 2:
            return False
    
    # Optional formality requirement
    if require_formality and metrics['formality_score'] < 1:
        return False
    
    return True


def clean_essay_references(text):
    """Strip reference sections from essays (to prevent model from cheating on citations)."""
    if not isinstance(text, str):
        return ""
    patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*Works Cited\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*Sources\s*\n',
    ]
    cleaned = text
    for p in patterns:
        parts = re.split(p, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if len(parts) > 1:
            cleaned = parts[0].strip()
    return cleaned

# =============================================================================
# FILE UTILITIES
# =============================================================================

def ensure_dirs():
    """Create output directories if they don't exist."""
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)


def save_batch(data_batch, output_dir, prefix, batch_idx):
    """Save a batch of data to parquet."""
    if not data_batch:
        return None
    df = pd.DataFrame(data_batch)
    out_path = output_dir / f"{prefix}_part_{batch_idx}.parquet"
    df.to_parquet(out_path)
    return out_path


def clean_existing_files(output_dir, prefix):
    """Remove existing parquet files with given prefix."""
    for f in output_dir.glob(f"{prefix}_part_*.parquet"):
        f.unlink()


def download_kaggle_dataset(dataset_id, dest_dir):
    """Download and unzip a Kaggle dataset."""
    if not KAGGLE_AVAILABLE:
        print(f"❌ Cannot download {dataset_id}: 'kaggle' package not installed.")
        return None
    
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading Kaggle dataset: {dataset_id}...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_id, path=dest_dir, unzip=True)
        
        candidates = list(dest_dir.glob("*.csv")) + list(dest_dir.glob("*.parquet"))
        if candidates:
            main_file = max(candidates, key=lambda p: p.stat().st_size)
            print(f"✅ Downloaded {dataset_id} -> {main_file.name}")
            return main_file
        
        print(f"⚠️  {dataset_id} downloaded but no CSV/Parquet found")
        return None
        
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        return None

# =============================================================================
# KAGGLE DATASET PARSERS
# =============================================================================

def parse_persuade_dataset(input_path, batch_size=50000):
    """
    Parse the PERSUADE dataset - Human student essays (grades 6-12).
    Dataset: https://www.kaggle.com/datasets/nbroad/persaude-corpus-2
    """
    print(f"\n{'='*60}")
    print("PARSING: PERSUADE Dataset (Human Essays)")
    print(f"{'='*60}")
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 0
    
    df = pd.read_csv(input_path) if input_path.suffix == '.csv' else pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    
    # Detect text column
    text_col = None
    for candidate in ['full_text', 'text', 'essay_text', 'discourse_text']:
        if candidate in df.columns:
            text_col = candidate
            break
    
    if text_col is None:
        print(f"❌ Could not find text column")
        return 0
    
    # Handle discourse aggregation if needed
    if text_col == 'discourse_text' and 'essay_id_comp' in df.columns:
        print("Aggregating discourse elements by essay_id...")
        df = df.groupby('essay_id_comp')[text_col].apply(' '.join).reset_index()
        df.columns = ['essay_id', 'text']
        text_col = 'text'
    
    clean_existing_files(HUMAN_DIR, "persuade")
    
    data_batch = []
    count = 0
    batch_idx = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PERSUADE"):
        text = str(row[text_col])
        
        # Light filtering - PERSUADE is already essays
        if len(text.split()) < 100:
            continue
        
        data_batch.append({'text': text, 'source': 'persuade', 'label': 0})
        count += 1
        
        if len(data_batch) >= batch_size:
            save_batch(data_batch, HUMAN_DIR, "persuade", batch_idx)
            data_batch = []
            batch_idx += 1
    
    if data_batch:
        save_batch(data_batch, HUMAN_DIR, "persuade", batch_idx)
    
    print(f"✅ PERSUADE: {count} essays saved")
    return count


def parse_ai_essays_dataset(input_path, batch_size=50000):
    """
    Parse AI Generated Essays dataset - contains BOTH human and AI essays.
    Dataset: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
    """
    print(f"\n{'='*60}")
    print("PARSING: AI Essays Dataset (Human + AI)")
    print(f"{'='*60}")
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 0, 0
    
    df = pd.read_csv(input_path) if input_path.suffix == '.csv' else pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    
    text_col = 'text' if 'text' in df.columns else 'essay'
    label_col = 'label' if 'label' in df.columns else 'generated'
    
    print(f"Label distribution:\n{df[label_col].value_counts()}")
    
    human_df = df[df[label_col] == 0]
    ai_df = df[df[label_col] == 1]
    
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
    
    print(f"✅ AI Essays: {human_count} human, {ai_count} AI saved")
    return human_count, ai_count


def parse_persuade_dataset_capped(input_path, cap, batch_size=50000):
    """
    Parse PERSUADE dataset with a hard cap on samples.
    """
    print(f"\n{'='*60}")
    print(f"PARSING: PERSUADE Dataset (Human Essays, cap={cap:,})")
    print(f"{'='*60}")
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 0
    
    df = pd.read_csv(input_path) if input_path.suffix == '.csv' else pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    
    # Detect text column
    text_col = None
    for candidate in ['full_text', 'text', 'essay_text', 'discourse_text']:
        if candidate in df.columns:
            text_col = candidate
            break
    
    if text_col is None:
        print(f"❌ Could not find text column")
        return 0
    
    # Handle discourse aggregation if needed
    if text_col == 'discourse_text' and 'essay_id_comp' in df.columns:
        print("Aggregating discourse elements by essay_id...")
        df = df.groupby('essay_id_comp')[text_col].apply(' '.join).reset_index()
        df.columns = ['essay_id', 'text']
        text_col = 'text'
    
    clean_existing_files(HUMAN_DIR, "persuade")
    
    data_batch = []
    count = 0
    batch_idx = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PERSUADE"):
        if count >= cap:
            print(f"   → Reached cap of {cap:,} samples, stopping.")
            break
        
        text = str(row[text_col])
        
        # Light filtering - PERSUADE is already essays
        if len(text.split()) < 100:
            continue
        
        data_batch.append({'text': text, 'source': 'persuade', 'label': 0})
        count += 1
        
        if len(data_batch) >= batch_size:
            save_batch(data_batch, HUMAN_DIR, "persuade", batch_idx)
            data_batch = []
            batch_idx += 1
    
    if data_batch:
        save_batch(data_batch, HUMAN_DIR, "persuade", batch_idx)
    
    print(f"✅ PERSUADE: {count} essays saved (cap was {cap:,})")
    return count


def parse_ai_essays_dataset_capped(input_path, human_cap, ai_cap, batch_size=50000):
    """
    Parse AI Generated Essays dataset with hard caps on both human and AI samples.
    """
    print(f"\n{'='*60}")
    print(f"PARSING: AI Essays Dataset (Human cap={human_cap:,}, AI cap={ai_cap:,})")
    print(f"{'='*60}")
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 0, 0
    
    df = pd.read_csv(input_path) if input_path.suffix == '.csv' else pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    
    text_col = 'text' if 'text' in df.columns else 'essay'
    label_col = 'label' if 'label' in df.columns else 'generated'
    
    print(f"Label distribution:\n{df[label_col].value_counts()}")
    
    human_df = df[df[label_col] == 0]
    ai_df = df[df[label_col] == 1]
    
    clean_existing_files(HUMAN_DIR, "aiessays_human")
    clean_existing_files(AI_DIR, "aiessays_ai")
    
    # Process human with cap
    human_count = 0
    data_batch = []
    batch_idx = 0
    
    for _, row in tqdm(human_df.iterrows(), total=min(len(human_df), human_cap), desc="AI Essays (Human)"):
        if human_count >= human_cap:
            print(f"   → Reached human cap of {human_cap:,} samples, stopping.")
            break
        
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
    
    # Process AI with cap
    ai_count = 0
    data_batch = []
    batch_idx = 0
    
    for _, row in tqdm(ai_df.iterrows(), total=min(len(ai_df), ai_cap), desc="AI Essays (AI)"):
        if ai_count >= ai_cap:
            print(f"   → Reached AI cap of {ai_cap:,} samples, stopping.")
            break
        
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
    
    print(f"✅ AI Essays: {human_count} human (cap {human_cap:,}), {ai_count} AI (cap {ai_cap:,}) saved")
    return human_count, ai_count

# =============================================================================
# HUGGINGFACE DATASET DOWNLOADERS
# =============================================================================

def download_fineweb_edu(limit=100000, batch_size=50000):
    """Download FineWeb-Edu filtered for essay-like content."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: FineWeb-Edu (Human, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        return 0
    
    clean_existing_files(HUMAN_DIR, "fineweb")
    
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        pbar = tqdm(total=limit, desc="FineWeb-Edu")
        
        for sample in dataset:
            if count >= limit:
                break
            
            text = sample.get('text', '')
            
            if not is_essay_like(text, min_words=300, max_words=4000, min_paragraphs=3):
                continue
            
            data_batch.append({'text': text, 'source': 'fineweb_edu', 'label': 0})
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
        print(f"❌ FineWeb-Edu failed: {e}")
        return 0


def download_ivypanda(limit=50000, batch_size=25000):
    """Download IvyPanda essays - college student essays."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: IvyPanda (Human College Essays, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        return 0
    
    clean_existing_files(HUMAN_DIR, "ivypanda")
    
    try:
        ds = load_dataset("qwedsacf/ivypanda-essays", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        
        for row in tqdm(ds, desc="IvyPanda", total=limit):
            if count >= limit:
                break
            
            text_raw = row.get('text') or row.get('TEXT') or row.get('essay')
            if not text_raw:
                continue
            
            text = clean_essay_references(text_raw)
            if len(text.split()) < 100:
                continue
            
            data_batch.append({'text': text, 'source': 'ivypanda', 'label': 0})
            count += 1
            
            if len(data_batch) >= batch_size:
                save_batch(data_batch, HUMAN_DIR, "ivypanda", batch_idx)
                data_batch = []
                batch_idx += 1
        
        if data_batch:
            save_batch(data_batch, HUMAN_DIR, "ivypanda", batch_idx)
        
        print(f"✅ IvyPanda: {count} essays saved")
        return count
        
    except Exception as e:
        print(f"❌ IvyPanda failed: {e}")
        return 0


def download_cosmopedia(limit=150000, batch_size=50000):
    """Download Cosmopedia - synthetic textbooks/stories (AI-generated)."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: Cosmopedia/stanford (AI, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        return 0
    
    clean_existing_files(AI_DIR, "cosmopedia")
    
    # Use only stanford subset (children's stories removed as they're not argumentative essays)
    subset = 'stanford'
    
    total_count = 0
    batch_idx = 0
    
    print(f"\n→ Loading Cosmopedia/{subset}...")
    
    try:
        dataset = load_dataset("HuggingFaceTB/cosmopedia", subset, split="train", streaming=True)
        
        data_batch = []
        count = 0
        
        pbar = tqdm(total=limit, desc=f"Cosmopedia/{subset}")
        
        for sample in dataset:
            if count >= limit:
                break
            
            text = sample.get('text', '')
            
            if not is_essay_like(text, min_words=300, max_words=4000, min_paragraphs=3):
                continue
            
            data_batch.append({'text': text, 'source': f'cosmopedia_{subset}', 'label': 1})
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
    """Download LMSYS Chat-1M filtered for essay-like AI responses."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: LMSYS Chat-1M (AI, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
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
            
            if sample.get('language') != 'English':
                continue
            
            conv = sample.get('conversation', [])
            
            for turn in conv:
                if turn['role'] != 'assistant':
                    continue
                
                text = turn.get('content', '')
                
                # Skip non-essay content (travel guides, business descriptions, etc.)
                if has_non_essay_patterns(text):
                    continue
                
                # Strict filtering for chat data
                if not is_essay_like(text, min_words=300, max_words=3000,
                                    min_paragraphs=3, strict=True):
                    continue
                
                data_batch.append({'text': text, 'source': 'lmsys', 'label': 1})
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
        print(f"❌ LMSYS failed: {e}")
        return 0


def download_wildchat(limit=50000, batch_size=50000):
    """Download WildChat filtered for essay-like AI responses."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: WildChat (AI, limit={limit})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
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
                
                # Skip non-essay content (travel guides, business descriptions, etc.)
                if has_non_essay_patterns(text):
                    continue
                
                # Strict filtering for chat data
                if not is_essay_like(text, min_words=300, max_words=3000,
                                    min_paragraphs=3, strict=True):
                    continue
                
                data_batch.append({'text': text, 'source': 'wildchat', 'label': 1})
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
        print(f"❌ WildChat failed: {e}")
        return 0

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def download_all(target_per_class=200000, persuade_path=None, ai_essays_path=None,
                 skip_kaggle=False, auto_kaggle=True):
    """
    Download all sources with balanced targets and hard caps per source.
    
    Hard Caps (as % of target_per_class):
        HUMAN:
            - PERSUADE: 30% max
            - AI Essays (human): 30% max
            - FineWeb-Edu: 35% max
            - IvyPanda: 25% max
        
        AI:
            - AI Essays (AI): 30% max
            - Cosmopedia/stanford: 35% max
            - LMSYS: 25% max
            - WildChat: 25% max
    """
    print("=" * 70)
    print("ESSAY-FOCUSED DATA DOWNLOAD (V4 - HARD CAP BALANCER)")
    print(f"Target: ~{target_per_class:,} samples per class")
    print("=" * 70)
    
    ensure_dirs()
    
    stats = {'human': {}, 'ai': {}}
    
    # Define hard caps as percentages of target
    CAPS = {
        'human': {
            'persuade': 0.30,
            'aiessays_human': 0.30,
            'fineweb_edu': 0.35,
            'ivypanda': 0.25,
        },
        'ai': {
            'aiessays_ai': 0.30,
            'cosmopedia': 0.35,
            'lmsys': 0.25,
            'wildchat': 0.25,
        }
    }
    
    # Calculate absolute caps
    human_caps = {k: int(target_per_class * v) for k, v in CAPS['human'].items()}
    ai_caps = {k: int(target_per_class * v) for k, v in CAPS['ai'].items()}
    
    print("\n📊 Hard Caps Per Source:")
    print("   HUMAN:")
    for source, cap in human_caps.items():
        print(f"      {source:20} {cap:>8,}")
    print("   AI:")
    for source, cap in ai_caps.items():
        print(f"      {source:20} {cap:>8,}")
    
    # =========================================================================
    # KAGGLE DATASETS (with caps)
    # =========================================================================
    
    if not skip_kaggle:
        # Auto-download if paths not provided
        if auto_kaggle and KAGGLE_AVAILABLE:
            kaggle_raw_dir = DATA_DIR / "raw_kaggle"
            
            if not persuade_path:
                p_file = download_kaggle_dataset(
                    'nbroad/persaude-corpus-2',
                    kaggle_raw_dir / "persuade"
                )
                if p_file:
                    persuade_path = str(p_file)
            
            if not ai_essays_path:
                a_file = download_kaggle_dataset(
                    'shanegerami/ai-vs-human-text',
                    kaggle_raw_dir / "ai_essays"
                )
                if a_file:
                    ai_essays_path = str(a_file)
        
        # Parse PERSUADE with cap
        if persuade_path:
            # Temporarily modify the parser to respect the cap
            stats['human']['persuade'] = parse_persuade_dataset_capped(persuade_path, human_caps['persuade'])
        
        # Parse AI Essays with caps for both human and AI
        if ai_essays_path:
            h, a = parse_ai_essays_dataset_capped(ai_essays_path, human_caps['aiessays_human'], ai_caps['aiessays_ai'])
            stats['human']['aiessays_human'] = h
            stats['ai']['aiessays_ai'] = a
    
    # =========================================================================
    # HUGGINGFACE DATASETS (fill remaining with caps)
    # =========================================================================
    
    if HF_AVAILABLE:
        # Calculate how much each source can still contribute
        human_remaining = {k: max(0, human_caps[k] - stats['human'].get(k, 0)) 
                          for k in human_caps}
        ai_remaining = {k: max(0, ai_caps[k] - stats['ai'].get(k, 0)) 
                       for k in ai_caps}
        
        total_human_remaining = sum(human_remaining.values())
        total_ai_remaining = sum(ai_remaining.values())
        
        print(f"\n📈 Remaining Capacity:")
        print(f"   HUMAN: {total_human_remaining:,} total")
        for source, remaining in human_remaining.items():
            if remaining > 0:
                print(f"      {source:20} {remaining:>8,}")
        print(f"   AI: {total_ai_remaining:,} total")
        for source, remaining in ai_remaining.items():
            if remaining > 0:
                print(f"      {source:20} {remaining:>8,}")
        
        # HUMAN SOURCES - fill remaining capacity
        if total_human_remaining > 0:
            if human_remaining['fineweb_edu'] > 0:
                stats['human']['fineweb_edu'] = download_fineweb_edu(limit=human_remaining['fineweb_edu'])
            if human_remaining['ivypanda'] > 0:
                stats['human']['ivypanda'] = download_ivypanda(limit=human_remaining['ivypanda'])
        
        # AI SOURCES - fill remaining capacity
        if total_ai_remaining > 0:
            if ai_remaining['cosmopedia'] > 0:
                stats['ai']['cosmopedia'] = download_cosmopedia(limit=ai_remaining['cosmopedia'])
            if ai_remaining['lmsys'] > 0:
                stats['ai']['lmsys'] = download_lmsys(limit=ai_remaining['lmsys'])
            if ai_remaining['wildchat'] > 0:
                stats['ai']['wildchat'] = download_wildchat(limit=ai_remaining['wildchat'])
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE - SUMMARY (with Hard Caps)")
    print("=" * 70)
    
    human_total = sum(stats['human'].values())
    ai_total = sum(stats['ai'].values())
    
    print(f"\n🎯 Target: {target_per_class:,} per class")
    print(f"📊 HUMAN DATA (label=0) - Total: {human_total:,}:")
    print(f"   {'Source':20} {'Count':>8} {'%':>6} {'Cap':>10} {'Status':>10}")
    print(f"   {'-'*60}")
    for source, count in stats['human'].items():
        pct = (count / human_total * 100) if human_total > 0 else 0
        cap = human_caps.get(source, target_per_class)
        status = "✅" if count <= cap else "⚠️ OVER"
        print(f"   {source:20} {count:>8,} {pct:>5.1f}% {cap:>10,} {status:>10}")
    print(f"   {'-'*60}")
    print(f"   {'TOTAL':20} {human_total:>8,}")
    
    print(f"\n🤖 AI DATA (label=1) - Total: {ai_total:,}:")
    print(f"   {'Source':20} {'Count':>8} {'%':>6} {'Cap':>10} {'Status':>10}")
    print(f"   {'-'*60}")
    for source, count in stats['ai'].items():
        pct = (count / ai_total * 100) if ai_total > 0 else 0
        cap = ai_caps.get(source, target_per_class)
        status = "✅" if count <= cap else "⚠️ OVER"
        print(f"   {source:20} {count:>8,} {pct:>5.1f}% {cap:>10,} {status:>10}")
    print(f"   {'-'*60}")
    print(f"   {'TOTAL':20} {ai_total:>8,}")
    
    print(f"\n📁 Output: {DATA_DIR.resolve()}")
    
    if human_total > 0 and ai_total > 0:
        ratio = human_total / ai_total
        status = "✅ balanced" if 0.8 <= ratio <= 1.2 else "⚠️ imbalanced"
        print(f"\n⚖️  Class Balance: {status} (ratio: {ratio:.2f})")
    
    # Diversity check
    human_sources = len([c for c in stats['human'].values() if c > 0])
    ai_sources = len([c for c in stats['ai'].values() if c > 0])
    print(f"\n🌈 Diversity: {human_sources} human sources, {ai_sources} AI sources")
    
    return stats

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download essay-focused data for AI detection training (V4)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--target", type=int, default=200000,
                        help="Target samples per class (default: 200000)")
    parser.add_argument("--persuade", type=str, default=None,
                        help="Path to PERSUADE dataset CSV/Parquet")
    parser.add_argument("--ai-essays", type=str, default=None,
                        help="Path to AI Generated Essays dataset")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle datasets, HuggingFace only")
    parser.add_argument("--no-auto-kaggle", action="store_true",
                        help="Don't auto-download Kaggle datasets")
    
    args = parser.parse_args()
    
    download_all(
        target_per_class=args.target,
        persuade_path=args.persuade,
        ai_essays_path=args.ai_essays,
        skip_kaggle=args.skip_kaggle,
        auto_kaggle=not args.no_auto_kaggle
    )


if __name__ == "__main__":
    main()
