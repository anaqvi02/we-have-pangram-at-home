"""
Essay-Focused Data Download Script for AI Detection Training (V5 - DYNAMIC BALANCER)

This script downloads and filters data from multiple sources to create a balanced
dataset specifically for training an AI detector on English essays.

DYNAMIC REBALANCING SYSTEM:
    Unlike V4's hardcoded caps, V5 uses a two-phase approach:
    
    Phase 1 - ESTIMATION:
        Download a small sample (~5k) from each source to measure actual acceptance rates.
        This accounts for filtering that rejects non-essay content.
    
    Phase 2 - ALLOCATION:
        Based on estimated yields, dynamically allocate quotas to each source.
        Sources with low acceptance rates get smaller quotas (not worth the time).
        Sources with high acceptance rates can absorb leftover quota from low-yield sources.
        
    CONSTRAINTS:
        - No single source can exceed 50% of total (maintains diversity)
        - Minimum allocation of 5% per source (ensures representation)
        - Priority ordering determines redistribution (higher priority = first to receive extra)

Usage:
    # Download with dynamic balancing
    python download_data_v5.py
    
    # Custom target per class
    python download_data_v5.py --target 600000
    
    # Skip estimation phase (use historical yields if available)
    python download_data_v5.py --skip-estimation
"""

import os
import sys
import gc
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

# =============================================================================
# AUTHENTICATION (same as V4)
# =============================================================================

HF_AUTHENTICATED = False
KAGGLE_AUTHENTICATED = False

try:
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔑 Authenticating with Hugging Face...")
        login(token=hf_token)
        HF_AUTHENTICATED = True
    else:
        print("⚠️  WARNING: HF_TOKEN not set. Gated datasets (LMSYS) will fail to download.")
except ImportError:
    print("⚠️  WARNING: huggingface_hub not installed.")

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ ERROR: 'datasets' library not available.")

try:
    import kaggle
    KAGGLE_AVAILABLE = True
    try:
        kaggle.api.authenticate()
        KAGGLE_AUTHENTICATED = True
    except Exception as e:
        print(f"⚠️  WARNING: Kaggle authentication failed: {e}")
except ImportError:
    KAGGLE_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Config.DATA_DIR
HUMAN_DIR = DATA_DIR / "human_corpus"
AI_DIR = DATA_DIR / "ai_corpus"
YIELD_CACHE_FILE = DATA_DIR / ".yield_estimates.json"

# =============================================================================
# DATA SOURCE DEFINITIONS
# =============================================================================

@dataclass
class DataSource:
    """Defines a data source with its properties."""
    name: str
    label: int  # 0 = human, 1 = AI
    priority: int  # Higher = preferred for reallocation
    min_share: float = 0.05  # Minimum share of class total (5%)
    max_share: float = 0.50  # Maximum share of class total (50%)
    is_kaggle: bool = False
    estimated_yield: Optional[float] = None  # Acceptance rate (0-1)
    
    def __post_init__(self):
        if self.estimated_yield is None:
            # Default optimistic estimates (will be updated in Phase 1)
            self.estimated_yield = 0.5


# Define all sources with priorities (higher = more preferred for reallocation)
HUMAN_SOURCES = [
    DataSource("aiessays_human", label=0, priority=3, is_kaggle=True, max_share=0.45),
    DataSource("fineweb_edu", label=0, priority=2, max_share=0.55),
    DataSource("ivypanda", label=0, priority=1, max_share=0.40),
]

AI_SOURCES = [
    DataSource("aiessays_ai", label=1, priority=4, is_kaggle=True, max_share=0.40),
    DataSource("cosmopedia_web_samples_v2", label=1, priority=3, max_share=0.50),
    DataSource("cosmopedia_stanford", label=1, priority=2, max_share=0.35),
    DataSource("lmsys", label=1, priority=1, max_share=0.30),  # Lowest priority (slow, strict filtering)
]

# =============================================================================
# ESSAY DETECTION (imported from V4 - same filters)
# =============================================================================

CODE_PATTERNS = [
    '```', 'def ', 'function ', 'class ', 'import ', 'from ', '$ ', 'sudo ',
    '<html', '<?php', 'SELECT ', '/**', '#include', 'public static', '= {',
    '=> {', 'npm install', 'pip install',
]

STRUCTURAL_ANTI_PATTERNS = [
    (r'^[-*•]\s', 15), (r'^\d+\.\s', 20), (r'\|.*\|.*\|', 3), (r'^#{1,6}\s', 10),
]

TEMPLATE_STRING_PATTERNS = [
    'table of contents', 'facts\n', 'issue\n', 'holding\n', 'reasoning\n',
    'references\n', 'executive summary', 'abstract\n', 'methodology\n',
    'findings\n', 'recommendations\n', 'appendix', '[insert', '[your name',
    '[date]', 'lorem ipsum', 'question:', 'answer:', 'q:', 'a:',
]

TEMPLATE_REGEX_PATTERNS = [
    r'\d{1,3}\s+u\.?\s*s\.?\s+\d+',
    r'question\s*\d+\s*:',
    r'answer\s*\d+\s*:',
]

NON_ESSAY_PATTERNS = [
    r'^\d+\.\s+\w+', r'^\d+\)\s+\w+', r'^Title:\s*', r'^Introduction:\s*$',
    r'^Conclusion:\s*$', r'^Step \d+', r'\btravel guide\b',
    r'\bpacking\s+(?:essentials|list|tips)', r'\btips\s+for\s+travel',
    r'Co\.,?\s+Ltd\.', r'Corporation\s+is\s+a', r'founded\s+in\s+\d{4}',
    r'leading\s+(?:provider|manufacturer|company)',
    r'specializes\s+in\s+(?:the\s+)?(?:research|production|development)',
    r'ingredients:', r'instructions:', r'how\s+to\s+make',
    r'product\s+review', r'pros\s+and\s+cons',
    r'as an ai\s+(?:language\s+)?model', r'i am an ai',
    r'={5,}', r'-{10,}', r'\*{5,}',
    r'\\left\\[\(\[]', r'\\begin\{', r'\\end\{',
    r'eigenvalue', r'eigenvector', r'matrix multiplication',
]

FORMAL_INDICATORS = [
    'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
    'consequently', 'additionally', 'subsequently', 'meanwhile',
    'according to', 'research shows', 'studies indicate', 'evidence suggests',
    'in conclusion', 'in summary', 'firstly', 'secondly',
    'on the other hand', 'in contrast', 'for example', 'for instance',
    'argue that', 'claim that', 'suggest that', 'believe that',
]


def analyze_text_structure(text):
    if not text or not isinstance(text, str):
        return None
    words = text.split()
    word_count = len(words)
    if word_count < 50:
        return None
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    sentence_count = len(sentences)
    if sentence_count < 3:
        return None
    avg_sentence_length = word_count / max(sentence_count, 1)
    sentence_lengths = [len(s.split()) for s in sentences]
    if len(sentence_lengths) > 1:
        mean_len = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - mean_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        sentence_variation = variance ** 0.5
    else:
        sentence_variation = 0
    if '\n\n' in text:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    else:
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
    paragraph_count = len(paragraphs)
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
    for pattern in CODE_PATTERNS:
        if pattern in text:
            return True
    return False


def has_structural_antipatterns(text):
    lines = text.split('\n')
    for pattern, threshold in STRUCTURAL_ANTI_PATTERNS:
        count = sum(1 for line in lines if re.match(pattern, line.strip()))
        if count > threshold:
            return True
    return False


def has_template_patterns(text):
    text_lower = text.lower()
    for pattern in TEMPLATE_STRING_PATTERNS:
        if pattern in text_lower:
            return True
    for pattern in TEMPLATE_REGEX_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def has_non_essay_patterns(text):
    text_lower = text.lower()
    for pattern in NON_ESSAY_PATTERNS:
        if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
            return True
    return False


def is_essay_like(text, min_words=200, max_words=5000, min_paragraphs=2,
                  require_formality=False, strict=False):
    if not text or not isinstance(text, str):
        return False
    if has_code_patterns(text):
        return False
    if has_structural_antipatterns(text):
        return False
    if has_template_patterns(text):
        return False
    metrics = analyze_text_structure(text)
    if metrics is None:
        return False
    if not (min_words <= metrics['word_count'] <= max_words):
        return False
    if metrics['paragraph_count'] < min_paragraphs:
        return False
    if metrics['sentence_count'] < 5:
        return False
    if metrics['avg_sentence_length'] < 8 or metrics['avg_sentence_length'] > 50:
        return False
    if strict:
        if metrics['paragraph_count'] < 3:
            return False
        if metrics['sentence_variation'] < 3:
            return False
        if metrics['formality_score'] < 2:
            return False
    if require_formality and metrics['formality_score'] < 2:
        return False
    return True


def clean_essay_references(text):
    if not isinstance(text, str):
        return ""
    patterns = [
        r'\n\s*References\s*\n', r'\n\s*Works Cited\s*\n',
        r'\n\s*Bibliography\s*\n', r'\n\s*Sources\s*\n',
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
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)


def save_batch(data_batch, output_dir, prefix, batch_idx):
    if not data_batch:
        return None
    df = pd.DataFrame(data_batch)
    out_path = output_dir / f"{prefix}_part_{batch_idx}.parquet"
    df.to_parquet(out_path)
    return out_path


def clean_existing_files(output_dir, prefix):
    for f in output_dir.glob(f"{prefix}_part_*.parquet"):
        f.unlink()


# =============================================================================
# DEDUPLICATION
# =============================================================================

_SEEN_HASHES = set()
_DEDUP_STATS = {'total_checked': 0, 'duplicates_found': 0}


def _text_fingerprint(text: str) -> str:
    import hashlib
    normalized = ' '.join(text.split())
    prefix = normalized[:200] if len(normalized) >= 200 else normalized
    content = f"{prefix}|len={len(normalized)}"
    return hashlib.md5(content.encode()).hexdigest()


def is_duplicate(text: str) -> bool:
    global _SEEN_HASHES, _DEDUP_STATS
    _DEDUP_STATS['total_checked'] += 1
    fp = _text_fingerprint(text)
    if fp in _SEEN_HASHES:
        _DEDUP_STATS['duplicates_found'] += 1
        return True
    _SEEN_HASHES.add(fp)
    return False


def reset_dedup():
    global _SEEN_HASHES, _DEDUP_STATS
    _SEEN_HASHES = set()
    _DEDUP_STATS = {'total_checked': 0, 'duplicates_found': 0}


def get_dedup_stats() -> dict:
    return _DEDUP_STATS.copy()


# =============================================================================
# YIELD ESTIMATION (Phase 1)
# =============================================================================

def estimate_yield_from_sample(source_name: str, sample_size: int = 5000,
                               skip_kaggle_estimation: bool = False) -> Tuple[float, int]:
    """
    Download a sample from the source and measure acceptance rate.
    
    Returns:
        Tuple of (acceptance_rate, processed_count)
    """
    print(f"\n   📊 Estimating yield for {source_name}...")
    
    if not HF_AVAILABLE:
        return 0.5, 0
    
    accepted = 0
    processed = 0
    
    try:
        if source_name == "fineweb_edu":
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", 
                                   split="train", streaming=True)
            for sample in dataset:
                if processed >= sample_size:
                    break
                processed += 1
                text = sample.get('text', '')
                if is_essay_like(text, min_words=300, max_words=4000, min_paragraphs=3):
                    accepted += 1
                    
        elif source_name == "ivypanda":
            dataset = load_dataset("qwedsacf/ivypanda-essays", split="train", streaming=True)
            for row in dataset:
                if processed >= sample_size:
                    break
                processed += 1
                text_raw = row.get('text') or row.get('TEXT') or row.get('essay')
                if text_raw:
                    text = clean_essay_references(text_raw)
                    if len(text.split()) >= 100:
                        accepted += 1
                        
        elif source_name == "cosmopedia_stanford":
            dataset = load_dataset("HuggingFaceTB/cosmopedia", "stanford", 
                                   split="train", streaming=True)
            for sample in dataset:
                if processed >= sample_size:
                    break
                processed += 1
                text = sample.get('text', '')
                if is_essay_like(text, min_words=300, max_words=4000, min_paragraphs=3):
                    accepted += 1
                    
        elif source_name == "cosmopedia_web_samples_v2":
            dataset = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", 
                                   split="train", streaming=True)
            for sample in dataset:
                if processed >= sample_size:
                    break
                processed += 1
                text = sample.get('text', '')
                if is_essay_like(text, min_words=250, max_words=4000, min_paragraphs=2):
                    accepted += 1
                    
        elif source_name == "lmsys":
            dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
            for sample in dataset:
                if processed >= sample_size:
                    break
                processed += 1
                if sample.get('language') != 'English':
                    continue
                conv = sample.get('conversation', [])
                for turn in conv:
                    if turn['role'] != 'assistant':
                        continue
                    text = turn.get('content', '')
                    if has_non_essay_patterns(text):
                        break
                    if is_essay_like(text, min_words=250, max_words=3000,
                                    min_paragraphs=3, require_formality=True):
                        accepted += 1
                    break
                    
        elif source_name in ("aiessays_human", "aiessays_ai"):
            # Kaggle sources - use historical estimate or optimistic default
            # Can't easily stream these, so we use defaults
            if source_name == "aiessays_human":
                return 0.60, 0  # Human essays have ~60% acceptance
            else:
                return 0.85, 0  # AI essays have ~85% acceptance
                
    except Exception as e:
        print(f"   ⚠️  Estimation failed for {source_name}: {e}")
        return 0.5, 0
    
    acceptance_rate = accepted / processed if processed > 0 else 0.5
    print(f"   → {source_name}: {accepted}/{processed} accepted ({acceptance_rate:.1%})")
    return acceptance_rate, processed


def load_yield_cache() -> Dict[str, float]:
    """Load historical yield estimates from cache file."""
    if YIELD_CACHE_FILE.exists():
        try:
            with open(YIELD_CACHE_FILE, 'r') as f:
                data = json.load(f)
                return data.get('yields', {})
        except Exception:
            pass
    return {}


def save_yield_cache(yields: Dict[str, float]):
    """Save yield estimates to cache file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        'timestamp': datetime.now().isoformat(),
        'yields': yields
    }
    with open(YIELD_CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# DYNAMIC ALLOCATION (Phase 2)
# =============================================================================

def calculate_allocations(
    sources: List[DataSource],
    target_total: int,
    yields: Dict[str, float]
) -> Dict[str, int]:
    """
    Calculate optimal allocations for each source based on estimated yields.
    
    Strategy:
    1. Start with proportional allocation based on yields
    2. Enforce min/max constraints
    3. Redistribute shortfalls to higher-priority sources
    4. Account for low-yield sources needing more raw downloads
    
    Returns:
        Dict mapping source name to target count
    """
    allocations = {}
    
    # Step 1: Calculate effective yields (how many we expect to get per download)
    effective_yields = {}
    for source in sources:
        yield_rate = yields.get(source.name, source.estimated_yield or 0.5)
        # Clamp to reasonable range
        effective_yields[source.name] = max(0.01, min(1.0, yield_rate))
    
    # Step 2: Initial proportional allocation weighted by priority and yield
    total_weight = 0
    weights = {}
    for source in sources:
        # Weight = priority * sqrt(yield) - favors high-yield, high-priority sources
        eff_yield = effective_yields[source.name]
        weight = source.priority * (eff_yield ** 0.5)
        weights[source.name] = weight
        total_weight += weight
    
    # Allocate based on weights
    for source in sources:
        share = weights[source.name] / total_weight if total_weight > 0 else 1 / len(sources)
        # Apply min/max constraints
        share = max(source.min_share, min(source.max_share, share))
        allocations[source.name] = int(target_total * share)
    
    # Step 3: Ensure total matches target (redistribute any gaps)
    allocated_total = sum(allocations.values())
    gap = target_total - allocated_total
    
    if gap != 0:
        # Sort by priority (highest first) for redistribution
        sorted_sources = sorted(sources, key=lambda s: -s.priority)
        
        for source in sorted_sources:
            if gap == 0:
                break
            current = allocations[source.name]
            max_allowed = int(target_total * source.max_share)
            
            if gap > 0:
                # Add to high-priority sources that haven't hit max
                can_add = max_allowed - current
                add_amount = min(gap, can_add)
                allocations[source.name] += add_amount
                gap -= add_amount
            else:
                # Remove from low-priority sources (reverse order)
                min_required = int(target_total * source.min_share)
                can_remove = current - min_required
                remove_amount = min(-gap, can_remove)
                allocations[source.name] -= remove_amount
                gap += remove_amount
    
    return allocations


def estimate_raw_download_needed(target_count: int, yield_rate: float) -> int:
    """
    Estimate how many raw samples need to be downloaded to get target accepted samples.
    
    For low-yield sources, we may need to download many more samples.
    """
    if yield_rate <= 0.01:
        yield_rate = 0.01  # Prevent division by zero
    
    # Add 20% buffer for safety
    raw_needed = int(target_count / yield_rate * 1.2)
    return raw_needed


# =============================================================================
# DOWNLOAD FUNCTIONS (with yield tracking)
# =============================================================================

def download_fineweb_edu(limit: int, batch_size: int = 50000, 
                        yield_estimate: float = 0.5) -> int:
    """Download FineWeb-Edu with dynamic limit awareness."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: FineWeb-Edu (Human, target={limit:,})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        return 0
    
    clean_existing_files(HUMAN_DIR, "fineweb")
    
    # Calculate how many raw samples we might need
    raw_estimate = estimate_raw_download_needed(limit, yield_estimate)
    print(f"   📊 Estimated yield: {yield_estimate:.1%}")
    print(f"   📦 Will process up to ~{raw_estimate:,} raw samples")
    
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", 
                               split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        raw_processed = 0
        
        pbar = tqdm(total=limit, desc="FineWeb-Edu")
        
        for sample in dataset:
            if count >= limit:
                break
            # Stop if we've processed way more than expected (source exhausted or bad estimate)
            if raw_processed > raw_estimate * 2:
                print(f"\n   ⚠️  Processed {raw_processed:,} samples, only got {count:,}. Stopping early.")
                break
            
            raw_processed += 1
            text = sample.get('text', '')
            
            if not is_essay_like(text, min_words=300, max_words=4000, min_paragraphs=3):
                continue
            if is_duplicate(text):
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
        actual_yield = count / raw_processed if raw_processed > 0 else 0
        print(f"✅ FineWeb-Edu: {count:,} samples saved (actual yield: {actual_yield:.1%})")
        return count
        
    except Exception as e:
        print(f"❌ FineWeb-Edu failed: {e}")
        return 0


def download_ivypanda(limit: int, batch_size: int = None, 
                      yield_estimate: float = 0.9) -> int:
    """Download IvyPanda essays with dynamic limit awareness."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: IvyPanda (Human College Essays, target={limit:,})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        return 0
    
    if batch_size is None:
        batch_size = max(5000, int(limit) // 10)
        batch_size = min(50000, batch_size)
    
    clean_existing_files(HUMAN_DIR, "ivypanda")
    
    raw_estimate = estimate_raw_download_needed(limit, yield_estimate)
    print(f"   📊 Estimated yield: {yield_estimate:.1%}")
    
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
            if is_duplicate(text):
                continue
            
            data_batch.append({'text': text, 'source': 'ivypanda', 'label': 0})
            count += 1
            
            if len(data_batch) >= batch_size:
                save_batch(data_batch, HUMAN_DIR, "ivypanda", batch_idx)
                data_batch = []
                batch_idx += 1
        
        if data_batch:
            save_batch(data_batch, HUMAN_DIR, "ivypanda", batch_idx)
        
        print(f"✅ IvyPanda: {count:,} essays saved")
        return count
        
    except Exception as e:
        print(f"❌ IvyPanda failed: {e}")
        return 0


def download_cosmopedia(
    stanford_limit: int,
    web_samples_v2_limit: int,
    batch_size: int = 50000,
    stanford_yield: float = 0.3,
    web_yield: float = 0.7
) -> Tuple[int, int]:
    """Download Cosmopedia subsets with dynamic awareness."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: Cosmopedia (AI)")
    print(f"  - stanford: {stanford_limit:,} samples (est. yield: {stanford_yield:.1%})")
    print(f"  - web_samples_v2: {web_samples_v2_limit:,} samples (est. yield: {web_yield:.1%})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        return 0, 0
    
    clean_existing_files(AI_DIR, "cosmopedia")
    
    stanford_count = 0
    web_samples_v2_count = 0
    batch_idx = 0
    
    # Stanford subset
    if stanford_limit > 0:
        print(f"\n→ Loading Cosmopedia/stanford...")
        raw_estimate = estimate_raw_download_needed(stanford_limit, stanford_yield)
        
        try:
            dataset = load_dataset("HuggingFaceTB/cosmopedia", "stanford", 
                                   split="train", streaming=True)
            
            data_batch = []
            count = 0
            raw_processed = 0
            
            pbar = tqdm(total=stanford_limit, desc="Cosmopedia/stanford")
            
            for sample in dataset:
                if count >= stanford_limit:
                    break
                if raw_processed > raw_estimate * 2:
                    print(f"\n   ⚠️  Stanford: Processed {raw_processed:,}, only got {count:,}. Stopping.")
                    break
                
                raw_processed += 1
                text = sample.get('text', '')
                
                if not is_essay_like(text, min_words=300, max_words=4000, min_paragraphs=3):
                    continue
                if is_duplicate(text):
                    continue
                
                data_batch.append({'text': text, 'source': 'cosmopedia_stanford', 'label': 1})
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
            stanford_count = count
            print(f"  ✓ stanford: {count:,} samples")
            
        except Exception as e:
            print(f"  ✗ stanford failed: {e}")
    
    # Web samples subset
    if web_samples_v2_limit > 0:
        print(f"\n→ Loading Cosmopedia/web_samples_v2...")
        
        try:
            dataset = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", 
                                   split="train", streaming=True)
            
            data_batch = []
            count = 0
            
            pbar = tqdm(total=web_samples_v2_limit, desc="Cosmopedia/web_samples_v2")
            
            for sample in dataset:
                if count >= web_samples_v2_limit:
                    break
                
                text = sample.get('text', '')
                
                if not is_essay_like(text, min_words=250, max_words=4000, min_paragraphs=2):
                    continue
                if is_duplicate(text):
                    continue
                
                data_batch.append({'text': text, 'source': 'cosmopedia_web_samples_v2', 'label': 1})
                count += 1
                pbar.update(1)
                
                if len(data_batch) >= batch_size:
                    save_batch(data_batch, AI_DIR, "cosmopedia", batch_idx)
                    data_batch = []
                    batch_idx += 1
                    gc.collect()
            
            if data_batch:
                save_batch(data_batch, AI_DIR, "cosmopedia", batch_idx)
            
            pbar.close()
            web_samples_v2_count = count
            print(f"  ✓ web_samples_v2: {count:,} samples")
            
        except Exception as e:
            print(f"  ✗ web_samples_v2 failed: {e}")
    
    total = stanford_count + web_samples_v2_count
    print(f"✅ Cosmopedia total: {total:,} samples saved")
    return stanford_count, web_samples_v2_count


def download_lmsys(limit: int, batch_size: int = 50000,
                   yield_estimate: float = 0.03) -> int:
    """Download LMSYS with awareness of its low yield rate."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: LMSYS Chat-1M (AI, target={limit:,})")
    print(f"{'='*60}")
    
    if not HF_AVAILABLE:
        return 0
    
    clean_existing_files(AI_DIR, "lmsys")
    
    raw_estimate = estimate_raw_download_needed(limit, yield_estimate)
    print(f"   📊 Estimated yield: {yield_estimate:.1%}")
    print(f"   ⚠️  Low yield expected - will process up to ~{raw_estimate:,} conversations")
    
    try:
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        
        data_batch = []
        count = 0
        batch_idx = 0
        raw_processed = 0
        
        pbar = tqdm(total=limit, desc="LMSYS")
        
        for sample in dataset:
            if count >= limit:
                break
            if raw_processed > raw_estimate * 1.5:  # More conservative for slow source
                print(f"\n   ⚠️  LMSYS: Processed {raw_processed:,}, only got {count:,}. Stopping.")
                break
            
            raw_processed += 1
            
            if sample.get('language') != 'English':
                continue
            
            conv = sample.get('conversation', [])
            
            for turn in conv:
                if turn['role'] != 'assistant':
                    continue
                
                text = turn.get('content', '')
                
                if has_non_essay_patterns(text):
                    break
                if not is_essay_like(text, min_words=250, max_words=3000,
                                    min_paragraphs=3, require_formality=True):
                    break
                if is_duplicate(text):
                    break
                
                data_batch.append({'text': text, 'source': 'lmsys', 'label': 1})
                count += 1
                pbar.update(1)
                break
            
            if len(data_batch) >= batch_size:
                save_batch(data_batch, AI_DIR, "lmsys", batch_idx)
                data_batch = []
                batch_idx += 1
                gc.collect()
        
        if data_batch:
            save_batch(data_batch, AI_DIR, "lmsys", batch_idx)
        
        pbar.close()
        actual_yield = count / raw_processed if raw_processed > 0 else 0
        print(f"✅ LMSYS: {count:,} samples saved (actual yield: {actual_yield:.1%})")
        return count
        
    except Exception as e:
        print(f"❌ LMSYS failed: {e}")
        return 0


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


def parse_ai_essays_dataset_capped(input_path, human_cap: int, ai_cap: int,
                                   batch_size: int = 50000) -> Tuple[int, int]:
    """Parse AI Generated Essays dataset with caps."""
    print(f"\n{'='*60}")
    print(f"PARSING: AI Essays Dataset (Human cap={human_cap:,}, AI cap={ai_cap:,})")
    print(f"{'='*60}")
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 0, 0
    
    df = pd.read_csv(input_path) if input_path.suffix == '.csv' else pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")
    
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
    
    for _, row in tqdm(human_df.iterrows(), total=min(len(human_df), human_cap), 
                       desc="AI Essays (Human)"):
        if human_count >= human_cap:
            print(f"   → Reached human cap of {human_cap:,} samples")
            break
        
        text = str(row[text_col])
        if len(text.split()) < 100:
            continue
        if is_duplicate(text):
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
    
    for _, row in tqdm(ai_df.iterrows(), total=min(len(ai_df), ai_cap), 
                       desc="AI Essays (AI)"):
        if ai_count >= ai_cap:
            print(f"   → Reached AI cap of {ai_cap:,} samples")
            break
        
        text = str(row[text_col])
        if len(text.split()) < 100:
            continue
        if is_duplicate(text):
            continue
        data_batch.append({'text': text, 'source': 'aiessays_ai', 'label': 1})
        ai_count += 1
        if len(data_batch) >= batch_size:
            save_batch(data_batch, AI_DIR, "aiessays_ai", batch_idx)
            data_batch = []
            batch_idx += 1
    if data_batch:
        save_batch(data_batch, AI_DIR, "aiessays_ai", batch_idx)
    
    print(f"✅ AI Essays: {human_count:,} human, {ai_count:,} AI saved")
    return human_count, ai_count


# =============================================================================
# MAIN ORCHESTRATOR (V5 - Dynamic Balancing)
# =============================================================================

def download_all(
    target_per_class: int = 600000,
    skip_estimation: bool = False,
    skip_kaggle: bool = False,
    estimation_sample_size: int = 5000,
):
    """
    Download all sources with DYNAMIC balancing based on estimated yields.
    
    Two-Phase Approach:
    1. ESTIMATION: Sample each source to measure acceptance rates
    2. ALLOCATION: Distribute quotas based on yields + priorities
    """
    print("=" * 70)
    print("ESSAY-FOCUSED DATA DOWNLOAD (V5 - DYNAMIC BALANCER)")
    print(f"Target: ~{target_per_class:,} samples per class")
    print("=" * 70)
    
    reset_dedup()
    ensure_dirs()
    
    stats = {'human': {}, 'ai': {}}
    yields = {}
    
    # =========================================================================
    # PHASE 1: YIELD ESTIMATION
    # =========================================================================
    
    if skip_estimation:
        print("\n📊 PHASE 1: Loading cached yield estimates...")
        yields = load_yield_cache()
        if not yields:
            print("   ⚠️  No cache found, using defaults")
            yields = {
                'aiessays_human': 0.60,
                'aiessays_ai': 0.85,
                'fineweb_edu': 0.45,
                'ivypanda': 0.90,
                'cosmopedia_stanford': 0.30,
                'cosmopedia_web_samples_v2': 0.70,
                'lmsys': 0.03,
            }
    else:
        print("\n📊 PHASE 1: ESTIMATING YIELDS")
        print("=" * 50)
        print(f"   Sampling {estimation_sample_size:,} items per source...")
        
        all_sources = HUMAN_SOURCES + AI_SOURCES
        for source in all_sources:
            if source.is_kaggle:
                # Use default estimates for Kaggle (can't stream easily)
                if source.name == 'aiessays_human':
                    yields[source.name] = 0.60
                else:
                    yields[source.name] = 0.85
            else:
                rate, _ = estimate_yield_from_sample(source.name, estimation_sample_size)
                yields[source.name] = rate
        
        # Cache for future runs
        save_yield_cache(yields)
        print(f"\n   💾 Yields cached to {YIELD_CACHE_FILE}")
    
    print("\n   📈 Estimated Yields:")
    for name, rate in sorted(yields.items()):
        print(f"      {name:30} {rate:.1%}")
    
    # =========================================================================
    # PHASE 2: DYNAMIC ALLOCATION
    # =========================================================================
    
    print("\n🎯 PHASE 2: DYNAMIC ALLOCATION")
    print("=" * 50)
    
    human_allocations = calculate_allocations(HUMAN_SOURCES, target_per_class, yields)
    ai_allocations = calculate_allocations(AI_SOURCES, target_per_class, yields)
    
    print("\n   📊 Calculated Allocations:")
    print("   HUMAN:")
    for source in HUMAN_SOURCES:
        alloc = human_allocations[source.name]
        yld = yields.get(source.name, 0.5)
        raw_needed = estimate_raw_download_needed(alloc, yld)
        print(f"      {source.name:30} {alloc:>8,} (yield: {yld:.1%}, ~{raw_needed:,} raw)")
    print(f"      {'TOTAL':30} {sum(human_allocations.values()):>8,}")
    
    print("   AI:")
    for source in AI_SOURCES:
        alloc = ai_allocations[source.name]
        yld = yields.get(source.name, 0.5)
        raw_needed = estimate_raw_download_needed(alloc, yld)
        print(f"      {source.name:30} {alloc:>8,} (yield: {yld:.1%}, ~{raw_needed:,} raw)")
    print(f"      {'TOTAL':30} {sum(ai_allocations.values()):>8,}")
    
    # =========================================================================
    # PHASE 3: DOWNLOAD
    # =========================================================================
    
    print("\n📥 PHASE 3: DOWNLOADING")
    print("=" * 50)
    
    # KAGGLE DATASETS
    if not skip_kaggle and KAGGLE_AVAILABLE:
        kaggle_raw_dir = DATA_DIR / "raw_kaggle"
        ai_essays_file = download_kaggle_dataset(
            'shanegerami/ai-vs-human-text',
            kaggle_raw_dir / "ai_essays"
        )
        if ai_essays_file:
            h, a = parse_ai_essays_dataset_capped(
                ai_essays_file,
                human_allocations.get('aiessays_human', 0),
                ai_allocations.get('aiessays_ai', 0)
            )
            stats['human']['aiessays_human'] = h
            stats['ai']['aiessays_ai'] = a
    
    # HUGGINGFACE DATASETS
    if HF_AVAILABLE:
        # Calculate remaining after Kaggle
        human_remaining = {
            k: max(0, human_allocations.get(k, 0) - stats['human'].get(k, 0))
            for k in human_allocations
        }
        ai_remaining = {
            k: max(0, ai_allocations.get(k, 0) - stats['ai'].get(k, 0))
            for k in ai_allocations
        }
        
        # HUMAN SOURCES
        if human_remaining.get('fineweb_edu', 0) > 0:
            stats['human']['fineweb_edu'] = download_fineweb_edu(
                limit=human_remaining['fineweb_edu'],
                yield_estimate=yields.get('fineweb_edu', 0.5)
            )
        
        if human_remaining.get('ivypanda', 0) > 0:
            stats['human']['ivypanda'] = download_ivypanda(
                limit=human_remaining['ivypanda'],
                yield_estimate=yields.get('ivypanda', 0.9)
            )
        
        # AI SOURCES
        stanford_limit = ai_remaining.get('cosmopedia_stanford', 0)
        web_limit = ai_remaining.get('cosmopedia_web_samples_v2', 0)
        if stanford_limit > 0 or web_limit > 0:
            s_count, w_count = download_cosmopedia(
                stanford_limit=stanford_limit,
                web_samples_v2_limit=web_limit,
                stanford_yield=yields.get('cosmopedia_stanford', 0.3),
                web_yield=yields.get('cosmopedia_web_samples_v2', 0.7)
            )
            stats['ai']['cosmopedia_stanford'] = s_count
            stats['ai']['cosmopedia_web_samples_v2'] = w_count
        
        if ai_remaining.get('lmsys', 0) > 0:
            stats['ai']['lmsys'] = download_lmsys(
                limit=ai_remaining['lmsys'],
                yield_estimate=yields.get('lmsys', 0.03)
            )
    
    # =========================================================================
    # PHASE 4: REBALANCING (if needed)
    # =========================================================================
    
    human_total = sum(stats['human'].values())
    ai_total = sum(stats['ai'].values())
    
    human_shortfall = target_per_class - human_total
    ai_shortfall = target_per_class - ai_total
    
    if human_shortfall > target_per_class * 0.1 or ai_shortfall > target_per_class * 0.1:
        print("\n🔄 PHASE 4: REBALANCING")
        print("=" * 50)
        print(f"   Human shortfall: {human_shortfall:,} ({human_shortfall/target_per_class:.1%})")
        print(f"   AI shortfall: {ai_shortfall:,} ({ai_shortfall/target_per_class:.1%})")
        print("   💡 Consider adjusting source priorities or adding new sources")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE - SUMMARY (V5 Dynamic Balancer)")
    print("=" * 70)
    
    human_total = sum(stats['human'].values())
    ai_total = sum(stats['ai'].values())
    
    print(f"\n🎯 Target: {target_per_class:,} per class")
    
    print(f"\n📊 HUMAN DATA (label=0) - Total: {human_total:,}:")
    print(f"   {'Source':<30} {'Count':>8} {'%':>6} {'Target':>10} {'Status':>10}")
    print(f"   {'-'*70}")
    for source, count in stats['human'].items():
        pct = (count / human_total * 100) if human_total > 0 else 0
        target = human_allocations.get(source, 0)
        status = "✅" if count >= target * 0.8 else "⚠️ SHORT"
        print(f"   {source:<30} {count:>8,} {pct:>5.1f}% {target:>10,} {status:>10}")
    print(f"   {'-'*70}")
    print(f"   {'TOTAL':<30} {human_total:>8,}")
    
    print(f"\n🤖 AI DATA (label=1) - Total: {ai_total:,}:")
    print(f"   {'Source':<30} {'Count':>8} {'%':>6} {'Target':>10} {'Status':>10}")
    print(f"   {'-'*70}")
    for source, count in stats['ai'].items():
        pct = (count / ai_total * 100) if ai_total > 0 else 0
        target = ai_allocations.get(source, 0)
        status = "✅" if count >= target * 0.8 else "⚠️ SHORT"
        print(f"   {source:<30} {count:>8,} {pct:>5.1f}% {target:>10,} {status:>10}")
    print(f"   {'-'*70}")
    print(f"   {'TOTAL':<30} {ai_total:>8,}")
    
    print(f"\n📁 Output: {DATA_DIR.resolve()}")
    
    if human_total > 0 and ai_total > 0:
        ratio = human_total / ai_total
        status = "✅ balanced" if 0.8 <= ratio <= 1.2 else "⚠️ imbalanced"
        print(f"\n⚖️  Class Balance: {status} (ratio: {ratio:.2f})")
    
    # Updated yields for next run
    print("\n📈 Updated Yield Estimates (for next run):")
    for source, count in stats['human'].items():
        old_yield = yields.get(source, 0.5)
        target = human_allocations.get(source, 0)
        # If we hit target, yield is good. If short, yield was lower.
        actual = count / target if target > 0 else old_yield
        print(f"   {source}: {old_yield:.1%} -> {actual:.1%}")
    for source, count in stats['ai'].items():
        old_yield = yields.get(source, 0.5)
        target = ai_allocations.get(source, 0)
        actual = count / target if target > 0 else old_yield
        print(f"   {source}: {old_yield:.1%} -> {actual:.1%}")
    
    dedup_stats = get_dedup_stats()
    if dedup_stats['total_checked'] > 0:
        dup_pct = (dedup_stats['duplicates_found'] / dedup_stats['total_checked']) * 100
        print(f"\n🔍 Deduplication: {dedup_stats['duplicates_found']:,} duplicates found "
              f"({dup_pct:.1f}% of {dedup_stats['total_checked']:,} checked)")
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download essay-focused data with DYNAMIC balancing (V5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 600k per class with yield estimation
  python download_data_v5.py
  
  # Custom target
  python download_data_v5.py --target 300000
  
  # Skip estimation (use cached yields)
  python download_data_v5.py --skip-estimation
  
  # Larger estimation sample for more accuracy
  python download_data_v5.py --estimation-size 10000
"""
    )
    
    parser.add_argument("--target", type=int, default=600000,
                        help="Target samples per class (default: 600000)")
    parser.add_argument("--skip-estimation", action="store_true",
                        help="Skip yield estimation, use cached values")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle datasets")
    parser.add_argument("--estimation-size", type=int, default=5000,
                        help="Sample size for yield estimation (default: 5000)")
    
    args = parser.parse_args()
    
    download_all(
        target_per_class=args.target,
        skip_estimation=args.skip_estimation,
        skip_kaggle=args.skip_kaggle,
        estimation_sample_size=args.estimation_size,
    )


if __name__ == "__main__":
    main()
