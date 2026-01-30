"""
Data Sampling Script - Quality Check (Lightweight Version)

Grabs a small number of samples from each HuggingFace data source
using direct HTTP requests (no pyarrow dependency).

This lets you visually inspect the data quality before running
the full download on Modal.

Usage:
    python scripts/sample_data.py
    python scripts/sample_data.py --samples 3
    python scripts/sample_data.py --output samples.txt
"""

import os
import sys
import re
import json
import requests
from pathlib import Path
from datetime import datetime

# =============================================================================
# AUTH TOKENS
# =============================================================================

def get_hf_token():
    """Get HuggingFace token from environment or CLI login file."""
    # Check environment first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    
    # Check huggingface-cli login file
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        try:
            return token_file.read_text().strip()
        except Exception:
            pass
    
    # Also check older location
    token_file_alt = Path.home() / ".huggingface" / "token"
    if token_file_alt.exists():
        try:
            return token_file_alt.read_text().strip()
        except Exception:
            pass
    
    return None

HF_TOKEN = get_hf_token()

# Check for Kaggle credentials
def get_kaggle_creds():
    """Check if Kaggle API credentials exist."""
    # Check standard home location
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    
    # Check project root (common alternative location)
    project_root = Path(__file__).resolve().parent.parent
    project_kaggle = project_root / ".kaggle" / "kaggle.json"
    if project_kaggle.exists():
        # Set the config directory so Kaggle API finds it
        os.environ["KAGGLE_CONFIG_DIR"] = str(project_root / ".kaggle")
        return True
    
    # Also check if it's directly in project root
    root_kaggle = project_root / "kaggle.json"
    if root_kaggle.exists():
        # Copy to temp .kaggle dir or set env
        os.environ["KAGGLE_CONFIG_DIR"] = str(project_root)
        return True
    
    return False

KAGGLE_AVAILABLE = get_kaggle_creds()

# =============================================================================
# ESSAY DETECTION (standalone)
# =============================================================================

CODE_PATTERNS = [
    '```', 'def ', 'function ', 'class ', 'import ', 'from ',
    '$ ', 'sudo ', '<html', '<?php', 'SELECT ', '/**', '#include',
    'public static', '= {', '=> {', 'npm install', 'pip install',
]

STRUCTURAL_ANTI_PATTERNS = [
    (r'^[-*•]\s', 15),
    (r'^\d+\.\s', 20),
    (r'\|.*\|.*\|', 3),
    (r'^#{1,6}\s', 10),
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
    r'^\d+\)\s+\w+',          # "1) First step"
    
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

FORMAL_INDICATORS = [
    'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
    'consequently', 'additionally', 'subsequently', 'meanwhile',
    'according to', 'research shows', 'studies indicate', 'evidence suggests',
    'it is important', 'significant', 'analysis', 'demonstrate',
    'in conclusion', 'in summary', 'to summarize', 'firstly', 'secondly',
    'on the other hand', 'in contrast', 'for example', 'for instance',
    'as a result', 'in addition', 'in particular', 'specifically',
    'argue that', 'claim that', 'suggest that', 'believe that',
    'this essay', 'this paper', 'thesis', 'perspective', 'viewpoint',
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
        'word_count': word_count, 'sentence_count': sentence_count,
        'paragraph_count': paragraph_count, 'avg_sentence_length': avg_sentence_length,
        'sentence_variation': sentence_variation, 'formality_score': formality_score,
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
    if require_formality and metrics['formality_score'] < 1:
        return False
    return True


def clean_essay_references(text):
    if not isinstance(text, str):
        return ""
    patterns = [r'\n\s*References\s*\n', r'\n\s*Works Cited\s*\n', r'\n\s*Bibliography\s*\n']
    cleaned = text
    for p in patterns:
        parts = re.split(p, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if len(parts) > 1:
            cleaned = parts[0].strip()
    return cleaned


# =============================================================================
# HF API SAMPLING (lightweight, no pyarrow needed)
# =============================================================================

def get_dataset_info(dataset_id):
    """Get dataset info to find available configs."""
    url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_hf_rows(dataset_id, config=None, split="train", offset=0, limit=100):
    """
    Fetch rows from HuggingFace datasets API.
    """
    base_url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset_id,
        "split": split,
        "offset": offset,
        "length": limit
    }
    if config:
        params["config"] = config
    
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    try:
        resp = requests.get(base_url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("rows", [])
    except requests.exceptions.HTTPError as e:
        # Try to get more info about the error
        if e.response.status_code == 422:
            # Try to find the right config
            info = get_dataset_info(dataset_id)
            if info:
                configs = list(info.get("dataset_info", {}).keys())
                if configs:
                    print(f"  ⚠️ Available configs for {dataset_id}: {configs}")
        print(f"  ⚠️ API error: {e}")
        return []
    except Exception as e:
        print(f"  ⚠️ API error: {e}")
        return []


def sample_from_hf(name, dataset_id, config, text_key, filter_kwargs, 
                   samples_wanted=2, max_batches=10, is_chat=False, label=None,
                   text_extractor=None):
    """
    Sample from a HuggingFace dataset using the API.
    """
    print(f"\n{'='*60}")
    print(f"📥 Sampling: {name}")
    print(f"   Dataset: {dataset_id}" + (f" ({config})" if config else ""))
    print(f"{'='*60}")
    
    samples = []
    inspected = 0
    offset = 0
    batch_size = 100
    
    for batch_num in range(max_batches):
        if len(samples) >= samples_wanted:
            break
        
        print(f"  Fetching batch {batch_num + 1}...")
        rows = fetch_hf_rows(dataset_id, config=config, offset=offset, limit=batch_size)
        
        if not rows:
            print(f"  No more rows available")
            break
        
        for row_data in rows:
            if len(samples) >= samples_wanted:
                break
            
            inspected += 1
            row = row_data.get("row", {})
            
            # Extract text
            if text_extractor:
                text = text_extractor(row)
            elif is_chat:
                # Handle conversation format
                conv = row.get('conversation', [])
                text = None
                for turn in conv:
                    if turn.get('role') == 'assistant':
                        text = turn.get('content', '')
                        break
            else:
                text = row.get(text_key, '')
            
            if not text:
                continue
            
            # Apply cleaning if needed
            if 'ivypanda' in name.lower():
                text = clean_essay_references(text)
            
            # Apply filtering
            if is_essay_like(text, **filter_kwargs):
                metrics = analyze_text_structure(text)
                samples.append({
                    'source': name,
                    'label': label,
                    'label_name': 'Human' if label == 0 else 'AI',
                    'text': text,
                    'stats': {
                        'words': metrics['word_count'],
                        'paragraphs': metrics['paragraph_count'],
                        'formal_indicators': metrics['formality_score']
                    }
                })
                print(f"  ✓ Found sample {len(samples)}/{samples_wanted} "
                      f"({metrics['word_count']} words, {metrics['formality_score']} formal)")
        
        offset += batch_size
    
    acceptance_rate = (len(samples) / inspected * 100) if inspected > 0 else 0
    print(f"  📊 Found {len(samples)}/{samples_wanted} (inspected {inspected}, rate: {acceptance_rate:.1f}%)")
    
    return samples


def truncate_text(text, max_chars=400):
    """Truncate text for display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."


def display_samples(samples):
    """Display samples in a readable format."""
    print("\n" + "=" * 70)
    print("📋 SAMPLE REVIEW")
    print("=" * 70)
    
    by_source = {}
    for s in samples:
        src = s['source']
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(s)
    
    for source, source_samples in by_source.items():
        label_str = source_samples[0].get('label_name', 'Unknown')
        print(f"\n{'─'*70}")
        print(f"📦 {source} [{label_str}] ({len(source_samples)} samples)")
        print(f"{'─'*70}")
        
        for i, sample in enumerate(source_samples, 1):
            stats = sample['stats']
            print(f"\n  Sample {i}:")
            print(f"  ├─ Words: {stats['words']} | Paragraphs: {stats['paragraphs']} | Formal: {stats['formal_indicators']}")
            print(f"  └─ Preview:")
            
            preview = truncate_text(sample['text'])
            for line in preview.split('\n')[:4]:
                if line.strip():
                    print(f"     │ {line[:80]}{'...' if len(line) > 80 else ''}")


def save_full_samples(samples, output_path):
    """Save full samples to a text file for review."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DATA QUALITY SAMPLES - FULL TEXT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        by_source = {}
        for s in samples:
            src = s['source']
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(s)
        
        for source, source_samples in by_source.items():
            label_str = source_samples[0].get('label_name', 'Unknown')
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"SOURCE: {source} [{label_str}]\n")
            f.write("=" * 80 + "\n")
            
            for i, sample in enumerate(source_samples, 1):
                stats = sample['stats']
                f.write(f"\n{'─'*80}\n")
                f.write(f"SAMPLE {i}\n")
                f.write(f"Words: {stats['words']} | Paragraphs: {stats['paragraphs']} | Formal Indicators: {stats['formal_indicators']}\n")
                f.write(f"{'─'*80}\n\n")
                f.write(sample['text'])
                f.write("\n\n")
    
    print(f"\n💾 Full samples saved to: {output_path}")


def sample_all(samples_per_source=2):
    """Sample from all data sources."""
    all_samples = []
    
    # =========================================================================
    # HUMAN SOURCES
    # =========================================================================
    
    
    # FineWeb-Edu
    all_samples.extend(sample_from_hf(
        name="FineWeb-Edu",
        dataset_id="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_key="text",
        filter_kwargs={'min_words': 300, 'max_words': 4000, 'min_paragraphs': 3},
        samples_wanted=samples_per_source,
        label=0
    ))
    
    # IvyPanda - uses uppercase TEXT key
    def ivypanda_extractor(row):
        # Handle both uppercase and lowercase key names
        text = row.get('TEXT') or row.get('text') or row.get('essay') or ''
        return clean_essay_references(text) if text else None
    
    all_samples.extend(sample_from_hf(
        name="IvyPanda",
        dataset_id="qwedsacf/ivypanda-essays",
        config="default",
        text_key=None,
        text_extractor=ivypanda_extractor,
        filter_kwargs={'min_words': 100, 'max_words': 5000, 'min_paragraphs': 2},
        samples_wanted=samples_per_source,
        label=0
    ))
    
    # =========================================================================
    # AI SOURCES
    # =========================================================================
    
    # Cosmopedia stanford (removed 'stories' - children's fiction not argumentative essays)
    all_samples.extend(sample_from_hf(
        name="Cosmopedia/stanford",
        dataset_id="HuggingFaceTB/cosmopedia",
        config="stanford",
        text_key="text",
        filter_kwargs={'min_words': 300, 'max_words': 4000, 'min_paragraphs': 3},
        samples_wanted=samples_per_source,
        label=1
    ))
    
    # LMSYS (try with auth)
    if HF_TOKEN:
        print("\n🔑 HF_TOKEN found - attempting authenticated datasets...")
        
        def lmsys_extractor(row):
            if row.get('language') != 'English':
                return None
            for turn in row.get('conversation', []):
                if turn.get('role') == 'assistant':
                    text = turn.get('content', '')
                    # Skip non-essay content (travel guides, business descriptions, etc.)
                    if has_non_essay_patterns(text):
                        return None
                    return text
            return None
        
        all_samples.extend(sample_from_hf(
            name="LMSYS Chat",
            dataset_id="lmsys/lmsys-chat-1m",
            config="default",
            text_key=None,
            text_extractor=lmsys_extractor,
            filter_kwargs={'min_words': 300, 'max_words': 3000, 'min_paragraphs': 3, 'strict': True},
            samples_wanted=samples_per_source,
            max_batches=20,  # Higher limit for strict filtering
            label=1
        ))
        
        def wildchat_extractor(row):
            for turn in row.get('conversation', []):
                if turn.get('role') == 'assistant':
                    text = turn.get('content', '')
                    # Skip non-essay content (travel guides, business descriptions, etc.)
                    if has_non_essay_patterns(text):
                        return None
                    return text
            return None
        
        all_samples.extend(sample_from_hf(
            name="WildChat",
            dataset_id="allenai/WildChat",
            config="default",
            text_key=None,
            text_extractor=wildchat_extractor,
            filter_kwargs={'min_words': 300, 'max_words': 3000, 'min_paragraphs': 3, 'strict': True},
            samples_wanted=samples_per_source,
            max_batches=20,
            label=1
        ))
    else:
        print("\n⚠️  No HF_TOKEN set - skipping authenticated datasets (LMSYS, WildChat)")
        print("   Set HF_TOKEN environment variable or run: huggingface-cli login")
    
    # =========================================================================
    # KAGGLE SOURCES (if credentials available)
    # =========================================================================
    
    if KAGGLE_AVAILABLE:
        print("\n📦 Kaggle credentials found - sampling Kaggle datasets...")
        kaggle_samples = sample_kaggle_datasets(samples_per_source)
        all_samples.extend(kaggle_samples)
    else:
        print("\n⚠️  Kaggle not configured - skipping PERSUADE and AI Essays")
        print("   Run: pip install kaggle && place kaggle.json in ~/.kaggle/")
    
    return all_samples


def sample_kaggle_datasets(samples_wanted=2):
    """Sample from Kaggle datasets (PERSUADE and AI Essays) using Python API."""
    import tempfile
    import zipfile
    
    samples = []
    
    # Try to import pandas for CSV reading
    try:
        import pandas as pd
    except ImportError:
        print("  ⚠️ pandas not available for Kaggle sampling")
        return samples
    
    # Try to import kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"  ⚠️ Kaggle API error: {e}")
        return samples
    
    temp_dir = Path(tempfile.mkdtemp())
    
    # PERSUADE dataset (human student essays)
    print(f"\n{'='*60}")
    print(f"📥 Sampling: PERSUADE (Kaggle)")
    print(f"   Dataset: nbroad/persaude-corpus-2")
    print(f"{'='*60}")
    
    try:
        # Download PERSUADE using Python API
        print("  Downloading...")
        api.dataset_download_files("nbroad/persaude-corpus-2", path=str(temp_dir), unzip=True)
        
        # Find the CSV file
        csv_files = list(temp_dir.glob("*.csv")) + list(temp_dir.glob("**/*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0], nrows=500)
            text_col = 'full_text' if 'full_text' in df.columns else 'text'
            
            found = 0
            for _, row in df.iterrows():
                if found >= samples_wanted:
                    break
                text = str(row.get(text_col, ''))
                if is_essay_like(text, min_words=200, max_words=3000, min_paragraphs=2):
                    metrics = analyze_text_structure(text)
                    samples.append({
                        'source': 'PERSUADE (Kaggle)',
                        'label': 0,
                        'label_name': 'Human',
                        'text': text,
                        'stats': {
                            'words': metrics['word_count'],
                            'paragraphs': metrics['paragraph_count'],
                            'formal_indicators': metrics['formality_score']
                        }
                    })
                    found += 1
                    print(f"  ✓ Found sample {found}/{samples_wanted} ({metrics['word_count']} words)")
            
            print(f"  📊 Found {found}/{samples_wanted}")
        else:
            print("  ⚠️ No CSV files found in download")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
    
    # AI Essays dataset
    print(f"\n{'='*60}")
    print(f"📥 Sampling: AI Essays (Kaggle)")
    print(f"   Dataset: shanegerami/ai-vs-human-text")
    print(f"{'='*60}")
    
    try:
        print("  Downloading...")
        api.dataset_download_files("shanegerami/ai-vs-human-text", path=str(temp_dir), unzip=True)
        
        csv_files = list(temp_dir.glob("*.csv")) + list(temp_dir.glob("**/*.csv"))
        # Get the newest CSV (in case multiple from both datasets)
        if csv_files:
            # Try to find the AI essays specific file
            ai_csv = [f for f in csv_files if 'ai' in f.name.lower() or 'human' in f.name.lower()]
            csv_file = ai_csv[0] if ai_csv else csv_files[-1]
            
            # Read the full dataset to get both human and AI samples
            # Human samples are at the beginning (generated=0), AI at the end (generated=1)
            df = pd.read_csv(csv_file)
            text_col = 'text' if 'text' in df.columns else df.columns[0]
            label_col = 'generated' if 'generated' in df.columns else 'label'
            
            # Split into human and AI subsets
            human_df = df[df[label_col] == 0].head(500)  # First 500 human
            ai_df = df[df[label_col] == 1].head(500)      # First 500 AI
            
            print(f"  Dataset has {len(df[df[label_col] == 0])} human, {len(df[df[label_col] == 1])} AI samples")
            
            # Sample human essays
            human_found = 0
            ai_found = 0
            
            # Sample from human subset
            for _, row in human_df.iterrows():
                if human_found >= samples_wanted:
                    break
                text = str(row.get(text_col, ''))
                if is_essay_like(text, min_words=200, max_words=3000, min_paragraphs=2):
                    metrics = analyze_text_structure(text)
                    samples.append({
                        'source': 'AI Essays Human (Kaggle)',
                        'label': 0,
                        'label_name': 'Human',
                        'text': text,
                        'stats': {
                            'words': metrics['word_count'],
                            'paragraphs': metrics['paragraph_count'],
                            'formal_indicators': metrics['formality_score']
                        }
                    })
                    human_found += 1
                    print(f"  ✓ Found Human sample {human_found}/{samples_wanted}")
            
            # Sample from AI subset
            for _, row in ai_df.iterrows():
                if ai_found >= samples_wanted:
                    break
                text = str(row.get(text_col, ''))
                
                if is_essay_like(text, min_words=200, max_words=3000, min_paragraphs=2):
                    metrics = analyze_text_structure(text)
                    samples.append({
                        'source': 'AI Essays (Kaggle)',
                        'label': 1,
                        'label_name': 'AI',
                        'text': text,
                        'stats': {
                            'words': metrics['word_count'],
                            'paragraphs': metrics['paragraph_count'],
                            'formal_indicators': metrics['formality_score']
                        }
                    })
                    ai_found += 1
                    print(f"  ✓ Found AI sample {ai_found}/{samples_wanted}")
            
            print(f"  📊 Found Human: {human_found}, AI: {ai_found}")
        else:
            print("  ⚠️ No CSV files found")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
    
    # Cleanup temp dir
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass
    
    return samples


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample data for quality inspection")
    parser.add_argument("--samples", type=int, default=2,
                        help="Samples per source (default: 2)")
    parser.add_argument("--output", type=str, default="samples.txt",
                        help="Output file for full samples (default: samples.txt)")
    args = parser.parse_args()
    
    print("🔍 Data Sampling Script - Quality Check")
    print(f"   Collecting {args.samples} samples per source...")
    print(f"   HF Token: {'✓ SET' if HF_TOKEN else '✗ NOT SET'}")
    print(f"   Kaggle:   {'✓ CONFIGURED' if KAGGLE_AVAILABLE else '✗ NOT CONFIGURED'}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    samples = sample_all(samples_per_source=args.samples)
    
    display_samples(samples)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    human = [s for s in samples if s.get('label') == 0]
    ai = [s for s in samples if s.get('label') == 1]
    
    print(f"   Human samples: {len(human)}")
    print(f"   AI samples: {len(ai)}")
    print(f"   Total: {len(samples)}")
    
    # Save full samples to file
    if args.output and samples:
        save_full_samples(samples, args.output)
    
    print("\n✅ Done! Review the samples above and in the output file.")


if __name__ == "__main__":
    main()
