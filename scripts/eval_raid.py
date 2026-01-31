"""
RAID Benchmark Evaluation Script

Evaluates a trained Pangram detector on the RAID benchmark for AI text detection.

RAID (Robust AI Detection) is a comprehensive benchmark with:
- 11 generative models (GPT-4, ChatGPT, Llama, Mistral, etc.)
- 11 domains (news, wiki, reddit, recipes, etc.)
- 11 adversarial attacks (homoglyph, paraphrase, synonym, etc.)
- 2 decoding strategies (greedy, sampling)

Usage:
    # Evaluate the best checkpoint
    python scripts/eval_raid.py
    
    # Evaluate a specific checkpoint
    python scripts/eval_raid.py --model_path /path/to/checkpoint
    
    # Limit samples for quick testing
    python scripts/eval_raid.py --max_samples 1000
    
    # Evaluate specific domains/models
    python scripts/eval_raid.py --domains news wiki --models chatgpt gpt4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.model.detector import PangramDetector

# =============================================================================
# RAID DATASET CONFIGURATION
# =============================================================================

RAID_MODELS = [
    'chatgpt', 'gpt4', 'gpt3', 'gpt2', 'llama-chat', 
    'mistral', 'mistral-chat', 'mpt', 'mpt-chat', 'cohere', 'cohere-chat'
]

RAID_DOMAINS = [
    'abstracts', 'books', 'code', 'czech', 'german', 
    'news', 'poetry', 'recipes', 'reddit', 'reviews', 'wiki'
]

RAID_ATTACKS = [
    'homoglyph', 'number', 'article_deletion', 'insert_paragraphs',
    'perplexity_misspelling', 'upper_lower', 'whitespace', 
    'zero_width_space', 'synonym', 'paraphrase', 'alternative_spelling'
]

# English-only domains (exclude czech/german)
ENGLISH_DOMAINS = [
    'abstracts', 'books', 'code', 'news', 'poetry', 
    'recipes', 'reddit', 'reviews', 'wiki'
]

# Essay-focused domains (prose-based, formal writing)
# Excludes: code (programming), poetry (verse), recipes (structured), reddit (informal/short)
ESSAY_DOMAINS = [
    'abstracts',  # Academic abstracts - formal prose
    'books',      # Book excerpts - narrative prose
    'news',       # News articles - journalistic essays
    'reviews',    # Reviews - opinion essays
    'wiki',       # Wikipedia - encyclopedic essays
]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_model(model_path: str) -> Tuple[PangramDetector, torch.nn.Module]:
    """Load the trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    detector = PangramDetector.load(model_path)
    detector.model.eval()
    
    return detector


def predict_batch(
    detector: PangramDetector,
    texts: List[str],
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a batch of texts.
    
    Returns:
        predictions: Binary predictions (0=human, 1=AI)
        probabilities: Probability of AI class
    """
    all_preds = []
    all_probs = []
    
    device = detector.config.DEVICE
    tokenizer = detector.tokenizer
    model = detector.model
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of AI class
    
    return np.array(all_preds), np.array(all_probs)


def evaluate_subset(
    detector: PangramDetector,
    dataset,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    desc: str = "Evaluating"
) -> Dict:
    """Evaluate on a subset of the dataset."""
    
    texts = []
    labels = []
    metadata = []
    
    # Collect samples
    for i, sample in enumerate(tqdm(dataset, desc=f"Loading {desc}")):
        if max_samples and i >= max_samples:
            break
        
        text = sample.get('generation', '')
        if not text or len(text.strip()) < 50:
            continue
        
        texts.append(text)
        # RAID: if model is None, it's human-written
        is_ai = sample.get('model') is not None
        labels.append(1 if is_ai else 0)
        metadata.append({
            'model': sample.get('model'),
            'domain': sample.get('domain'),
            'attack': sample.get('attack'),
            'decoding': sample.get('decoding'),
        })
    
    if len(texts) == 0:
        return {'error': 'No valid samples found'}
    
    # Run inference
    print(f"Running inference on {len(texts):,} samples...")
    predictions, probabilities = predict_batch(detector, texts, batch_size)
    labels = np.array(labels)
    
    # Calculate metrics
    metrics = {
        'num_samples': len(texts),
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
    }
    
    # ROC-AUC (only if both classes present)
    if len(np.unique(labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(labels, probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    # True/False positive/negative rates
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics


def evaluate_by_dimension(
    detector: PangramDetector,
    dataset,
    dimension: str,  # 'model', 'domain', or 'attack'
    values: List[str],
    max_samples_per_value: int = 1000,
    batch_size: int = 32,
) -> Dict[str, Dict]:
    """Evaluate performance broken down by a specific dimension."""
    
    results = {}
    
    for value in tqdm(values, desc=f"Evaluating by {dimension}"):
        # Filter dataset for this value
        subset = dataset.filter(lambda x: x.get(dimension) == value)
        
        if len(subset) == 0:
            results[value] = {'error': 'No samples found'}
            continue
        
        metrics = evaluate_subset(
            detector, 
            subset, 
            max_samples=max_samples_per_value,
            batch_size=batch_size,
            desc=value
        )
        results[value] = metrics
    
    return results


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_full_evaluation(
    model_path: str,
    max_samples: Optional[int] = None,
    domains: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
    essays_only: bool = True,
):
    """Run comprehensive RAID benchmark evaluation."""
    
    print("=" * 70)
    print("RAID BENCHMARK EVALUATION" + (" (Essays Only)" if essays_only else ""))
    print("=" * 70)
    
    # Load model
    detector = load_model(model_path)
    Config.print_hardware_status()
    
    # Load RAID dataset
    print("\n📥 Loading RAID dataset...")
    raid = load_dataset("liamdugan/raid", split="train")
    print(f"   Total samples: {len(raid):,}")
    
    # Filter to specified domains/models
    if domains:
        raid = raid.filter(lambda x: x.get('domain') in domains)
        print(f"   After domain filter: {len(raid):,}")
    elif essays_only:
        raid = raid.filter(lambda x: x.get('domain') in ESSAY_DOMAINS)
        print(f"   After essay-only filter: {len(raid):,}")
        print(f"   (Domains: {', '.join(ESSAY_DOMAINS)})")
    
    if models:
        # Include both AI samples from specified models AND human samples (model=None)
        raid = raid.filter(lambda x: x.get('model') in models or x.get('model') is None)
        print(f"   After model filter: {len(raid):,}")
    
    results = {
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'max_samples': max_samples,
            'domains': domains or (ESSAY_DOMAINS if essays_only else RAID_DOMAINS),
            'models': models or RAID_MODELS,
            'batch_size': batch_size,
            'essays_only': essays_only,
        }
    }
    
    # Overall evaluation
    print("\n" + "=" * 50)
    print("OVERALL EVALUATION")
    print("=" * 50)
    
    overall = evaluate_subset(
        detector, raid, 
        max_samples=max_samples,
        batch_size=batch_size,
        desc="Overall"
    )
    results['overall'] = overall
    
    print(f"\n📊 Overall Results:")
    print(f"   Samples: {overall['num_samples']:,}")
    print(f"   Accuracy: {overall['accuracy']:.4f}")
    print(f"   Precision: {overall['precision']:.4f}")
    print(f"   Recall: {overall['recall']:.4f}")
    print(f"   F1 Score: {overall['f1']:.4f}")
    if 'roc_auc' in overall:
        print(f"   ROC-AUC: {overall['roc_auc']:.4f}")
    
    # By domain
    print("\n" + "=" * 50)
    print("EVALUATION BY DOMAIN")
    print("=" * 50)
    
    eval_domains = domains or (ESSAY_DOMAINS if essays_only else RAID_DOMAINS)
    by_domain = evaluate_by_dimension(
        detector, raid, 'domain', eval_domains,
        max_samples_per_value=max_samples // len(eval_domains) if max_samples else 2000,
        batch_size=batch_size
    )
    results['by_domain'] = by_domain
    
    print(f"\n📊 Results by Domain:")
    print(f"   {'Domain':<15} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print(f"   {'-'*45}")
    for domain, metrics in sorted(by_domain.items()):
        if 'error' in metrics:
            print(f"   {domain:<15} {'N/A':>8}")
        else:
            auc = metrics.get('roc_auc', 0)
            print(f"   {domain:<15} {metrics['accuracy']:>8.4f} {metrics['f1']:>8.4f} {auc:>8.4f}")
    
    # By model (AI generator)
    print("\n" + "=" * 50)
    print("EVALUATION BY AI MODEL")
    print("=" * 50)
    
    eval_models = models or RAID_MODELS
    # Filter to AI samples only for by-model evaluation
    ai_only = raid.filter(lambda x: x.get('model') is not None)
    by_model = evaluate_by_dimension(
        detector, ai_only, 'model', eval_models,
        max_samples_per_value=max_samples // len(eval_models) if max_samples else 2000,
        batch_size=batch_size
    )
    results['by_model'] = by_model
    
    print(f"\n📊 Detection Rate by AI Model:")
    print(f"   {'Model':<15} {'Detected':>10} {'Samples':>10}")
    print(f"   {'-'*40}")
    for model, metrics in sorted(by_model.items(), key=lambda x: x[1].get('recall', 0), reverse=True):
        if 'error' in metrics:
            print(f"   {model:<15} {'N/A':>10}")
        else:
            # For AI-only, recall = detection rate
            print(f"   {model:<15} {metrics['recall']*100:>9.1f}% {metrics['num_samples']:>10,}")
    
    # By attack (for adversarial robustness)
    print("\n" + "=" * 50)
    print("EVALUATION BY ADVERSARIAL ATTACK")
    print("=" * 50)
    
    # Filter to samples with attacks
    attacked = raid.filter(lambda x: x.get('attack') is not None)
    if len(attacked) > 0:
        by_attack = evaluate_by_dimension(
            detector, attacked, 'attack', RAID_ATTACKS,
            max_samples_per_value=max_samples // len(RAID_ATTACKS) if max_samples else 1000,
            batch_size=batch_size
        )
        results['by_attack'] = by_attack
        
        print(f"\n📊 Robustness to Adversarial Attacks:")
        print(f"   {'Attack':<25} {'Detected':>10} {'Accuracy':>10}")
        print(f"   {'-'*50}")
        for attack, metrics in sorted(by_attack.items(), key=lambda x: x[1].get('recall', 0), reverse=True):
            if 'error' in metrics:
                print(f"   {attack:<25} {'N/A':>10}")
            else:
                print(f"   {attack:<25} {metrics['recall']*100:>9.1f}% {metrics['accuracy']*100:>9.1f}%")
    else:
        print("   No adversarial samples found in filtered dataset")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Config.CHECKPOINT_DIR
    
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"raid_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
🎯 Overall Performance:
   Accuracy: {overall['accuracy']*100:.1f}%
   F1 Score: {overall['f1']:.4f}
   ROC-AUC:  {overall.get('roc_auc', 'N/A')}

📈 Key Insights:
   - Tested on {overall['num_samples']:,} samples from RAID benchmark
   - {len([d for d in by_domain.values() if d.get('accuracy', 0) > 0.9])} domains with >90% accuracy
   - Best detected model: {max(by_model.items(), key=lambda x: x[1].get('recall', 0))[0] if by_model else 'N/A'}
   - Hardest to detect: {min(by_model.items(), key=lambda x: x[1].get('recall', 1))[0] if by_model else 'N/A'}
""")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pangram detector on RAID benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1000 samples)
  python scripts/eval_raid.py --max_samples 1000
  
  # Full evaluation on best model
  python scripts/eval_raid.py --model_path /mnt/weightsandotherstuff/pangram_final/pangram_best
  
  # Evaluate specific domains
  python scripts/eval_raid.py --domains news wiki reddit
  
  # Evaluate against specific AI models
  python scripts/eval_raid.py --models chatgpt gpt4 llama-chat
"""
    )
    
    parser.add_argument(
        "--model_path", type=str,
        default=str(Config.CHECKPOINT_DIR / "pangram_final" / "pangram_best"),
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum samples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        choices=RAID_DOMAINS,
        help="Specific domains to evaluate"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=RAID_MODELS,
        help="Specific AI models to evaluate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results"
    )
    parser.add_argument(
        "--all-domains", action="store_true",
        help="Evaluate on all domains (not just essays). Default is essay-only."
    )
    
    args = parser.parse_args()
    
    run_full_evaluation(
        model_path=args.model_path,
        max_samples=args.max_samples,
        domains=args.domains,
        models=args.models,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        essays_only=not args.all_domains,
    )


if __name__ == "__main__":
    main()
