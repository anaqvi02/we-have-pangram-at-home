"""
Essay Benchmark Evaluation Script

Evaluates a trained Pangram detector on essay-focused datasets that were NOT
used during training, providing a fair out-of-distribution benchmark.

Benchmark Datasets:
    HUMAN:
    - Persuade 2.0 Corpus: Student argumentative essays (Kaggle competition)
    - Hewlett Foundation ASAP: Student essays from automated scoring competition
    
    AI:
    - GPT-wiki-intro: GPT-2 generated Wikipedia introductions
    - HC3 (Human-ChatGPT Comparison Corpus): ChatGPT responses

Usage:
    # Quick benchmark (default 1000 samples per source)
    python scripts/eval_essays.py
    
    # Custom sample count
    python scripts/eval_essays.py --samples 500
    
    # Specific model path
    python scripts/eval_essays.py --model_path /path/to/checkpoint
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.model.detector import PangramDetector

# =============================================================================
# BENCHMARK DATASETS (not used in training)
# =============================================================================

BENCHMARK_SOURCES = {
    # Human essay sources
    'human': [
        {
            'name': 'HC3-Human',
            'hf_path': 'Hello-SimpleAI/HC3',
            'hf_subset': 'all',
            'split': 'train',
            'text_field': 'human_answers',  # List of human answers
            'description': 'Human answers from HC3 corpus (comparison with ChatGPT)',
            'is_list_field': True,  # Text field contains a list
        },
        {
            'name': 'Reddit-Writing',
            'hf_path': 'webis/tldr-17',
            'hf_subset': None,
            'split': 'train',
            'text_field': 'content', 
            'description': 'Reddit writing prompts and responses',
            'filter_fn': lambda x: len(x.get('content', '').split()) > 150,  # Only longer posts
        },
    ],
    # AI-generated sources
    'ai': [
        {
            'name': 'HC3-ChatGPT',
            'hf_path': 'Hello-SimpleAI/HC3',
            'hf_subset': 'all',
            'split': 'train', 
            'text_field': 'chatgpt_answers',  # List of ChatGPT answers
            'description': 'ChatGPT answers from HC3 corpus',
            'is_list_field': True,
        },
        {
            'name': 'GPT-Wiki-Intro',
            'hf_path': 'aadityaubhat/GPT-wiki-intro',
            'hf_subset': None,
            'split': 'train',
            'text_field': 'generated_intro',
            'description': 'GPT-3 generated Wikipedia introductions',
        },
    ],
}


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_model(model_path: str) -> PangramDetector:
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
        all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)


def load_benchmark_source(source_config: dict, max_samples: int) -> List[str]:
    """Load samples from a benchmark source."""
    from datasets import load_dataset
    
    name = source_config['name']
    print(f"   Loading {name}...")
    
    try:
        # Load dataset with streaming to avoid huge downloads
        if source_config.get('hf_subset'):
            ds = load_dataset(
                source_config['hf_path'],
                source_config['hf_subset'],
                split=source_config['split'],
                streaming=True,
                trust_remote_code=True
            )
        else:
            ds = load_dataset(
                source_config['hf_path'],
                split=source_config['split'],
                streaming=True,
                trust_remote_code=True
            )
        
        texts = []
        text_field = source_config['text_field']
        is_list_field = source_config.get('is_list_field', False)
        filter_fn = source_config.get('filter_fn', lambda x: True)
        
        for sample in ds:
            if len(texts) >= max_samples:
                break
            
            # Apply filter if present
            if not filter_fn(sample):
                continue
            
            # Get text (handle list fields like HC3)
            if is_list_field:
                answers = sample.get(text_field, [])
                if answers:
                    # Take first answer from list
                    text = answers[0] if isinstance(answers, list) else answers
                else:
                    continue
            else:
                text = sample.get(text_field, '')
            
            # Basic quality filter
            if not text or len(text.strip()) < 100:
                continue
            
            texts.append(text.strip())
        
        print(f"      ✓ Loaded {len(texts):,} samples from {name}")
        return texts
        
    except Exception as e:
        print(f"      ✗ Error loading {name}: {e}")
        return []


def evaluate_texts(
    detector: PangramDetector,
    texts: List[str],
    labels: List[int],
    batch_size: int = 32,
    desc: str = "Evaluating"
) -> Dict:
    """Evaluate model on a set of texts with known labels."""
    
    if len(texts) == 0:
        return {'error': 'No samples to evaluate'}
    
    print(f"   Running inference on {len(texts):,} samples...")
    predictions, probabilities = predict_batch(detector, texts, batch_size)
    labels = np.array(labels)
    
    # Calculate metrics
    metrics = {
        'num_samples': len(texts),
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
    }
    
    # ROC-AUC
    if len(np.unique(labels)) > 1:
        metrics['roc_auc'] = float(roc_auc_score(labels, probabilities))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive_rate'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
    
    return metrics


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_benchmark(
    model_path: str,
    samples_per_source: int = 1000,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
):
    """Run the essay benchmark evaluation."""
    
    print("=" * 70)
    print("ESSAY BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Samples per source: {samples_per_source:,}")
    print()
    
    # Load model
    detector = load_model(model_path)
    Config.print_hardware_status()
    
    results = {
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'samples_per_source': samples_per_source,
            'batch_size': batch_size,
        },
        'sources': {},
    }
    
    all_texts = []
    all_labels = []
    
    # Load human sources
    print("\n📖 Loading HUMAN essay sources...")
    for source in BENCHMARK_SOURCES['human']:
        texts = load_benchmark_source(source, samples_per_source)
        if texts:
            all_texts.extend(texts)
            all_labels.extend([0] * len(texts))  # 0 = human
            results['sources'][source['name']] = {
                'count': len(texts),
                'label': 'human',
                'description': source['description'],
            }
    
    # Load AI sources
    print("\n🤖 Loading AI-GENERATED sources...")
    for source in BENCHMARK_SOURCES['ai']:
        texts = load_benchmark_source(source, samples_per_source)
        if texts:
            all_texts.extend(texts)
            all_labels.extend([1] * len(texts))  # 1 = AI
            results['sources'][source['name']] = {
                'count': len(texts),
                'label': 'ai',
                'description': source['description'],
            }
    
    # Summary
    human_count = sum(1 for l in all_labels if l == 0)
    ai_count = sum(1 for l in all_labels if l == 1)
    print(f"\n📊 Dataset Summary:")
    print(f"   Human samples: {human_count:,}")
    print(f"   AI samples: {ai_count:,}")
    print(f"   Total: {len(all_texts):,}")
    
    if len(all_texts) == 0:
        print("\n❌ No samples loaded! Check dataset availability.")
        return None
    
    # Run evaluation
    print("\n" + "=" * 50)
    print("RUNNING EVALUATION")
    print("=" * 50)
    
    overall_metrics = evaluate_texts(
        detector, all_texts, all_labels, 
        batch_size=batch_size,
        desc="Overall"
    )
    results['overall'] = overall_metrics
    
    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"""
📊 Overall Performance:
   Samples:   {overall_metrics['num_samples']:,}
   Accuracy:  {overall_metrics['accuracy']*100:.1f}%
   Precision: {overall_metrics['precision']*100:.1f}%
   Recall:    {overall_metrics['recall']*100:.1f}%
   F1 Score:  {overall_metrics['f1']:.4f}
   ROC-AUC:   {overall_metrics.get('roc_auc', 'N/A')}
""")
    
    # Confusion matrix breakdown
    if 'confusion_matrix' in overall_metrics:
        cm = overall_metrics['confusion_matrix']
        if len(cm) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            print(f"📋 Confusion Matrix:")
            print(f"   True Negatives (Human→Human):  {tn:,}")
            print(f"   False Positives (Human→AI):    {fp:,}")
            print(f"   False Negatives (AI→Human):    {fn:,}")
            print(f"   True Positives (AI→AI):        {tp:,}")
            print()
            print(f"   False Positive Rate: {overall_metrics.get('false_positive_rate', 0)*100:.1f}%")
            print(f"   True Positive Rate:  {overall_metrics.get('true_positive_rate', 0)*100:.1f}%")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Config.CHECKPOINT_DIR
    
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"essay_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pangram detector on held-out essay benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark (1000 samples per source)
  python scripts/eval_essays.py
  
  # More samples for better statistics
  python scripts/eval_essays.py --samples 5000
  
  # Specific model
  python scripts/eval_essays.py --model_path /path/to/checkpoint
"""
    )
    
    parser.add_argument(
        "--model_path", type=str,
        default=str(Config.CHECKPOINT_DIR / "pangram_final" / "pangram_best"),
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of samples per source (default: 1000)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        model_path=args.model_path,
        samples_per_source=args.samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
