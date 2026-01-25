import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
import numpy as np
from torch.utils.data import DataLoader, Dataset

class BenchmarkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def run_benchmark(csv_path, model_path=None):
    print(f"--- Pangram Benchmark ---")
    print(f"Data: {csv_path}")
    
    # Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check columns
    if 'text' not in df.columns or 'label' not in df.columns:
        print("Error: CSV must contain 'text' and 'label' columns.")
        print(f"Found: {df.columns}")
        return

    print(f"Loaded {len(df)} samples.")
    print(df['label'].value_counts())

    # Load Model
    device = Config.DEVICE
    print(f"Device: {device}")
    
    if model_path is None:
        # Try finding the best model, then latest, then default
        best_path = Config.CHECKPOINT_DIR / "pangram_best"
        latest_path = Config.CHECKPOINT_DIR / "pangram_latest"
        
        if best_path.exists():
            model_path = best_path
            print("Using Best Checkpoint.")
        elif latest_path.exists():
            model_path = latest_path
            print("Using Latest Checkpoint.")
        else:
            model_path = Config.MODEL_NAME
            print(f"Using Base Model ({model_path}). Warning: Results will be random if not trained!")

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Prepare Data
    dataset = BenchmarkDataset(df['text'].tolist(), df['label'].tolist(), tokenizer, max_length=Config.MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE * 4, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of AI (Label 1)
            all_labels.extend(labels.cpu().numpy())

    # Calculate Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    correct = (all_preds == all_labels)
    accuracy = correct.mean()

    # Confusion Matrix items
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False Positive Rate (Human flagged as AI)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n--- Results ---")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"FPR (False Accusations): {fpr:.4f} ({fpr*100:.2f}%)")
    
    print("\n--- Breakdown ---")
    print(f"True Positives (AI Caught):      {tp}")
    print(f"True Negatives (Humans Safe):    {tn}")
    print(f"False Positives (Humans Flagged): {fp}")
    print(f"False Negatives (AI Missed):     {fn}")

    # Output to file
    results_path = Config.PROJECT_ROOT / "benchmark_results.csv"
    res_df = df.copy()
    res_df['prediction'] = all_preds
    res_df['ai_probability'] = all_probs
    res_df['correct'] = correct
    res_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to {results_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="tests/benchmark_data.csv")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()
    
    run_benchmark(args.csv, args.model)
