import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import numpy as np
import argparse
from tqdm import tqdm

from src.config import Config
from src.model.detector import PangramDetector
from src.data.loader import StreamingTextDataset
# Reuse main's mock data for simplicity if needed, or import
from train import create_mock_data

def calculate_metrics(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print("Running Evaluation Inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].numpy()
            
            outputs = model(input_ids, attention_mask)
            prob_ai = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            
            all_preds.extend(prob_ai)
            all_labels.extend(labels)
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate FPR @ 95% Recall (True Positive Rate)
    # AI is Positive Class (1)
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    
    # Find the threshold where TPR is >= 0.95
    # roc_curve returns increasing thresholds? No, decreasing typically. 
    # But usually sklearn sorts them.
    
    # We want index where tpr >= 0.95
    # Warning: if model is terrible, we might not reach 0.95
    idx = np.argmax(tpr >= 0.95)
    
    fpr_at_95 = fpr[idx]
    threshold_at_95 = thresholds[idx]
    
    print(f"\n--- Results ---")
    print(f"FPR @ 95% Recall: {fpr_at_95:.4f} ({fpr_at_95*100:.2f}%)")
    print(f"Threshold: {threshold_at_95:.4f}")
    
    return fpr_at_95

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--use_mock_data", action="store_true")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    detector = PangramDetector.load(args.model_path)
    
    if args.use_mock_data:
        data = create_mock_data(200)
        dataset = StreamingTextDataset(data, detector.tokenizer)
    else:
        raise NotImplementedError("Real evaluation data loading not implemented.")
        
    calculate_metrics(detector.model, dataset)

if __name__ == "__main__":
    main()
