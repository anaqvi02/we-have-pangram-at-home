import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import Config
from src.data.indexing import VectorIndexer
import gc

class HardNegativeMiner:
    def __init__(self, model, indexer: VectorIndexer):
        self.model = model
        self.indexer = indexer
        self.device = Config.DEVICE

    def mine(self, human_dataset, batch_size=Config.BATCH_SIZE * 4, top_k=1):
        """
        1. Run inference on human_dataset.
        2. Find False Positives (predicted as AI).
        3. Retrieve nearest AI neighbors for those FPs.
        4. Return new training pairs.
        """
        print(f"--- Starting Mining Phase on {len(human_dataset)} human samples ---")
        
        self.model.eval()
        hard_negatives = [] # List of text strings
        
        # Create a simple loader for the human text
        # We assume human_dataset[i] returns {'text': str, 'label': 0}
        # We need a collate_fn to handle tokenization on the fly if not pre-tokenized
        # But for mining, we might just want to process raw text if possible, 
        # but the model needs tensors. 
        # For simplicity, we assume dataset returns tensors valid for the model.
        
        loader = DataLoader(human_dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Mining Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                texts = batch['text']
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # AI is label 1. Human is label 0.
                # If prob(AI) > 0.5 for a Human text, it's a False Positive (Hard Negative).
                ai_probs = probs[:, 1]
                fp_indices = torch.where(ai_probs > 0.5)[0]
                
                for idx in fp_indices:
                    hard_negatives.append(texts[idx])
                    
        print(f"Found {len(hard_negatives)} Hard Negatives (False Positives).")
        
        if not hard_negatives:
            return []
            
        # 2. Retrieve Static Mirrors
        print("Retrieving Static Mirrors from AI Index...")
        # We might need to batch this if hard_negatives is huge, but usually it's manageable
        
        # Encode hard negatives to get query vectors
        # Note: VectorIndexer uses sentence-transformers, self.model is DeBERTa classifier.
        # They are different models!
        
        retrieved_mirrors = self.indexer.search(hard_negatives, top_k=top_k)
        
        new_batch = []
        for i, hn_text in enumerate(hard_negatives):
            # Add the Hard Negative (Human) with label 0
            new_batch.append({'text': hn_text, 'label': 0})
            
            # Add the Retrieved Mirrors (AI) with label 1
            mirrors = retrieved_mirrors[i]
            for mirror_text in mirrors:
                new_batch.append({'text': mirror_text, 'label': 1})
                
        # Clean up memory
        if self.device == 'mps':
            torch.mps.empty_cache()
        gc.collect()
        
        print(f"Generated {len(new_batch)} new training samples from mirrors.")
        return new_batch
