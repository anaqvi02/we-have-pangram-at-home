import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import Config
from src.data.indexing import VectorIndexer
import gc
import torch.amp as amp

class HardNegativeMiner:
    def __init__(self, model, indexer: VectorIndexer):
        self.model = model
        self.indexer = indexer
        self.device = Config.DEVICE
        self.use_amp = self.device == "cuda"
        self.autocast_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    def mine(self, human_dataset, batch_size=Config.BATCH_SIZE * 8, top_k=1, max_negatives=50000):
        """
        1. Run inference on human_dataset.
        2. Find False Positives (predicted as AI).
        3. Retrieve nearest AI neighbors for those FPs.
        4. Return new training pairs.
        """
        print(f"--- Starting Mining Phase on {len(human_dataset)} human samples ---")
        print(f"--- Mining Batch Size: {batch_size} | Max Hard Negatives: {max_negatives} ---")
        
        self.model.eval()
        hard_negatives = [] # List of text strings
        
        # Create a simple loader for the human text
        # We assume human_dataset[i] returns {'text': str, 'label': 0}
        
        loader = DataLoader(
            human_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        with torch.no_grad():
            with amp.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_amp):
                for batch in tqdm(loader, desc="Mining Inference"):
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
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
                    
                    # Early Exit Strategy
                    if len(hard_negatives) >= max_negatives:
                        print(f"Reached limit of {max_negatives} hard negatives. Stopping early.")
                        break
                    
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
