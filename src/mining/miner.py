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

    def mine(self, human_dataset, tokenizer, batch_size=Config.BATCH_SIZE * 8, top_k=1, max_negatives=50000):
        """Mine hard negatives.

        Strategy:
        - Run inference on human_dataset.
        - Score by P(AI) (higher means more likely a false positive).
        - Keep top-N by score (more stable than a fixed 0.5 threshold early in training).
        - Retrieve nearest AI neighbors for those FPs.
        - Return new training pairs.
        """
        print(f"--- Starting Mining Phase on {len(human_dataset)} human samples ---")
        print(f"--- Mining Batch Size: {batch_size} | Max Hard Negatives: {max_negatives} ---")

        self.model.eval()
        scored_hard_negatives = []  # list[(score: float, text: str)]
        
        num_workers = 4 if self.device == "cuda" else 0

        def collate_fn(features):
            texts = [f.get('text', '') for f in features]

            enc = tokenizer(
                texts,
                truncation=True,
                max_length=Config.MAX_LENGTH,
                padding=True,
                return_tensors='pt',
            )

            # Keep raw text for scoring + later retrieval.
            enc['text'] = texts
            return enc

        # Create a loader for the human text (batched tokenization + dynamic padding)
        loader = DataLoader(
            human_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=collate_fn,
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

                    # Score by P(AI). Higher score => more likely false positive.
                    ai_probs = probs[:, 1].detach().float().cpu().tolist()

                    for score, text in zip(ai_probs, texts):
                        scored_hard_negatives.append((float(score), text))

        if not scored_hard_negatives:
            return []

        scored_hard_negatives.sort(key=lambda x: x[0], reverse=True)
        scored_hard_negatives = scored_hard_negatives[:max_negatives]

        hard_negatives = [t for _, t in scored_hard_negatives]
        print(f"Selected {len(hard_negatives)} hard negatives (top by P(AI)).")
            
        # 2. Retrieve Static Mirrors
        print("Retrieving Static Mirrors from AI Index...")
        # We might need to batch this if hard_negatives is huge, but usually it's manageable
        
        # Encode hard negatives to get query vectors
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
