import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
from src.config import Config
from src.mining.miner import HardNegativeMiner
from src.data.loader import StreamingTextDataset
import gc

class PangramTrainer:
    def __init__(self, model, tokenizer, indexer):
        self.model = model
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.miner = HardNegativeMiner(model, indexer)
        self.device = Config.DEVICE
        
    def train_epoch(self, dataset, epoch_idx):
        print(f"--- Epoch {epoch_idx} Training ---")
        self.model.train()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        
        num_training_steps = len(dataloader)
        progress_bar = tqdm(range(num_training_steps))
        
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            
            # Gradient Accumulation
            loss = loss / Config.GRAD_ACCUMULATION
            loss.backward()
            
            if (step + 1) % Config.GRAD_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(Config.GRAD_ACCUMULATION)
                
                # MPS Memory Management
                if self.device == 'mps' and step % 100 == 0:
                    torch.mps.empty_cache()
                    
        return dataset # Return dataset for continuity

    def evaluate(self, dataset):
        """Run evaluation on a held-out dataset."""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
        
        return avg_loss, accuracy

    def run_curriculum(self, train_dataset, human_eval_pool, val_dataset=None, epochs=Config.NUM_EPOCHS, start_epoch=0):
        """
        Curriculum Loop with Validation and Best Model Saving.
        """
        print(f"--- Starting Curriculum Training for {epochs} Epochs (Starting from {start_epoch}) ---")
        current_train_data = train_dataset
        
        best_val_loss = float('inf')
        log_file = Config.PROJECT_ROOT / "training_log.csv"
        
        # Initialize log header
        if not log_file.exists():
            with open(log_file, "w") as f:
                f.write("epoch,train_loss,val_loss,val_acc,dataset_size\n")
        
        for epoch in range(start_epoch, epochs):
            # 1. Train
            self.train_epoch(current_train_data, epoch)
            
            # 2. Validate
            val_loss = 0.0
            val_acc = 0.0
            if val_dataset:
                print(f"--- Epoch {epoch} Validation ---")
                val_loss, val_acc = self.evaluate(val_dataset)
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
                # Save Best Model
                if val_loss < best_val_loss:
                    print(f"New Best Model! (Loss: {val_loss:.4f})")
                    best_val_loss = val_loss
                    best_dir = Config.CHECKPOINT_DIR / "pangram_best"
                    self.save_checkpoint(best_dir)

            # Log to CSV
            with open(log_file, "a") as f:
                 f.write(f"{epoch},0.0,{val_loss:.4f},{val_acc:.4f},{len(current_train_data)}\n")

            # 3. Mine
            print(f"--- Epoch {epoch} Mining ---")
            
            # We treat the 'human_eval_pool' as the source for mining hard negatives
            # User has ample compute: Scan the ENTIRE pool.
            import random
            pool_len = len(human_eval_pool)
            print(f"Scanning FULL Human Pool ({pool_len} samples) for Hard Negatives...")
            
            # No subsampling - take everything
            # Convert to list if it's not already a sequence to ensure stability
            if isinstance(human_eval_pool, list):
                 current_pool = human_eval_pool
            else:
                 # For map-style datasets or others, iterate/convert
                 current_pool = [human_eval_pool[i] for i in range(pool_len)]

            human_ds = StreamingTextDataset(current_pool, tokenizer=self.tokenizer)
            
            new_pairs = self.miner.mine(human_ds)
            
            if len(new_pairs) == 0:
                print("No hard negatives found. Stopping curriculum early.")
                break
                
            # 3. Augment
            # For the next epoch, we should optimally MIX the new hard pairs with the old data
            # or train ONLY on the hard pairs (Fine-tuning approach).
            # Pangram suggests a curriculum, implying we focus on the hard stuff.
            # We'll prepend the new data to the training set list.
            # 3. Augment
            # New optimization: Tokenize immediately and extend the tensor dataset
            if hasattr(current_train_data, 'extend'):
                print(f"Augmenting PretokenizedDataset with {len(new_pairs)} new samples...")
                texts = [p['text'] for p in new_pairs]
                labels = [p['label'] for p in new_pairs]
                
                # Tokenize new batch
                encodings = self.tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=Config.MAX_LENGTH,
                    return_tensors="pt"
                )
                
                # Create mini dataset
                # We need to import PretokenizedDataset or simple create a temporary one?
                # Actually we can just update the tensors directly if we didn't import the class, 
                # but better to assume the method 'extend' expects a compatible object.
                # Let's import PretokenizedDataset at top or just use the same logic here.
                
                # Optimization: We defined extend(self, other). So we need an object with .input_ids etc.
                from src.data.loader import PretokenizedDataset
                new_ds = PretokenizedDataset(
                    encodings['input_ids'],
                    encodings['attention_mask'],
                    torch.tensor(labels, dtype=torch.long)
                )
                
                current_train_data.extend(new_ds)
                print(f"Dataset Size grew to {len(current_train_data)}")
                
            elif isinstance(current_train_data.data_source, list):
                # Legacy path for tests
                current_train_data.data_source.extend(new_pairs)
                print(f"Dataset Size grew to {len(current_train_data)}")
            else:
                print("Warning: Dataset augmentation not supported for this type.")
            
            # --- Auto-Save Checkpoint ---
            checkpoint_dir = Config.CHECKPOINT_DIR / f"pangram_epoch_{epoch+1}"
            self.save_checkpoint(checkpoint_dir)
            
            # Also update 'latest'
            latest_dir = Config.CHECKPOINT_DIR / "pangram_latest"
            self.save_checkpoint(latest_dir)

    def save_checkpoint(self, path):
        """Robust saving with local fallback."""
        try:
            print(f"Saving checkpoint to {path}...")
            path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print("Checkpoint saved.")
        except Exception as e:
            print(f"❌ Failed to save checkpoint to {path}: {e}")
            
            # Fallback to local
            fallback_path = Config.PROJECT_ROOT / "local_backups" / path.name
            print(f"⚠️  Attempting fallback save to {fallback_path}...")
            try:
                fallback_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(fallback_path)
                self.tokenizer.save_pretrained(fallback_path)
                print(f"✅ Fallback save successful.")
            except Exception as e2:
                print(f"❌ FATAL: Fallback save also failed: {e2}")
