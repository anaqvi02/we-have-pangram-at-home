import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
from src.config import Config
from src.mining.miner import HardNegativeMiner
from src.data.loader import StreamingTextDataset, GrowableDataset
import gc
import torch.amp as amp

class PangramTrainer:
    def __init__(self, model, tokenizer, indexer):
        self.model = model
        self.tokenizer = tokenizer
        self.indexer = indexer
        self.miner = HardNegativeMiner(model, indexer)
        self.device = Config.DEVICE
        
        # Mixed Precision Setup
        self.use_amp = self.device == "cuda"
        self.autocast_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = amp.GradScaler('cuda') if (self.use_amp and self.autocast_dtype == torch.float16) else None
        
        if self.use_amp:
            print(f"✨ Mixed Precision Enabled: {self.autocast_dtype}")

        # Initialize Optimizer (State is persisted across epochs)
        # Prefer fused optimizer on CUDA when available (PyTorch 2+).
        use_fused = False
        if self.device == "cuda":
            try:
                use_fused = True
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=Config.LEARNING_RATE,
                    fused=True,
                )
            except TypeError:
                use_fused = False

        if not use_fused:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        
        # Gradient clipping threshold
        self.max_grad_norm = 1.0
        
    def train_epoch(self, dataset, epoch_idx, total_epochs):
        """
        Train for one epoch.
        
        Args:
            dataset: Training dataset
            epoch_idx: Current epoch number (0-indexed)
            total_epochs: Total number of epochs (for scheduler calculation)
        """
        print(f"--- Epoch {epoch_idx} Training ---")
        self.model.train()
        
        # Determine number of workers based on device
        num_workers = 4 if self.device == 'cuda' else 0
        
        def collate_fn(features):
            texts = [f['text'] for f in features]
            labels = torch.tensor([int(f['labels']) for f in features], dtype=torch.long)

            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=Config.MAX_LENGTH,
                padding=True,
                return_tensors='pt',
            )

            enc['labels'] = labels
            return enc

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=collate_fn,
        )
        
        # Recreate scheduler each epoch based on CURRENT dataset size
        # This is crucial for curriculum learning where dataset grows
        steps_per_epoch = len(dataloader) // Config.GRAD_ACCUMULATION
        remaining_epochs = total_epochs - epoch_idx
        num_training_steps = steps_per_epoch * remaining_epochs
        num_warmup_steps = int(steps_per_epoch * 0.1)  # 10% of first epoch
        
        print(f"  → Dataset size: {len(dataset):,} samples")
        print(f"  → Steps this epoch: {steps_per_epoch:,}")
        print(f"  → Scheduler: {num_training_steps} steps, {num_warmup_steps} warmup")
        
        scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch_idx}")
        running_loss = 0.0
        step_count = 0
        
        def _optimizer_step():
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            scheduler.step()
            self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            # Autocast Forward Pass
            with amp.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_amp):
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss / Config.GRAD_ACCUMULATION

            # Scaled Backward Pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * Config.GRAD_ACCUMULATION

            if (step + 1) % Config.GRAD_ACCUMULATION == 0:
                _optimizer_step()

                step_count += 1
                avg_loss = running_loss / step_count
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                progress_bar.update(1)

                # Memory Management
                if self.device == 'mps' and step % 100 == 0:
                    torch.mps.empty_cache()
                elif self.device == 'cuda' and step % 500 == 0:
                    torch.cuda.empty_cache()

        # Flush remainder microbatches (otherwise final partial accumulation is dropped)
        if (step + 1) % Config.GRAD_ACCUMULATION != 0:
            _optimizer_step()
            step_count += 1
        
        progress_bar.close()
        final_loss = running_loss / max(step_count, 1)
        print(f"  → Epoch {epoch_idx} complete. Avg Loss: {final_loss:.4f}")
        return final_loss

    def evaluate(self, dataset):
        """Run evaluation on a held-out dataset."""
        self.model.eval()

        num_workers = 4 if self.device == 'cuda' else 0

        def collate_fn(features):
            texts = [f['text'] for f in features]
            labels = torch.tensor([int(f['labels']) for f in features], dtype=torch.long)

            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=Config.MAX_LENGTH,
                padding=True,
                return_tensors='pt',
            )

            enc['labels'] = labels
            return enc

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            collate_fn=collate_fn,
        )
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            with amp.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.use_amp):
                for batch in tqdm(dataloader, desc="Evaluating"):
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)
                    
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
        
        Args:
            train_dataset: GrowableDataset that can be extended with mined samples
            human_eval_pool: HuggingFace Dataset for mining hard negatives
            val_dataset: Optional validation dataset
            epochs: Total number of epochs
            start_epoch: Epoch to resume from
        """
        print(f"--- Starting Curriculum Training for {epochs} Epochs (Starting from {start_epoch}) ---")
        
        best_val_loss = float('inf')
        log_file = Config.PROJECT_ROOT / "training_log.csv"
        
        # Initialize log header
        if not log_file.exists():
            with open(log_file, "w") as f:
                f.write("epoch,train_loss,val_loss,val_acc,dataset_size,mined_samples\n")
        
        for epoch in range(start_epoch, epochs):
            # 1. Train
            train_loss = self.train_epoch(train_dataset, epoch, epochs)
            
            # 2. Validate
            val_loss = 0.0
            val_acc = 0.0
            if val_dataset:
                print(f"--- Epoch {epoch} Validation ---")
                val_loss, val_acc = self.evaluate(val_dataset)
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
                # Save Best Model
                if val_loss < best_val_loss:
                    print(f"🏆 New Best Model! (Loss: {val_loss:.4f})")
                    best_val_loss = val_loss
                    best_dir = Config.CHECKPOINT_DIR / "pangram_best"
                    self.save_checkpoint(best_dir)

            # Get buffer size for logging
            mined_total = train_dataset.get_buffer_size() if hasattr(train_dataset, 'get_buffer_size') else 0
            
            # Log to CSV
            with open(log_file, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f},{len(train_dataset)},{mined_total}\n")

            # 3. Mine (skip on last epoch - no point mining if we won't train again)
            if epoch < epochs - 1:
                print(f"--- Epoch {epoch} Mining ---")
                
                # Subsample mining pool for efficiency
                import random
                from datasets import Dataset as HFDataset
                
                miner_pool_size = 60000
                pool_len = len(human_eval_pool)
                
                if pool_len > miner_pool_size:
                    print(f"  → Subsampling {miner_pool_size:,} from {pool_len:,} for mining...")
                    # Use HF Dataset's efficient selection
                    indices = random.sample(range(pool_len), miner_pool_size)
                    current_pool = human_eval_pool.select(indices)
                else:
                    current_pool = human_eval_pool
                
                # Wrap for miner (needs tokenizer)
                human_ds = StreamingTextDataset(current_pool, tokenizer=self.tokenizer)
                
                # Mine hard negatives
                new_pairs = self.miner.mine(human_ds, max_negatives=50000)
                
                if len(new_pairs) == 0:
                    print("  ⚠️ No hard negatives found. Stopping curriculum early.")
                    break
                
                # Extend training dataset with mined samples
                print(f"  → Adding {len(new_pairs):,} mined samples to training set...")
                train_dataset.extend(new_pairs)
                print(f"  → Dataset now has {len(train_dataset):,} samples")
            
            # --- Auto-Save Checkpoint ---
            checkpoint_dir = Config.CHECKPOINT_DIR / f"pangram_epoch_{epoch+1}"
            self.save_checkpoint(checkpoint_dir)
            
            # Also update 'latest'
            latest_dir = Config.CHECKPOINT_DIR / "pangram_latest"
            self.save_checkpoint(latest_dir)
            
            # Memory cleanup between epochs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_checkpoint(self, path):
        """Robust saving with local fallback and verification."""
        try:
            print(f"Saving checkpoint to {path}...")
            path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Save optimizer state for true resumption
            torch.save({
                'optimizer_state': self.optimizer.state_dict(),
                'scaler_state': self.scaler.state_dict() if self.scaler else None,
            }, path / "trainer_state.pt")
            
            # Verify
            if self.verify_checkpoint(path):
                print(f"✅ Checkpoint verified at {path}")
            else:
                raise RuntimeError("Verification failed")
                
        except Exception as e:
            print(f"❌ Failed to save/verify checkpoint to {path}: {e}")
            
            # Fallback to local
            fallback_path = Config.PROJECT_ROOT / "local_backups" / path.name
            print(f"⚠️  Attempting fallback save to {fallback_path}...")
            try:
                fallback_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(fallback_path)
                self.tokenizer.save_pretrained(fallback_path)
                
                if self.verify_checkpoint(fallback_path):
                     print(f"✅ Fallback save successful and verified.")
                else:
                     print(f"❌ Fallback saved but failed verification!")
            except Exception as e2:
                print(f"❌ FATAL: Fallback save also failed: {e2}")

    def verify_checkpoint(self, path):
        """Checks if critical files exist and are non-empty."""
        required_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
        if not path.exists(): return False
        
        for f in required_files:
            file_path = path / f
            if not file_path.exists():
                print(f"⚠️ Verification Error: Missing {f}")
                return False
            if file_path.stat().st_size == 0:
                print(f"⚠️ Verification Error: Empty file {f}")
                return False
                
        # Optional: Try lightweight loading (Config/Tokenizer only to avoid heavy VRAM)
        try:
            from transformers import AutoConfig, AutoTokenizer
            AutoConfig.from_pretrained(path)
            AutoTokenizer.from_pretrained(path)
        except Exception as e:
             print(f"⚠️ Verification Error: Corrupt config/tokenizer: {e}")
             return False
             
        return True
