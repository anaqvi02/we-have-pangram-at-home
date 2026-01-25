import torch
import random
from pathlib import Path
from transformers import DebertaV2TokenizerFast
from datasets import load_dataset
import gc

from src.model.detector import PangramDetector
from src.data.indexing import VectorIndexer
from src.data.loader import StreamingTextDataset, PretokenizedDataset
from src.training.trainer import PangramTrainer
from src.config import Config

def train_pipeline(
    human_data_dir: str,
    ai_data_dir: str,
    index_path: str,
    output_model_dir: str,
    epochs: int = 3,
    use_mock_data: bool = False
):
    """
    Core training logic refactored for portability.
    Accepts explicit paths to allow running on Modal (or any env).
    """
    human_data_path = Path(human_data_dir)
    ai_data_path = Path(ai_data_dir)
    idx_path = Path(index_path)
    save_path = Path(output_model_dir)

    # 1. Initialize Components
    print("Initializing Model...")
    
    # Check for resumption
    start_epoch = 0
    resume_path = None
    
    # Simple scan for pangram_epoch_N folders
    existing_epochs = []
    if Config.CHECKPOINT_DIR.exists():
        for p in Config.CHECKPOINT_DIR.iterdir():
            if p.is_dir() and p.name.startswith("pangram_epoch_"):
                try:
                    ep_num = int(p.name.split("_")[-1])
                    existing_epochs.append(ep_num)
                except ValueError:
                    pass
    
    if existing_epochs:
        latest_epoch = max(existing_epochs)
        resume_path = Config.CHECKPOINT_DIR / f"pangram_epoch_{latest_epoch}"
        print(f"🔄 Resuming training from Epoch {latest_epoch} (Checkpoint: {resume_path})")
        
        # Load the model from the checkpoint
        detector = PangramDetector.load(resume_path)
        start_epoch = latest_epoch
    else:
        print("🆕 Starting training from scratch.")
        detector = PangramDetector()

    tokenizer = detector.tokenizer
    
    print(f"Loading Indexer from {idx_path}...")
    if idx_path.exists():
        # Use Parquet backing for efficiency
        indexer = VectorIndexer.load(idx_path, parquet_file=ai_data_path)
    else:
        print("Index not found. Initializing empty indexer (Warning: Mining will fail without an index).")
        indexer = VectorIndexer() 
        if use_mock_data:
            print("Populating mock index...")
            indexer.add_texts(["This is a mock AI text mirror." for _ in range(10)])

    # 2. Load Data
    if use_mock_data:
        train_data_source = [{'text': "Mock text "*5, 'label': 0}] * 100
        # In reality, filter for label=0
        human_pool_source = [{'text': "Mock human "*5, 'label': 0}] * 50
    else:
        print("Loading Real Datasets (Optimized)...")
        
        # 1. Load Human Data
        print(f"Loading Human Corpus from {human_data_path}...")
        human_files = str(human_data_path / "part_*.parquet")
        human_ds = load_dataset("parquet", data_files=human_files, split="train")
        
        # 2. Load AI Data
        print(f"Loading AI Corpus from {ai_data_path}...")
        ai_files = str(ai_data_path / "part_*.parquet")
        ai_ds = load_dataset("parquet", data_files=ai_files, split="train")
        
        # 3. Validation Set (Held Out - 2k samples)
        print("Preparing Validation Set...")
        val_human = human_ds[0:1000]
        val_ai = ai_ds[0:1000]
        val_texts = val_human['text'] + val_ai['text']
        val_labels = [0] * len(val_human['text']) + [1] * len(val_ai['text'])
        
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )
        val_dataset = PretokenizedDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(val_labels, dtype=torch.long)
        )
        
        del val_texts, val_labels, val_encodings
        gc.collect()

        # 4. Select Initial Training Data (20k Human + 20k AI)
        print("Preparing Initial Training Set (40k samples)...")
        initial_human = human_ds[1000:21000]
        initial_ai = ai_ds[1000:21000]
        
        # 5. Pre-Tokenize NOW (Batch Process)
        print("Pre-tokenizing data...")
        
        texts = initial_human['text'] + initial_ai['text']
        labels = [0] * len(initial_human['text']) + [1] * len(initial_ai['text'])
        
        encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=Config.MAX_LENGTH, 
            return_tensors="pt"
        )
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        train_dataset = PretokenizedDataset(
            encodings['input_ids'], 
            encodings['attention_mask'], 
            labels_tensor
        )
        
        del texts, labels, encodings, labels_tensor
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 6. Prepare Mining Pool (Streaming)
        class HumanPoolWrapper:
            def __init__(self, hf_ds, start_idx=21000):
                self.ds = hf_ds
                self.start_idx = start_idx
            
            def __len__(self):
                return len(self.ds) - self.start_idx
            
            def __getitem__(self, idx):
                real_idx = self.start_idx + idx
                item = self.ds[real_idx]
                return {'text': item['text'], 'label': 0}
                
        human_pool_source = HumanPoolWrapper(human_ds, start_idx=21000)
    
    # 3. Trainer
    trainer = PangramTrainer(detector.model, tokenizer, indexer)
    
    # 4. Run Curriculum
    trainer.run_curriculum(train_dataset, human_pool_source, val_dataset=val_dataset, epochs=epochs, start_epoch=start_epoch)
    
    # 5. Save Model
    save_path.mkdir(parents=True, exist_ok=True)
    detector.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
