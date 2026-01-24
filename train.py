import torch
import argparse
from pathlib import Path
from transformers import DebertaV2TokenizerFast

from src.config import Config
from src.model.detector import PangramDetector
from src.data.indexing import VectorIndexer
from src.data.indexing import VectorIndexer
from src.data.loader import StreamingTextDataset, PretokenizedDataset
from src.training.trainer import PangramTrainer
import random
from datasets import load_dataset

def create_mock_data(size=100):
    """Generates dummy data for testing the pipeline."""
    print(f"Generating {size} mock samples...")
    data = []
    for _ in range(size):
        # 50% Human, 50% AI
        label = random.randint(0, 1)
        text = "This is a sample text used for testing the Pangram implementation pipeline on Apple Silicon." * 5
        data.append({'text': text, 'label': label})
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default=str(Config.INDEX_PATH))
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--use_mock_data", action="store_true", help="Use generated dummy data")
    args = parser.parse_args()
    
    # User requested reduction to 3 epochs while script was running
    if args.epochs > 3:
        print(f"Overriding requested epochs {args.epochs} to 3 per user request.")
        args.epochs = 3
    
    # 1. Initialize Components
    print("Initializing Model...")
    detector = PangramDetector()
    tokenizer = detector.tokenizer
    
    print("Loading Indexer...")
    if Path(args.index_path).exists():
        # Use Parquet backing for efficiency
        indexer = VectorIndexer.load(args.index_path, parquet_file=Config.AI_DATASET_PATH)
    else:
        print("Index not found. Initializing empty indexer (Warning: Mining will fail without an index).")
        indexer = VectorIndexer() 
        # For mock run we might want to populate it slightly
        if args.use_mock_data:
            print("Populating mock index...")
            indexer.add_texts(["This is a mock AI text mirror." for _ in range(10)])

    # 2. Load Data
    if args.use_mock_data:
        train_data_source = create_mock_data(100)
        human_pool_source = create_mock_data(50) # In reality, filter for label=0
        human_pool_source = [x for x in human_pool_source if x['label'] == 0]
    else:
        print("Loading Real Datasets (Optimized)...")
        
        # 1. Load Human Data (C4) via Arrow (Zero-Copy)
        if not Config.HUMAN_DATASET_PATH.exists():
             raise FileNotFoundError(f"Human dataset not found at {Config.HUMAN_DATASET_PATH}")
        print(f"Loading Human Corpus from {Config.HUMAN_DATASET_PATH}...")
        # Load from directory of parquet files
        human_files = str(Config.HUMAN_DATASET_PATH / "*.parquet")
        human_ds = load_dataset("parquet", data_files=human_files, split="train")
        
        # 2. Load AI Data (WildChat) via Arrow (Zero-Copy)
        if not Config.AI_DATASET_PATH.exists():
             raise FileNotFoundError(f"AI dataset not found at {Config.AI_DATASET_PATH}")
        print(f"Loading AI Corpus from {Config.AI_DATASET_PATH}...")
        ai_files = str(Config.AI_DATASET_PATH / "*.parquet")
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
        import gc
        gc.collect()

        # 4. Select Initial Training Data (20k Human + 20k AI)
        print("Preparing Initial Training Set (40k samples)...")
        # Offset by 1000 to avoid overlap with validation
        initial_human = human_ds[1000:21000]
        initial_ai = ai_ds[1000:21000]
        
        # 5. Pre-Tokenize NOW (Batch Process)
        print("Pre-tokenizing data (this may take a minute)...")
        
        # Combine texts and labels
        texts = initial_human['text'] + initial_ai['text']
        labels = [0] * len(initial_human['text']) + [1] * len(initial_ai['text'])
        
        # Tokenize in one go
        encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=Config.MAX_LENGTH, 
            return_tensors="pt"
        )
        
        # Create Tensor Dataset
        print("Creating Tensor Dataset...")
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        train_dataset = PretokenizedDataset(
            encodings['input_ids'], 
            encodings['attention_mask'], 
            labels_tensor
        )
        
        # Free heavy memory immediately
        del texts, labels, encodings, labels_tensor
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # 6. Prepare Mining Pool (Streaming)
        # We use the rest of human_ds for mining (Offset 21000)
        
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
        
        print(f"Validation Set: {len(val_dataset)} samples")
        print(f"Initial Training Set: {len(train_dataset)} samples (Pre-tokenized)")
        print(f"Mining Pool: {len(human_pool_source)} samples (Disk-backed)")
    
    # 3. Trainer
    trainer = PangramTrainer(detector.model, tokenizer, indexer)
    
    # 4. Run Curriculum
    trainer.run_curriculum(train_dataset, human_pool_source, val_dataset=val_dataset, epochs=args.epochs)
    
    # 5. Save Model
    save_path = Config.PROJECT_ROOT / "checkpoints" / "pangram_m3"
    detector.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
