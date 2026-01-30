import torch
import random
from pathlib import Path
from transformers import DebertaV2TokenizerFast
from datasets import load_dataset, concatenate_datasets
import gc

from src.model.detector import PangramDetector
from src.data.indexing import VectorIndexer
from src.data.loader import MemoryMappedDataset, GrowableDataset
from src.training.trainer import PangramTrainer
from src.config import Config


def _load_trainer_state(path: Path):
    state_path = Path(path) / "trainer_state.pt"
    if not state_path.exists():
        return None

    try:
        return torch.load(state_path, map_location="cpu")
    except Exception as e:
        print(f"⚠️ Failed to load trainer state from {state_path}: {e}")
        return None

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
    
    Memory-Optimized: Uses memory-mapped datasets instead of loading all data into RAM.
    """
    human_data_path = Path(human_data_dir)
    ai_data_path = Path(ai_data_dir)
    idx_path = Path(index_path)
    save_path = Path(output_model_dir)

    # 1. Initialize Components
    print("Initializing Model...")

    # Checkpoint root is driven by output_model_dir so Modal/notebook runs can persist
    # everything to the mounted volume.
    checkpoint_root = save_path

    # Check for resumption
    start_epoch = 0
    resume_path = None

    # Simple scan for pangram_epoch_N folders
    existing_epochs = []
    if checkpoint_root.exists():
        for p in checkpoint_root.iterdir():
            if p.is_dir() and p.name.startswith("pangram_epoch_"):
                try:
                    ep_num = int(p.name.split("_")[-1])
                    existing_epochs.append(ep_num)
                except ValueError:
                    pass

    if existing_epochs:
        latest_epoch = max(existing_epochs)
        resume_path = checkpoint_root / f"pangram_epoch_{latest_epoch}"
        print(f"🔄 Resuming training from Epoch {latest_epoch} (Checkpoint: {resume_path})")

        detector = PangramDetector.load(str(resume_path))
        start_epoch = latest_epoch
    else:
        print("🆕 Starting training from scratch.")
        detector = PangramDetector()
    
    # Apply torch.compile for CUDA (significant speedup on modern GPUs)
    if Config.DEVICE == "cuda" and hasattr(torch, 'compile'):
        print("⚡ Compiling model with torch.compile (max-autotune)...")
        try:
            detector.model = torch.compile(detector.model, mode="max-autotune")
            print("✅ Model compiled successfully")
        except Exception as e:
            print(f"⚠️ torch.compile failed, continuing without: {e}")

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

    # 2. Load Data (Memory-Mapped)
    if use_mock_data:
        # For testing: Create small in-memory datasets
        from datasets import Dataset as HFDataset
        mock_human = HFDataset.from_dict({
            'text': ["Mock human text " * 10] * 100,
            'label': [0] * 100
        })
        mock_ai = HFDataset.from_dict({
            'text': ["Mock AI text " * 10] * 100,
            'label': [1] * 100
        })
        human_ds = mock_human
        ai_ds = mock_ai
    else:
        print("Loading Datasets (Memory-Mapped)...")
        
        # 1. Load Human Data (memory-mapped via Arrow - NOT loaded into RAM)
        print(f"  → Loading Human Corpus from {human_data_path}...")
        human_files = str(human_data_path / "*.parquet")
        human_ds = load_dataset("parquet", data_files=human_files, split="train")
        print(f"    Found {len(human_ds):,} human samples (memory-mapped)")
        
        # 2. Load AI Data (memory-mapped)
        print(f"  → Loading AI Corpus from {ai_data_path}...")
        ai_files = str(ai_data_path / "*.parquet")
        ai_ds = load_dataset("parquet", data_files=ai_files, split="train")
        print(f"    Found {len(ai_ds):,} AI samples (memory-mapped)")
    
    # 3. Prepare Splits (still memory-mapped, no RAM usage)
    print("Preparing data splits...")
    
    # Shuffle with seed for reproducibility
    human_ds = human_ds.shuffle(seed=42)
    ai_ds = ai_ds.shuffle(seed=42)
    
    # Validation Set (5k per class for reliable metrics)
    val_size = min(5000, len(human_ds) // 20, len(ai_ds) // 20)
    print(f"  → Validation set: {val_size * 2:,} samples ({val_size} per class)")
    
    val_human = human_ds.select(range(val_size))
    val_ai = ai_ds.select(range(val_size))
    
    # Add label column if missing (for concatenation)
    if 'label' not in val_human.column_names:
        val_human = val_human.add_column('label', [0] * len(val_human))
    if 'label' not in val_ai.column_names:
        val_ai = val_ai.add_column('label', [1] * len(val_ai))
    
    val_combined = concatenate_datasets([val_human, val_ai]).shuffle(seed=42)
    val_dataset = MemoryMappedDataset(val_combined, tokenizer, max_length=Config.MAX_LENGTH)
    
    # Training Set (start with 50k per class)
    train_start_idx = val_size

    # Available samples per class after carving out validation.
    # We take up to 50k per class (not half of what's left).
    human_available = max(0, len(human_ds) - val_size)
    ai_available = max(0, len(ai_ds) - val_size)
    train_per_class = min(50000, human_available, ai_available)
    train_end_idx = train_start_idx + train_per_class
    
    print(f"  → Initial training set: {train_per_class * 2:,} samples ({train_per_class} per class)")
    
    train_human = human_ds.select(range(train_start_idx, train_end_idx))
    train_ai = ai_ds.select(range(train_start_idx, train_end_idx))
    
    # Add label column if missing
    if 'label' not in train_human.column_names:
        train_human = train_human.add_column('label', [0] * len(train_human))
    if 'label' not in train_ai.column_names:
        train_ai = train_ai.add_column('label', [1] * len(train_ai))
    
    train_combined = concatenate_datasets([train_human, train_ai]).shuffle(seed=42)
    
    # Use GrowableDataset for curriculum learning (can add mined samples)
    train_dataset = GrowableDataset(train_combined, tokenizer, max_length=Config.MAX_LENGTH)
    
    # Mining Pool (for finding hard negatives) - kept memory-mapped
    mining_start_idx = train_end_idx
    human_pool = human_ds.select(range(mining_start_idx, len(human_ds)))
    if 'label' not in human_pool.column_names:
        human_pool = human_pool.add_column('label', [0] * len(human_pool))
    
    print(f"  → Mining pool: {len(human_pool):,} human samples")
    
    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n📊 Memory Usage Summary:")
    print(f"   Datasets are memory-mapped (minimal RAM usage)")
    print(f"   Tokenization happens on-the-fly (batched in collate_fn, dynamic padding)")
    
    # 3. Trainer
    trainer = PangramTrainer(detector.model, tokenizer, indexer, checkpoint_root=checkpoint_root)

    # If resuming, restore optimizer/scaler state for true continuity.
    if resume_path is not None:
        state = _load_trainer_state(resume_path)
        if state:
            try:
                trainer.optimizer.load_state_dict(state.get("optimizer_state", {}))
                if trainer.scaler is not None and state.get("scaler_state") is not None:
                    trainer.scaler.load_state_dict(state["scaler_state"])
                print("✅ Restored optimizer/scaler state")
            except Exception as e:
                print(f"⚠️ Failed to restore optimizer/scaler state: {e}")

    # 4. Run Curriculum
    trainer.run_curriculum(
        train_dataset,
        human_pool,  # Pass the HF dataset directly
        val_dataset=val_dataset,
        epochs=epochs,
        start_epoch=start_epoch,
    )
    
    # 5. Save Model
    save_path.mkdir(parents=True, exist_ok=True)
    detector.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
