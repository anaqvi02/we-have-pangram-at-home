from src.data.indexing import VectorIndexer
from src.config import Config
from datasets import load_dataset
import random

def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_out", type=str, default=str(Config.INDEX_PATH))
    parser.add_argument("--ai_data_dir", type=str, default=str(Config.AI_DATASET_PATH))
    args = parser.parse_args()
    
    print("Initializing Indexer...")
    indexer = VectorIndexer(store_text=False)
    
    ai_data_path = Path(args.ai_data_dir)
    print(f"Loading AI Corpus from {ai_data_path}...")
    
    # Load from directory of partitioned parquets using streaming
    ai_files = str(ai_data_path / "wildchat_part_*.parquet")
    try:
        dataset = load_dataset("parquet", data_files=ai_files, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset from parts: {e}")
        # Fallback
        legacy_file = str(ai_data_path / "wildchat.parquet")
        dataset = load_dataset("parquet", data_files=legacy_file, split="train", streaming=True)

    print(f"Indexing stream...")
    
    import torch
    batch_size = 64
    batch_texts = []
    total_indexed = 0
    
    from tqdm import tqdm
    
    for sample in tqdm(dataset, desc="Indexing"):
        text = sample['text']
        batch_texts.append(text)
        
        if len(batch_texts) >= batch_size:
            indexer.add_texts(batch_texts)
            total_indexed += len(batch_texts)
            batch_texts = []
            
            if total_indexed % 1000 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    # Final batch
    if batch_texts:
        indexer.add_texts(batch_texts)
        total_indexed += len(batch_texts)
        
    print(f"Saving Index to {args.index_out}...")
    # Ensure parent dir exists
    Path(args.index_out).parent.mkdir(parents=True, exist_ok=True)
    indexer.save(Path(args.index_out))
    print(f"Done! Indexed {total_indexed} documents.")

if __name__ == "__main__":
    main()
