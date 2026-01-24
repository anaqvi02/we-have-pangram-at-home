import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.indexing import VectorIndexer
from src.config import Config
from datasets import load_dataset
import random

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build Vector Index")
    parser.add_argument("--index_out", type=str, default=str(Config.INDEX_PATH), help="Path to save the index")
    parser.add_argument("--ai_data_dir", type=str, default=str(Config.AI_DATASET_PATH), help="Path to AI parquet files")
    args = parser.parse_args()

    index_out = Path(args.index_out)
    ai_data_path = Path(args.ai_data_dir)

    print("Initializing Indexer...")
    # Vital Optimization: store_text=False to prevents loading 1M strings into RAM.
    # We rely on the Parquet file position matches (Implicit ID) for retrieval.
    indexer = VectorIndexer(store_text=False)
    
    print(f"Loading AI Corpus from {ai_data_path}...")
    
    # Updated: Load from directory of partitioned parquets using streaming
    ai_files = str(ai_data_path / "part_*.parquet")
    try:
        dataset = load_dataset("parquet", data_files=ai_files, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to single file if parts fail (legacy support)
        legacy_file = str(ai_data_path / "part_0.parquet")
        dataset = load_dataset("parquet", data_files=legacy_file, split="train", streaming=True)

    print(f"Indexing stream...")
    
    import torch
    batch_size = 64
    batch_texts = []
    total_indexed = 0
    
    from tqdm import tqdm
    # Calculate estimated total for progress bar
    # 1. Count partition files
    partition_files = list(ai_data_path.glob("part_*.parquet"))
    # 2. Assume roughly 100k per file (based on download script)
    # If no partitions found (legacy single file), assume 1M default or try to read metadata (expensive)
    if partition_files:
        estimated_total = len(partition_files) * 100000
    else:
        estimated_total = 1000000 # Fallback default
        
    print(f"Estimated Total Samples: {estimated_total}")

    for sample in tqdm(dataset, desc="Indexing", total=estimated_total):
        text = sample['text']
        batch_texts.append(text)
        
        if len(batch_texts) >= batch_size:
            indexer.add_texts(batch_texts)
            total_indexed += len(batch_texts)
            batch_texts = []
            
            if total_indexed % 1000 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Final batch
    if batch_texts:
        indexer.add_texts(batch_texts)
        total_indexed += len(batch_texts)
        
    print(f"Saving Index to {index_out}...")
    index_out.parent.mkdir(parents=True, exist_ok=True)
    indexer.save(index_out)
    print(f"Done! Indexed {total_indexed} documents.")

if __name__ == "__main__":
    main()
