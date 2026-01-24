from src.data.indexing import VectorIndexer
from src.config import Config
from datasets import load_dataset
import random

def main():
    print("Initializing Indexer...")
    # Vital Optimization: store_text=False to prevents loading 1M strings into RAM.
    # We rely on the Parquet file position matches (Implicit ID) for retrieval.
    indexer = VectorIndexer(store_text=False)
    
    print(f"Loading AI Corpus from {Config.AI_DATASET_PATH}...")
    
    # Updated: Load from directory of partitioned parquets using streaming
    # We use a glob pattern or just directory path if supported
    # datasets load_dataset("parquet", data_files="dir/*.parquet", streaming=True)
    
    ai_files = str(Config.AI_DATASET_PATH / "wildchat_part_*.parquet")
    try:
        dataset = load_dataset("parquet", data_files=ai_files, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to single file if parts fail (legacy support)
        legacy_file = str(Config.AI_DATASET_PATH / "wildchat.parquet")
        dataset = load_dataset("parquet", data_files=legacy_file, split="train", streaming=True)

    print(f"Indexing stream...")
    
    import torch
    batch_size = 64
    batch_texts = []
    total_indexed = 0
    
    from tqdm import tqdm
    # We don't know exact length in streaming mode easily without metadata, 
    # but we can just use progress bar updates.
    
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
        
    print("Saving Index...")
    indexer.save(Config.INDEX_PATH)
    print(f"Done! Indexed {total_indexed} documents.")

if __name__ == "__main__":
    main()
