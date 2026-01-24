from src.data.indexing import VectorIndexer
import pandas as pd
from pathlib import Path
import shutil
import torch

TEST_DIR = Path("test_zero_copy_data")
TEST_PARQUET = TEST_DIR / "data.parquet"
TEST_INDEX = TEST_DIR / "test.index"

def setup():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir()
    
    # Create dummy data
    data = [
        {'text': f"Text sample {i}", 'label': 1, 'id': i} 
        for i in range(10)
    ]
    df = pd.DataFrame(data)
    df.to_parquet(TEST_PARQUET)
    return [d['text'] for d in data]

def test_zero_copy():
    print("1. Creating Index...")
    texts = setup()
    
    # Create Index normally
    indexer = VectorIndexer(device="cpu") # Force CPU for test
    indexer.add_texts(texts)
    indexer.save(TEST_INDEX)
    
    print("2. Loading with Parquet Backing (Zero Copy)...")
    # Load WITHOUT metadata map, pointing to parquet
    indexer_loaded = VectorIndexer.load(TEST_INDEX, device="cpu", parquet_file=TEST_PARQUET)
    
    assert getattr(indexer_loaded, 'use_parquet', False) == True, "use_parquet flag not set"
    assert len(indexer_loaded.id_map) == 0, "Memory map should be empty"
    
    print("3. Testing Retrieval...")
    # Search for something that matches sample 0
    query = texts[0]
    results = indexer_loaded.search([query], top_k=1)
    
    retrieved = results[0][0]
    print(f"Query: {query}")
    print(f"Retrieved: {retrieved}")
    
    assert retrieved == query, "Failed to retrieve correct text via Parquet"
    print("✅ Zero-Copy Retrieval Verified!")
    
    # Cleanup
    shutil.rmtree(TEST_DIR)

if __name__ == "__main__":
    test_zero_copy()
