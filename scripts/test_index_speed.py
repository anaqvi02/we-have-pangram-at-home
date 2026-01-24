from src.data.indexing import VectorIndexer
import time
import torch

def main():
    print("Initializing Indexer...")
    indexer = VectorIndexer()
    
    texts = ["This is a test sentence for benchmarking speed." for _ in range(100)]
    
    print("Encoding 100 items...")
    start = time.time()
    indexer.add_texts(texts)
    end = time.time()
    
    print(f"Done. Time: {end - start:.4f}s")
    print(f"Speed: {100 / (end - start):.2f} items/sec")

if __name__ == "__main__":
    main()
