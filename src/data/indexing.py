import torch
from sentence_transformers import SentenceTransformer
from usearch.index import Index
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Generator

from src.config import Config

class VectorIndexer:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL, device: str = Config.DEVICE, store_text: bool = False):
        self.device = device
        self.store_text = store_text
        print(f"Loading embedding model {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize USearch Index
        # metric='cos' for Cosine Similarity
        # dtype='f16' for memory efficiency on M3
        self.index = Index(ndim=self.vector_dim, metric='cos', dtype='f16')
        self.id_map = {} # Simple in-memory map for basic implementation (ID -> Text)
        self.current_id = 0

    def add_texts(self, texts: List[str]):
        """
        Encode and add a batch of texts to the index.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        start_id = self.current_id
        keys = np.arange(start_id, start_id + len(texts), dtype=np.uint64)
        
        self.index.add(keys, embeddings)
        
        # Update mapping
        if self.store_text:
            for i, text in enumerate(texts):
                self.id_map[start_id + i] = text
            
        self.current_id += len(texts)

    def search(self, queries: List[str], top_k: int = 1):
        """
        Search for nearest neighbors.
        """
        embeddings = self.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        matches = self.index.search(embeddings, top_k)
        
        # Matches is a BatchMatches object in usearch
        # We need to extract keys and reconstruct results
        results = []
        
        # usearch returns slightly different structures depending on version
        # Access keys directly as numpy array to avoid object iteration issues
        # matches.keys is usually (batch_size, top_k)
        try:
             batch_keys = matches.keys
        except AttributeError:
             # Fallback for older versions or single-query edge cases
             # If matches is a single object, wrap it?
             # But we doing batch search.
             print("Warning: matches.keys not found. Debugging needed.")
             batch_keys = []

        for i in range(len(queries)):
            # keys for this query
            if len(batch_keys) > i:
                query_keys = batch_keys[i]
            else:
                query_keys = []
                
            # Flatten if needed or ensure iterable
            try:
                iter(query_keys)
            except TypeError:
                query_keys = [query_keys]

            retrieved = []
            for key in query_keys:
                 # key might be numpy uint64
                 key_int = int(key)
                 if key_int == -1: # USearch returns -1 for no match sometimes? Or standard mask?
                     continue
                     
                 if hasattr(self, 'use_parquet') and self.use_parquet:
                     try:
                         retrieved.append(self.dataset[key_int]['text'])
                     except IndexError:
                         pass
                 elif key_int in self.id_map:
                     retrieved.append(self.id_map[key_int])
            results.append(retrieved)
            
        return results

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save USearch Index
        self.index.save(str(path))
        
        # Save Metadata (ID Map) - In prod this would be a real DB (SQLite/LMDB)
        # For <50GB limit, a compressed JSON or Parquet is fine for the map
        metadata_path = path.with_suffix(".meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.id_map, f)
            
        print(f"Index saved to {path}")

    @classmethod
    def load(cls, path: Path, model_name: str = Config.EMBEDDING_MODEL, device: str = Config.DEVICE, parquet_file: Path = None):
        """
        Load an index.
        Args:
            parquet_file: Optional path to the source parquet file to use as the text store.
                          If provided, avoids loading the huge JSON metadata map into RAM.
        """
        indexer = cls(model_name, device)
        path = Path(path)
        
        indexer.index.load(str(path))
        
        if parquet_file and parquet_file.exists():
            print(f"Using Parquet backing for text retrieval: {parquet_file}")
            # Use HuggingFace Datasets for memory-mapped access
            from datasets import load_dataset
            # Fix: Handle directory of part files using glob pattern or data_files list
            # parquet_file is now a Directory Path (from config)
            parquet_path = Path(parquet_file)
            if parquet_path.is_dir():
                data_files = str(parquet_path / "*.parquet")
            else:
                data_files = str(parquet_path)
            
            indexer.dataset = load_dataset("parquet", data_files=data_files, split="train")
            indexer.use_parquet = True
        else:
            indexer.use_parquet = False
            metadata_path = path.with_suffix(".meta.json")
            if metadata_path.exists():
                print(f"Loading metadata map from {metadata_path}...")
                with open(metadata_path, 'r') as f:
                    indexer.id_map = {int(k): v for k, v in json.load(f).items()}
                    if indexer.id_map:
                        indexer.current_id = max(indexer.id_map.keys()) + 1
            else:
                print("Warning: No metadata found. Retrieval will return IDs only.")
        
        print(f"Index loaded from {path}")
        return indexer

