import torch
from sentence_transformers import SentenceTransformer
from usearch.index import Index
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import hashlib
from typing import List

from src.config import Config


def _compute_corpus_fingerprint(parquet_files: list[str], total_count: int) -> str:
    """Compute a fingerprint for the corpus to detect changes.
    
    Based on sorted file basenames + total row count.
    Changes to files (add/remove/rename) will change the fingerprint.
    """
    # Use basenames to be path-agnostic (works across different mount points)
    basenames = sorted([Path(f).name for f in parquet_files])
    content = "|".join(basenames) + f"|count={total_count}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _read_index_corpus_manifest(index_path: Path) -> dict | None:
    """Read the corpus manifest, returning the full payload or None."""
    manifest_path = Path(index_path).with_suffix(".corpus.json")
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "r") as f:
            payload = json.load(f)
        return payload
    except Exception:
        return None


def _get_manifest_parquet_files(index_path: Path) -> list[str] | None:
    """Get just the parquet files list from manifest (backward compatible)."""
    payload = _read_index_corpus_manifest(index_path)
    if payload is None:
        return None
    parquet_files = payload.get("parquet_files")
    if not isinstance(parquet_files, list) or not parquet_files:
        return None
    return [str(p) for p in parquet_files]


def _sorted_parquet_files(parquet_path: Path) -> list[str]:
    parquet_path = Path(parquet_path)
    if parquet_path.is_dir():
        files = sorted((p.resolve() for p in parquet_path.glob("*.parquet")), key=lambda p: p.as_posix())
        return [str(p) for p in files]
    return [str(parquet_path.resolve())]


class VectorIndexer:
    def __init__(
        self,
        model_name: str = Config.EMBEDDING_MODEL,
        device: str = Config.DEVICE,
        store_text: bool = False,
        encode_batch_size: int = 512,
        cast_embeddings_to_f16: bool = True,
    ):
        self.device = device
        self.store_text = store_text
        self.encode_batch_size = int(encode_batch_size)
        self.cast_embeddings_to_f16 = bool(cast_embeddings_to_f16)

        print(f"Loading embedding model {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.vector_dim = self.model.get_sentence_embedding_dimension()

        # Initialize USearch Index
        # metric='cos' for Cosine Similarity
        # dtype='f16' for memory efficiency
        self.index = Index(ndim=self.vector_dim, metric="cos", dtype="f16")
        self.id_map = {}  # Optional in-memory map (ID -> Text)
        self.current_id = 0

    def add_texts(self, texts: List[str]):
        """Encode and add a batch of texts to the index."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.encode_batch_size,
        )

        if self.cast_embeddings_to_f16 and embeddings.dtype != np.float16:
            embeddings = embeddings.astype(np.float16, copy=False)

        start_id = self.current_id
        keys = np.arange(start_id, start_id + len(texts), dtype=np.uint64)

        self.index.add(keys, embeddings)

        if self.store_text:
            for i, text in enumerate(texts):
                self.id_map[start_id + i] = text

        self.current_id += len(texts)

    def search(self, queries: List[str], top_k: int = 1, batch_size: int | None = None, chunk_size: int | None = None):
        """Search for nearest neighbors.

        For large query sets (e.g. mining), encoding in chunks keeps GPU utilization high
        while avoiding excessive peak CPU/GPU memory.
        """
        if batch_size is None:
            batch_size = int(getattr(Config, "INDEX_ENCODE_BATCH_SIZE", self.encode_batch_size))

        if chunk_size is None:
            chunk_size = int(getattr(Config, "INDEX_QUERY_CHUNK_SIZE", 8192))

        if len(queries) <= chunk_size:
            embeddings = self.model.encode(
                queries,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            matches = self.index.search(embeddings, top_k)
            all_matches = matches
        else:
            # Chunked encode+search
            all_keys = []
            for start in range(0, len(queries), chunk_size):
                chunk = queries[start : start + chunk_size]
                embeddings = self.model.encode(
                    chunk,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
                chunk_matches = self.index.search(embeddings, top_k)

                # usearch returns an object with `.keys`; we only need keys here.
                try:
                    all_keys.append(chunk_matches.keys)
                except AttributeError:
                    all_keys.append([])

            class _Matches:
                def __init__(self, keys):
                    self.keys = keys

            all_matches = _Matches(np.concatenate(all_keys, axis=0) if all_keys and len(all_keys[0]) else [])

        matches = all_matches

        results = []

        try:
            batch_keys = matches.keys
        except AttributeError:
            print("Warning: matches.keys not found. Debugging needed.")
            batch_keys = []

        for i in range(len(queries)):
            if len(batch_keys) > i:
                query_keys = batch_keys[i]
            else:
                query_keys = []

            try:
                iter(query_keys)
            except TypeError:
                query_keys = [query_keys]

            retrieved = []
            for key in query_keys:
                key_int = int(key)
                if key_int == -1:
                    continue

                if getattr(self, "use_parquet", False):
                    try:
                        retrieved.append(self.dataset[key_int]["text"])
                    except IndexError:
                        pass
                elif key_int in self.id_map:
                    retrieved.append(self.id_map[key_int])

            results.append(retrieved)

        return results

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.index.save(str(path))

        metadata_path = path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(self.id_map, f)

        print(f"Index saved to {path}")

    @classmethod
    def load(
        cls,
        path: Path,
        model_name: str = Config.EMBEDDING_MODEL,
        device: str = Config.DEVICE,
        parquet_file: Path | None = None,
    ):
        """Load an index.

        If `parquet_file` is provided, the index uses the parquet corpus as the text store.
        This avoids loading a huge in-RAM `{id -> text}` map.
        """
        indexer = cls(model_name, device)
        path = Path(path)

        indexer.index.load(str(path))

        # Ensure newly-added vectors (if any) continue after existing IDs.
        # USearch exposes count via `__len__`.
        try:
            indexer.current_id = int(len(indexer.index))
        except Exception:
            pass

        if parquet_file and Path(parquet_file).exists():
            print(f"Using Parquet backing for text retrieval: {parquet_file}")
            from datasets import load_dataset

            manifest = _read_index_corpus_manifest(path)
            if manifest and "parquet_files" in manifest:
                data_files = [str(p) for p in manifest["parquet_files"]]
            else:
                data_files = _sorted_parquet_files(Path(parquet_file))

            indexer.dataset = load_dataset("parquet", data_files=data_files, split="train")
            indexer.use_parquet = True
            indexer.parquet_files = data_files
            
            # Validate corpus consistency
            if manifest:
                stored_fingerprint = manifest.get("corpus_fingerprint")
                stored_count = manifest.get("total_indexed", 0)
                if stored_fingerprint:
                    current_fingerprint = _compute_corpus_fingerprint(data_files, len(indexer.dataset))
                    if current_fingerprint != stored_fingerprint:
                        print(f"⚠️  WARNING: Corpus fingerprint mismatch!")
                        print(f"   Stored: {stored_fingerprint} ({stored_count} samples)")
                        print(f"   Current: {current_fingerprint} ({len(indexer.dataset)} samples)")
                        print(f"   Index IDs may not match corpus rows. Consider rebuilding the index.")
                    else:
                        print(f"✅ Corpus fingerprint validated ({len(indexer.dataset):,} samples)")
        else:
            indexer.use_parquet = False
            metadata_path = path.with_suffix(".meta.json")
            if metadata_path.exists():
                print(f"Loading metadata map from {metadata_path}...")
                with open(metadata_path, "r") as f:
                    indexer.id_map = {int(k): v for k, v in json.load(f).items()}
                    if indexer.id_map:
                        indexer.current_id = max(indexer.id_map.keys()) + 1
            else:
                print("Warning: No metadata found. Retrieval will return IDs only.")

        print(f"Index loaded from {path}")
        return indexer

