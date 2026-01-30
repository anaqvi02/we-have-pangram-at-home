import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.indexing import VectorIndexer
from src.config import Config
from datasets import load_dataset

import argparse


def _sorted_parquet_files(parquet_dir: Path) -> list[str]:
    parquet_dir = Path(parquet_dir)
    if parquet_dir.is_dir():
        files = sorted((p.resolve() for p in parquet_dir.glob("*.parquet")), key=lambda p: p.as_posix())
        return [str(p) for p in files]
    return [str(parquet_dir.resolve())]


def _write_index_corpus_manifest(index_out: Path, parquet_files: list[str], total_indexed: int):
    payload = {
        "parquet_files": parquet_files,
        "total_indexed": int(total_indexed),
    }
    manifest_path = Path(index_out).with_suffix(".corpus.json")
    with open(manifest_path, "w") as f:
        json.dump(payload, f)


def main():
    parser = argparse.ArgumentParser(description="Build Vector Index")
    parser.add_argument("--index_out", type=str, default=str(Config.INDEX_PATH), help="Path to save the index")
    parser.add_argument("--ai_data_dir", type=str, default=str(Config.AI_DATASET_PATH), help="Path to AI parquet files")
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=2048,
        help="Embedding encode batch size (SentenceTransformer). Tune for GPU VRAM.",
    )
    parser.add_argument(
        "--add_batch_size",
        type=int,
        default=16384,
        help="How many texts to accumulate before calling add_texts().",
    )
    args = parser.parse_args()

    index_out = Path(args.index_out)
    ai_data_path = Path(args.ai_data_dir)

    print("\n--- 🖥️  Hardware Diagnostics ---")
    import torch

    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print("✅ CUDA Available: Yes")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("✅ MPS Available: Yes (Apple Silicon)")
    else:
        print("⚠️  WARNING: No GPU detected. Running on CPU (will be slow).")
        print("   If you have a GPU, check your PyTorch installation.")
    print("--------------------------------\n")

    print("Initializing Indexer...")
    # Vital Optimization: store_text=False prevents loading 1M strings into RAM.
    # Correctness requirement: corpus file order must match later Parquet-backed retrieval.
    indexer = VectorIndexer(store_text=False, encode_batch_size=args.encode_batch_size)

    print(f"Loading AI Corpus from {ai_data_path}...")

    parquet_files = _sorted_parquet_files(ai_data_path)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {ai_data_path}")

    # Use an explicit ordered list for deterministic ID mapping.
    # Non-streaming parquet loading is Arrow-backed and typically memory-mapped.
    dataset = load_dataset("parquet", data_files=parquet_files, split="train")

    print("Indexing...")

    # Dynamic default add-batch size based on VRAM (MiniLM is small, can use large batches)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 140:  # B200 / H200
            default_add_batch_size = 16384
        elif vram_gb >= 40:  # A100 / H100
            default_add_batch_size = 8192
        else:
            default_add_batch_size = 2048
    else:
        default_add_batch_size = 1024  # CPU/MPS
        vram_gb = 0

    add_batch_size = int(args.add_batch_size) if args.add_batch_size else int(default_add_batch_size)

    print(
        f"Using add batch size: {add_batch_size} (VRAM: {vram_gb:.1f} GB) | encode batch size: {args.encode_batch_size}"
        if vram_gb > 0
        else f"Using add batch size: {add_batch_size} (CPU/MPS) | encode batch size: {args.encode_batch_size}"
    )

    batch_texts = []
    total_indexed = 0

    from tqdm import tqdm

    # Progress estimate: fall back to row-per-file heuristic.
    estimated_total = len(parquet_files) * 100000
    print(f"Estimated Total Samples: {estimated_total}")

    for sample in tqdm(dataset, desc="Indexing", total=estimated_total):
        text = sample["text"]
        batch_texts.append(text)

        if len(batch_texts) >= add_batch_size:
            indexer.add_texts(batch_texts)
            total_indexed += len(batch_texts)
            batch_texts = []

            if total_indexed % 10000 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if batch_texts:
        indexer.add_texts(batch_texts)
        total_indexed += len(batch_texts)

    print(f"Saving Index to {index_out}...")
    index_out.parent.mkdir(parents=True, exist_ok=True)
    indexer.save(index_out)
    _write_index_corpus_manifest(index_out, parquet_files, total_indexed)

    print(f"Done! Indexed {total_indexed} documents.")


if __name__ == "__main__":
    main()
