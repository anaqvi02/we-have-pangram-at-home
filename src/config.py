import torch
import os
from pathlib import Path

# Suppress tokenizer parallelism warnings when using multiple DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TensorFloat32 for better performance on NVIDIA GPUs (H100/B200)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    # User requested custom data mount
    if Path("/mnt/dataset").exists():
        DATA_DIR = Path("/mnt/dataset")
    else:
        DATA_DIR = PROJECT_ROOT / "data"

    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    
    # Model Architecture
    MODEL_NAME = "microsoft/deberta-v3-large"
    
    # Hardware / Device
    # Priority: Env Var > MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if os.environ.get("FORCE_CPU", "0") == "1":
        DEVICE = "cpu"
    else:
        DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # Context Window
    # Context Window
    # Reverted to 512 for speed (Linear/Quadratic cost of 1024 was too high).
    MAX_LENGTH = 512
    
    # Training Hyperparameters
    # High-Performance Settings for Large Model
    # Dynamic batch size based on available VRAM
    if DEVICE == "cuda" and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 140:  # B200 / H200 (192GB / 141GB)
            BATCH_SIZE = 64
            GRAD_ACCUMULATION = 1
        elif vram_gb >= 70:  # H100 80GB / A100 80GB
            BATCH_SIZE = 48
            GRAD_ACCUMULATION = 1
        elif vram_gb >= 40:  # A100 40GB
            BATCH_SIZE = 32
            GRAD_ACCUMULATION = 1
        else:
            BATCH_SIZE = 8
            GRAD_ACCUMULATION = 4
    else:
        vram_gb = 0
        BATCH_SIZE = 4
        GRAD_ACCUMULATION = 8

    # DataLoader tuning
    # Linux notebook on H100 typically benefits from a moderate worker count.
    # Too many workers can increase host RAM and cause CPU contention.
    if DEVICE == "cuda":
        _cpu = os.cpu_count() or 4
        DATALOADER_WORKERS = min(8, max(2, _cpu // 2))
    else:
        DATALOADER_WORKERS = 0

    # Embedding throughput knobs (index build + mining queries)
    # These primarily affect GPU utilization; host RAM impact is usually small.
    if DEVICE == "cuda" and torch.cuda.is_available():
        if vram_gb >= 140:
            INDEX_ENCODE_BATCH_SIZE = 8192
        elif vram_gb >= 40:
            INDEX_ENCODE_BATCH_SIZE = 4096
        else:
            INDEX_ENCODE_BATCH_SIZE = 1024
    else:
        INDEX_ENCODE_BATCH_SIZE = 256

    # Upper bound for query chunking during mining/search.
    # Keeps peak CPU/GPU memory stable when embedding many queries.
    INDEX_QUERY_CHUNK_SIZE = 8192
        
    LEARNING_RATE = 1e-5 # Lower LR for fine-tuning Large model
    NUM_EPOCHS = 3
    
    # Vector Search
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_PATH = DATA_DIR / "ai_mirrors.usearch"
    
    # Data Paths
    # Now pointing to directories containing partitioned parquet files
    HUMAN_DATASET_PATH = DATA_DIR / "human_corpus"
    AI_DATASET_PATH = DATA_DIR / "ai_corpus"
    
    # Checkpoints (Custom Mount)
    if Path("/mnt/weightsandotherstuff").exists():
        CHECKPOINT_DIR = Path("/mnt/weightsandotherstuff")
    else:
        CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    
    @staticmethod
    def print_hardware_status():
        print(f"--- Hardware Configuration ---")
        print(f"Device: {Config.DEVICE.upper()}")
        if Config.DEVICE == "mps":
            print(f"MPS Available: {torch.backends.mps.is_available()}")
            print(f"MPS Built: {torch.backends.mps.is_built()}")
        elif Config.DEVICE == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"------------------------------")

if __name__ == "__main__":
    Config.print_hardware_status()
