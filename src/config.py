import torch
import os
from pathlib import Path

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
    # Formal writing benefits from longer context (1024).
    # We use 1024 on CUDA (H100/B200) and 512 on MPS (to save local VRAM).
    if DEVICE == "cuda":
        MAX_LENGTH = 1024
    else:
        MAX_LENGTH = 512
    
    # Training Hyperparameters
    # High-Performance Settings for Large Model
    BATCH_SIZE = 4  # Lower per-device batch for 'Large' model memory
    GRAD_ACCUMULATION = 8 # 4 * 8 = 32 effective batch size
    LEARNING_RATE = 1e-5 # Lower LR for fine-tuning Large model
    NUM_EPOCHS = 2
    
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
