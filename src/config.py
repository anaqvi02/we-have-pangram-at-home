import torch
import os
from pathlib import Path

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    
    # Model Architecture
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512  # Token limit
    
    # Hardware / Device
    # Priority: Env Var > MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if os.environ.get("FORCE_CPU", "0") == "1":
        DEVICE = "cpu"
    else:
        DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training Hyperparameters
    BATCH_SIZE = 8  # Adjusted for M3 Air/Pro memory limits
    GRAD_ACCUMULATION = 4 # 8 * 4 = 32 effective batch size
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    
    # Vector Search
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_PATH = DATA_DIR / "ai_mirrors.usearch"
    
    # Data Paths
    # Now pointing to directories containing partitioned parquet files
    HUMAN_DATASET_PATH = DATA_DIR / "human_corpus"
    AI_DATASET_PATH = DATA_DIR / "ai_corpus"
    
    @staticmethod
    def print_hardware_status():
        print(f"--- Hardware Configuration ---")
        print(f"Device: {Config.DEVICE.upper()}")
        if Config.DEVICE == "mps":
            print(f"MPS Available: {torch.backends.mps.is_available()}")
            print(f"MPS Built: {torch.backends.mps.is_built()}")
        print(f"------------------------------")

if __name__ == "__main__":
    Config.print_hardware_status()
