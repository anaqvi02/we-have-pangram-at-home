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
    # On high-end GPUs like B200, we can use a much larger batch size.
    # Current effective batch size goal is 32.
    if DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 40e9:
        BATCH_SIZE = 32  # Use full power of modern GPUs
        GRAD_ACCUMULATION = 1
    else:
        BATCH_SIZE = 4  
        GRAD_ACCUMULATION = 8 
        
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
