import modal
from pathlib import Path

# 1. Define Image (Environment)
image = (
    modal.Image.debian_slim()
    # Install system dependencies if needed
    .apt_install("git")
    # Install Python dependencies
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "sentence-transformers",
        "accelerate",
        "usearch",
        "pandas",
        "scikit-learn",
        "pyarrow"
    )
)

# 2. Define App & Volume
app = modal.App("we-have-pangram-cloud")
volume = modal.Volume.from_name("pangram-data", create_if_missing=True)

# 3. Configuration
# We map the volume to /data inside the container
DATA_ROOT = Path("/data")
AI_DIR = DATA_ROOT / "ai_corpus"
HUMAN_DIR = DATA_ROOT / "human_corpus"
INDEX_PATH = DATA_ROOT / "ai_mirrors.usearch"
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"

# 4. Functions

@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=3600)
def download_data():
    from scripts.download_data import download_wildchat, download_c4_realnewslike, ensure_dirs
    
    print("--- Starting Cloud Download ---")
    ensure_dirs(str(HUMAN_DIR), str(AI_DIR))
    
    # Download directly to the Volume path
    download_wildchat(output_dir=str(AI_DIR), limit=1000000, batch_size=100000)
    download_c4_realnewslike(output_dir=str(HUMAN_DIR), limit=1000000, batch_size=100000)
    
    volume.commit()
    print("--- Download Complete & Committed to Volume ---")

@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=3600, memory=8192)
def build_index():
    from scripts.build_index import main as build_index_main
    import sys
    
    print("--- Starting Cloud Index Build ---")
    
    # We can invoke the main function by mocking sys.argv 
    # OR better yet, let's just make build_index importable.
    # For now, we'll use subprocess or just call the script logic if we refactored it well.
    # Since we refactored `build_index.py`, we can run it via CLI shim or import.
    # Let's use subprocess to be safe with the argument parsing we just added.
    import subprocess
    
    cmd = [
        "python", "scripts/build_index.py",
        "--index_out", str(INDEX_PATH),
        "--ai_data_dir", str(AI_DIR)
    ]
    subprocess.run(cmd, check=True)
    
    volume.commit()
    print("--- Index Build Complete & Committed ---")

@app.function(image=image, volumes={DATA_ROOT: volume}, gpu="A10G", timeout=7200)
def train():
    from src.train_lib import train_pipeline
    
    print("--- Starting Cloud Training (A10G) ---")
    
    train_pipeline(
        human_data_dir=str(HUMAN_DIR),
        ai_data_dir=str(AI_DIR),
        index_path=str(INDEX_PATH),
        output_model_dir=str(CHECKPOINT_DIR),
        epochs=3,
        use_mock_data=False
    )
    
    volume.commit()
    print("--- Training Complete & Checkpoints Saved ---")

@app.local_entrypoint()
def main():
    print("Triggering Modal Pipeline...")
    # 1. Download
    # download_data.remote()
    
    # 2. Index (Uncomment to run)
    # build_index.remote()
    
    # 3. Train
    train.remote()
