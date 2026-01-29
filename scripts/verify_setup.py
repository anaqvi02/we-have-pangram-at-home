import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def check_setup():
    print("=" * 60)
    print("🚀 PANGRAM PRE-FLIGHT CHECK")
    print("=" * 60)

    # 1. Check Paths & Mounts
    print("\n📁 PATHS & VOLUMES:")
    data_dir = Config.DATA_DIR
    checkpoint_dir = Config.CHECKPOINT_DIR
    
    print(f"   DATA_DIR: {data_dir}")
    if "/mnt/dataset" in str(data_dir):
        print("   ✅ Data volume /mnt/dataset is correctly mounted.")
    else:
        print("   ⚠️  Data volume /mnt/dataset NOT found. Using local project directory.")

    print(f"   CHECKPOINT_DIR: {checkpoint_dir}")
    if "/mnt/weightsandotherstuff" in str(checkpoint_dir):
        print("   ✅ Weights volume /mnt/weightsandotherstuff is correctly mounted.")
    else:
        print("   ⚠️  Weights volume NOT found. Using local project directory.")

    # Check write permissions
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        print("   ✅ Write permission to DATA_DIR verified.")
    except Exception as e:
        print(f"   ❌ Write permission error on DATA_DIR: {e}")

    # 2. Check Kaggle Auth
    print("\n📊 KAGGLE AUTHENTICATION:")
    k_user = os.environ.get("KAGGLE_USERNAME")
    k_key = os.environ.get("KAGGLE_KEY")
    
    if k_user and k_key:
        print(f"   ✅ KAGGLE_USERNAME found: {k_user[:3]}***")
        print(f"   ✅ KAGGLE_KEY found: {k_key[:5]}***")
        
        try:
            import kaggle
            kaggle.api.authenticate()
            print("   ✅ Kaggle API successfully authenticated.")
        except Exception as e:
            print(f"   ❌ Kaggle API authentication FAILED: {e}")
    else:
        print("   ❌ Kaggle credentials missing. Expected: KAGGLE_USERNAME and KAGGLE_KEY")

    # 3. Check Hugging Face Auth
    print("\n🤗 HUGGING FACE AUTHENTICATION:")
    hf_token = os.environ.get("HF_TOKEN")
    
    if hf_token:
        print(f"   ✅ HF_TOKEN found: {hf_token[:5]}***")
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("   ✅ Hugging Face successfully authenticated.")
        except Exception as e:
            print(f"   ❌ Hugging Face authentication FAILED: {e}")
    else:
        print("   ⚠️  HF_TOKEN missing. Gated datasets (like LMSYS) will not be accessible.")

    # 4. Hardware Check
    print("\n💻 HARDWARE:")
    Config.print_hardware_status()

    print("\n" + "=" * 60)
    print("Done!")

if __name__ == "__main__":
    check_setup()
