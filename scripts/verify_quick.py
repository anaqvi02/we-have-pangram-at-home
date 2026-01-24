import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config

def verify():
    print("--- 🔍 Verifying Data Integrity ---")
    
    datasets = {
        "AI Corpus": Config.AI_DATASET_PATH,
        "Human Corpus": Config.HUMAN_DATASET_PATH
    }
    
    all_good = True
    
    for name, path in datasets.items():
        print(f"\nChecking {name} at {path}...")
        
        if not path.exists():
            print(f"❌ Directory not found: {path}")
            all_good = False
            continue
            
        files = list(path.glob("part_*.parquet"))
        if not files:
            # Fallback for legacy naming if needed, but we expect part_*.parquet
            files = list(path.glob("*.parquet"))
            
        if not files:
            print(f"❌ No parquet files found in {path}")
            all_good = False
            continue
            
        print(f"✅ Found {len(files)} partition files.")
        
        # Verify readability of the first file
        try:
            first_file = files[0]
            df = pd.read_parquet(first_file)
            if len(df) == 0:
                print(f"⚠️ Warning: First file {first_file.name} is empty.")
            else:
                print(f"✅ Read sample file ({first_file.name}): {len(df)} rows.")
                print(f"   Columns: {list(df.columns)}")
                if 'text' not in df.columns or 'label' not in df.columns:
                     print(f"❌ Missing required columns 'text' or 'label'")
                     all_good = False
        except Exception as e:
            print(f"❌ Failed to read {files[0].name}: {e}")
            all_good = False

    print("\n-------------------------------")
    if all_good:
        print("🎉 Data Verified! Ready for training.")
        return True
    else:
        print("running verify_ops.py... FAILED.")
        print("Some checks failed. See above for details.")
        return False

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)
