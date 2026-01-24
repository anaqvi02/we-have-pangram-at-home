import sys
from pathlib import Path

def check_imports():
    print("--- 🐍 Python Import Diagnostic ---")
    
    # 1. Project Root Check
    root = Path(__file__).resolve().parent.parent
    print(f"Detected Project Root: {root}")
    
    if str(root) not in sys.path:
        print(f"Adding {root} to sys.path...")
        sys.path.append(str(root))
    else:
        print(f"Root already in sys.path.")
        
    # 2. File Existence Check
    files_to_check = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/config.py",
        "src/data/indexing.py"
    ]
    
    files_ok = True
    for f in files_to_check:
        full_path = root / f
        if full_path.exists():
            print(f"✅ Found file: {f}")
        else:
            print(f"❌ MISSING File: {f}")
            files_ok = False
            
    if not files_ok:
        print("CRITICAL: Some source files or package init files are missing.")
        print("Please run: git pull")
        return False
        
    # 3. Import Check
    print("\n--- Attempting Imports ---")
    try:
        import src
        print(f"✅ Imported src package from: {src.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import src: {e}")
        return False
        
    try:
        from src.config import Config
        print(f"✅ Imported Config. DATA_DIR={Config.DATA_DIR}")
    except ImportError as e:
        print(f"❌ Failed to import src.config: {e}")
        return False

    try:
        from src.data.indexing import VectorIndexer
        print(f"✅ Imported VectorIndexer successfully!")
    except ImportError as e:
        print(f"❌ Failed to import src.data.indexing: {e}")
        return False
        
    print("\n🎉 Environment is healthy!")
    return True

if __name__ == "__main__":
    check_imports()
