#!/bin/bash
set -e # Exit on error
export PYTHONPATH=$PYTHONPATH:.

echo "=========================================="
echo "   Pangram Autonomous Pipeline Started    "
echo "=========================================="

# 0. Data Download
# 0. Data Download
# Check for part_0.parquet as indicator of data presence
if [ -f "data/ai_corpus/wildchat_part_0.parquet" ] && [ -f "data/human_corpus/c4_part_0.parquet" ]; then
    echo "Data detected (partitioned). Skipping download."
else
    echo ""
    echo "--- [0/3] Downloading Data (Target: 1M Samples) ---"
    python3 scripts/download_data.py
fi

# 1. Build Index (Force CPU for stability as verified)
echo ""
echo "--- [1/3] Building Vector Index (CPU Mode) ---"
if [ ! -f "data/ai_mirrors.usearch" ]; then
    echo "Index not found. Starting build..."
    export FORCE_CPU=1
    python3 scripts/build_index.py
    unset FORCE_CPU
else
    echo "Index found at data/ai_mirrors.usearch. Skipping build."
    # Optional: Logic to force rebuild if parquet is newer? 
    # For now, assume if it exists, it's good.
fi

# 2. Train (MPS Enabled for Speed)
echo ""
echo "--- [2/3] Starting Curriculum Training (MPS Mode) ---"
# We run for 3 epochs to balance convergence vs overfitting (User Request)
python3 train.py --epochs 3 --index_path data/ai_mirrors.usearch

# 3. Evaluate
echo ""
echo "--- [3/3] Final Evaluation ---"
python3 evaluate.py --model_path checkpoints/pangram_m3

echo ""
echo "=========================================="
echo "      Autonomous Pipeline Complete        "
echo "=========================================="
