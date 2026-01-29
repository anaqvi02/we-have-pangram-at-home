# 🚀 Setup & Training Guide

This guide walks you through the end-to-end process of setting up the **Pangram Essay Detector**, from data collection to model evaluation.

---

## 📋 1. Installation

Ensure you have a modern Python environment (3.10+) and install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 2. Data Collection (Balanced Essay Dataset)

The provided script downloads and filters data specifically for **high-school and university level English essays**. It automatically balances the dataset between Human and AI sources.

### Option A: Complete Download (HuggingFace)
This will fetch ~200k samples per class from Wikipedia, arXiv, FineWeb-Edu (Human) and Cosmopedia, LMSYS, WildChat (AI).

```bash
python scripts/download_data.py --target 200000
```

### Option B: Using Local Kaggle Files
If you have the PERSUADE or AI Generated Essays datasets locally:

```bash
python scripts/download_data.py --persuade /path/to/persuade.csv --ai-essays /path/to/ai_essays.csv
```

---

## 🔍 3. Data Verification

Before training, verify that the data was saved correctly and is readable:

```bash
python scripts/verify_quick.py
```
*If this prints "🎉 Data Verified!", you are ready to proceed.*

---

## 🏗️ 4. Build Vector Index

The model uses a curriculum learning approach that requires a search index of AI samples for "mining" hard negatives. Building this index is an essential one-time step.

```bash
python scripts/build_index.py
```

---

## 🎓 5. Training

Start the training pipeline. The script will automatically:
1. Load and shuffle all parquet shards.
2. Initialize a DeBERTa-v3-large model.
3. Run a curriculum loop (Train → Evaluate → Mine Hard Negatives → Augment).
4. Auto-resume if a checkpoint exists.

```bash
python train.py --epochs 3
```

---

## 🧪 6. Inference & Evaluation

To test the model on a small evaluation slice and calculate performance metrics (like FPR @ 95% Recall):

```bash
python evaluate.py --model_path checkpoints/pangram_final
```

---

## 📈 Monitoring

During training, progress is logged to `training_log.csv`. You can monitor:
- **Accuracy**: Should ideally exceed 95%.
- **Loss**: Should steadily decrease.
- **Dataset Size**: Will grow as the miner finds harder samples for the model.

---

## 💡 Troubleshooting

- **Out of Memory (OOM)**: If training crashes, reduce `BATCH_SIZE` or increase `GRAD_ACCUMULATION` in `src/config.py`.
- **Permission Denied (Git)**: If you cannot push changes, check your SSH keys or personal access tokens for the repository.
- **No GPU found**: The scripts will default to CPU, but training DeBERTa-v3-large on CPU is extremely slow. Ensure `MPS` (Mac) or `CUDA` (NVIDIA) is supported by your installation.
