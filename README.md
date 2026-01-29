# We Have Pangram at Home

AI-generated text detector built for efficient training and inference on Apple Silicon (M-series) Macs. This project demonstrates how to build a competitive detector using open-source models (DeBERTa-v3) and data (WildChat, C4) with a fraction of the resources.

## Quick Start

For a detailed, step-by-step walkthrough of the data collection and training process, see the [**Setup & Training Guide**](SETUP_GUIDE.md).

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/anaqvi02/we-have-pangram-at-home.git
cd we-have-pangram-at-home
pip install -r requirements.txt
```

### 2. Retrieve Data

The project uses open-source datasets (AllenAI's WildChat and C4). Run the download script to fetch and partition ~2 million samples:

```bash
python scripts/download_data.py
```
*Note: This downloads approximately 2.5GB of data to `data/`.*

### 3. Build Vector Index

Build the search index for hard-negative mining (optimized for Apple Silicon):

```bash
python scripts/build_index.py
```

### 4. Train Model

Train the detector with the curriculum learning loop:

```bash
python train.py --epochs 3
```

## Architecture

*   **Model**: Microsoft DeBERTa-v3-base
*   **Training**: Curriculum Learning (mining hard negatives vs. true positives)
*   **Hardware Optimization**:
    *   **MPS (Metal)**: Uses Apple GPU for training and inference.
    *   **Streaming**: Zero-copy data loading using Apache Arrow (Parquet).
    *   **Vector Search**: `usearch` for fast, low-memory similarity search.

## Author

[anaqvi02](https://github.com/anaqvi02)
