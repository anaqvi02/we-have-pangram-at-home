# Running on Modal.com

This project supports serverless execution on [Modal](https://modal.com) for faster training (using NVIDIA T4 GPUs) and persistent storage.

## Setup

1.  **Install Modal**:
    ```bash
    pip install modal
    ```
2.  **Authenticate**:
    ```bash
    modal setup
    ```

## Usage

The pipeline is defined in `src/modal_app.py`. You can execute different stages using the `--action` flag.

### 1. Download Data & Build Index (First Run)
Downloads the datasets (WildChat, C4) and builds the vector index on a remote volume.

```bash
modal run src/modal_app.py --action download
modal run src/modal_app.py --action index
```
*Note: This data is stored in a distributed Volume named `pangram-data`.*

### 2. Train Model
Runs the training loop on an NVIDIA A10G GPU (SOTA Performance).

```bash
modal run src/modal_app.py --action train
```

### 3. Run Everything
```bash
modal run src/modal_app.py --action all
```

## Monitoring
Visit the [Modal Dashboard](https://modal.com/dashboard) to view streaming logs, CPU/GPU usage, and volume status.
