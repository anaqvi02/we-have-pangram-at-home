"""
Extract embeddings from the trained detector and project them to 2D with UMAP.

Produces two logo-ready visuals from the model itself:

  1. umap_ground_truth.png  - human vs AI clusters (blue = human, orange = AI)
  2. umap_confidence.png    - the same points colored by the model's own
                              P(AI). The overlap zone in the middle is the
                              hard-negative region the curriculum loop mines.

Also saves umap_data.csv (source, label, ai_prob, umap_x, umap_y) so the
points can be reused for other visuals (e.g. the mirror graph).

Run from the repo root (Modal mounts the repo at /):
    modal run scripts/extract_embeddings.py

If the volume name below does not exist on your account, list the real name
with `modal volume list` and pass it with --volume-name.

Download the results:
    modal volume get weightsandotherstuff logo/umap_ground_truth.png
    modal volume get weightsandotherstuff logo/umap_confidence.png
    modal volume get weightsandotherstuff logo/umap_data.csv

The mirror graph (--action mirror) draws the retrieval pairs at the heart of
the hard-negative mining: human texts on the left, the AI "mirrors" found by
the usearch index on the right, edges between them. Human node size is the
detector's own P(AI) score (bigger = harder negative). It needs the index at
/data/ai_mirrors.usearch on the pangram-data volume:
    modal run scripts/extract_embeddings.py --action mirror
"""

import modal
from pathlib import Path

image = (
    modal.Image.debian_slim()
    .apt_install("git", "libgomp1")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "umap-learn",
        "matplotlib",
        "sentence-transformers",
        "usearch",
        "tqdm",
    )
)

app = modal.App("we-have-pangram-logo")

# The volume that holds the trained checkpoint. Training saved the best
# model to /mnt/weightsandotherstuff/pangram_final/pangram_best.
WEIGHTS_VOLUME = modal.Volume.from_name("weightsandotherstuff", create_if_missing=False)
WEIGHTS_ROOT = Path("/mnt/weightsandotherstuff")
OUT_DIR = WEIGHTS_ROOT / "logo"

# The volume that holds the AI corpus and the retrieval index
# (ai_mirrors.usearch), mounted at /data like in modal_app.py.
DATA_VOLUME = modal.Volume.from_name("pangram-data", create_if_missing=False)
DATA_ROOT = Path("/data")

# Theme constants, matching the repo's other plots (plot_training_log.py).
BG = "#0d0d0d"
HUMAN_COLOR = "#4dabf7"
AI_COLOR = "#ffa94d"


@app.function(
    image=image,
    volumes={WEIGHTS_ROOT: WEIGHTS_VOLUME, DATA_ROOT: DATA_VOLUME},
    gpu="A10G",
    timeout=3600,
)
def mirror_graph(n_humans: int = 60, top_k: int = 3, seed: int = 42):
    """Draw the hard-negative mirror pairs: humans -> nearest AI mirrors."""
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.config import Config
    from src.model.detector import PangramDetector
    from src.data.indexing import (
        VectorIndexer,
        _read_index_corpus_manifest,
        _sorted_parquet_files,
    )
    from scripts.eval_essays import BENCHMARK_SOURCES, load_benchmark_source
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    INDEX_PATH = DATA_ROOT / "ai_mirrors.usearch"
    AI_DIR = DATA_ROOT / "ai_corpus"

    # ---------------------------------------------------------------- index
    print(f"Loading index from {INDEX_PATH} ...")
    if not INDEX_PATH.exists():
        raise SystemExit(
            f"Index not found at {INDEX_PATH}. Build it first "
            "(modal run src/modal_app.py --action index)."
        )
    try:
        indexer = VectorIndexer.load(INDEX_PATH, parquet_file=AI_DIR)
    except Exception as e:
        print(f"Standard load failed ({e}); falling back to explicit parquet list ...")
        indexer = VectorIndexer()
        indexer.index.load(str(INDEX_PATH))
        manifest = _read_index_corpus_manifest(INDEX_PATH)
        if manifest and all(Path(f).exists() for f in manifest.get("parquet_files", [])):
            data_files = manifest["parquet_files"]
        else:
            data_files = _sorted_parquet_files(AI_DIR)
        indexer.dataset = load_dataset("parquet", data_files=data_files, split="train")
        indexer.use_parquet = True

    # ------------------------------------------------------------ human side
    texts, src_names = [], []
    for src in BENCHMARK_SOURCES["human"]:
        t = load_benchmark_source(src, n_humans)
        texts.extend(t)
        src_names.extend([src["name"]] * len(t))
    texts = texts[:n_humans]
    src_names = src_names[:n_humans]
    if not texts:
        raise SystemExit("No human samples loaded.")
    print(f"Using {len(texts)} human texts")

    # Hardness = the detector's own P(AI) on each human text.
    ckpt = WEIGHTS_ROOT / "pangram_final" / "pangram_best"
    print(f"Scoring humans with the detector ({ckpt}) ...")
    detector = PangramDetector.load(str(ckpt))
    detector.model.eval()
    tokenizer = detector.tokenizer
    model = detector.model
    device = detector.config.DEVICE

    probs = []
    bs = 16
    i = 0
    while i < len(texts):
        chunk = texts[i : i + bs]
        try:
            inputs = tokenizer(
                chunk,
                truncation=True,
                max_length=Config.MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                p = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
            probs.extend(p.tolist())
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
    probs = np.array(probs)

    # ------------------------------------------------------------ AI mirrors
    print("Searching AI mirrors (top-%d per human) ..." % top_k)
    embs = indexer.model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=False,
    )
    matches = indexer.index.search(embs, top_k)
    keys = np.atleast_2d(np.asarray(matches.keys))
    dists = np.atleast_2d(np.asarray(matches.distances))
    # usearch returns shape (k,) for a single query and (n, k) for a batch;
    # normalize the single-query case to (n, 1).
    if keys.shape[0] == 1 and keys.shape[1] > 1 and keys.shape[1] != top_k:
        keys = keys.T
        dists = dists.T

    mirror_id_of_text = {}
    mirror_texts = {}
    pairs = []  # (human_idx, mirror_id, similarity)
    for h in range(len(texts)):
        for k_i in range(keys.shape[1]):
            key = int(keys[h, k_i])
            if key == -1:
                continue
            try:
                mtext = indexer.dataset[key]["text"]
            except (IndexError, KeyError):
                continue
            if mtext not in mirror_id_of_text:
                mid = len(mirror_id_of_text)
                mirror_id_of_text[mtext] = mid
                mirror_texts[mid] = mtext
            sim = 1.0 - float(dists[h, k_i])
            pairs.append((h, mirror_id_of_text[mtext], sim))

    if not pairs:
        raise SystemExit("No mirror pairs found. Is the index empty or mismatched?")
    print(f"{len(pairs)} pairs, {len(mirror_texts)} unique mirrors")

    # --------------------------------------------------------------- layout
    # Humans sorted by P(AI): the hardest (most AI-looking) sit in the middle.
    rng = np.random.default_rng(seed)
    order = np.argsort(probs, kind="stable")
    human_y = {h: rank + rng.uniform(-0.25, 0.25) for rank, h in enumerate(order)}
    mirror_y = {
        mid: float(np.mean([human_y[h] for h, m, _ in pairs if m == mid]))
        for mid in mirror_texts
    }
    mirror_ids = sorted(mirror_texts, key=lambda m: mirror_y[m])

    # ----------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    for h, m, sim in pairs:
        ax.plot(
            [0, 1], [human_y[h], mirror_y[m]],
            color="#888888", lw=0.6,
            alpha=0.15 + 0.55 * float(sim), zorder=1,
        )
    human_sizes = [60 + 140 * float(probs[h]) for h in range(len(texts))]
    ax.scatter(
        [0.0] * len(texts), [human_y[h] for h in range(len(texts))],
        s=human_sizes, c=HUMAN_COLOR, alpha=0.9, edgecolors="none", zorder=2,
    )
    ax.scatter(
        [1.0] * len(mirror_ids), [mirror_y[m] for m in mirror_ids],
        s=22, c=AI_COLOR, alpha=0.7, edgecolors="none", zorder=2,
    )
    fig.savefig(OUT_DIR / "mirror_graph.png", dpi=300,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # ---------------------------------------------------------------- export
    pd.DataFrame(
        [
            {
                "human_source": src_names[h],
                "human_text": texts[h],
                "human_p_ai": float(probs[h]),
                "mirror_text": mirror_texts[m],
                "similarity": float(sim),
            }
            for h, m, sim in pairs
        ]
    ).to_csv(OUT_DIR / "mirror_pairs.csv", index=False)

    WEIGHTS_VOLUME.commit()
    DATA_VOLUME.commit()
    print("Saved to the volume:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"   {f}")


@app.function(
    image=image,
    volumes={WEIGHTS_ROOT: WEIGHTS_VOLUME},
    gpu="A10G",
    timeout=3600,
)
def extract(samples_per_source: int = 1500, batch_size: int = 16, seed: int = 42):
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import umap
    from tqdm import tqdm

    from src.config import Config
    from src.model.detector import PangramDetector
    from scripts.eval_essays import BENCHMARK_SOURCES, load_benchmark_source

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- model
    ckpt = WEIGHTS_ROOT / "pangram_final" / "pangram_best"
    print(f"Loading checkpoint from {ckpt} ...")
    detector = PangramDetector.load(str(ckpt))
    detector.model.eval()
    detector.model.config.output_hidden_states = True
    device = detector.config.DEVICE

    # ------------------------------------------------------------------ data
    # Reuse the held-out essay benchmark sources (HC3, Reddit, GPT-wiki-intro).
    # These are texts the model never saw during training.
    texts, labels, src_names = [], [], []
    for label_key, label in [("human", 0), ("ai", 1)]:
        for src in BENCHMARK_SOURCES[label_key]:
            t = load_benchmark_source(src, samples_per_source)
            texts.extend(t)
            labels.extend([label] * len(t))
            src_names.extend([src["name"]] * len(t))
            print(f"   {src['name']}: {len(t):,} samples (label={label})")

    if not texts:
        raise SystemExit("No samples loaded. Check dataset availability.")

    n_human = sum(1 for l in labels if l == 0)
    n_ai = len(labels) - n_human
    if n_human == 0 or n_ai == 0:
        raise SystemExit(
            f"Single-class sample set ({n_human:,} human, {n_ai:,} AI). "
            "A UMAP of one class is meaningless."
        )
    print(f"Total: {len(texts):,} samples ({n_human:,} human, {n_ai:,} AI)")

    # ------------------------------------------------------------- embeddings
    # One forward pass, grab the last hidden layer's [CLS] vector per text.
    # Batch size halves automatically on CUDA OOM (DeBERTa-large attention is
    # memory-hungry at 512 tokens).
    tokenizer = detector.tokenizer
    model = detector.model

    def embed(texts_in):
        embs, probs = [], []
        bs = batch_size
        i = 0
        while i < len(texts_in):
            chunk = texts_in[i : i + bs]
            try:
                inputs = tokenizer(
                    chunk,
                    truncation=True,
                    max_length=Config.MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden = outputs.hidden_states[-1]  # [B, L, 1024]
                    cls = hidden[:, 0, :].float().cpu().numpy()
                    p = torch.softmax(outputs.logits, dim=-1)[:, 1].float().cpu().numpy()
                embs.append(cls)
                probs.append(p)
                i += len(chunk)
            except torch.cuda.OutOfMemoryError:
                if bs <= 1:
                    raise
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
        return np.concatenate(embs, axis=0), np.concatenate(probs, axis=0)

    print("Extracting [CLS] embeddings ...")
    emb, probs = embed(texts)
    print(f"Embeddings: {emb.shape}")

    # ------------------------------------------------------------------ umap
    print("Running UMAP (cosine, n_neighbors=15) ...")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="cosine", random_state=seed,
    )
    xy = reducer.fit_transform(emb)
    print(f"UMAP done: {xy.shape}")

    # ---------------------------------------------------------------- plots
    labels_arr = np.array(labels)
    human_mask = labels_arr == 0

    def style(ax):
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # 1. Ground truth clusters
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    style(ax)
    ax.scatter(xy[human_mask, 0], xy[human_mask, 1], s=6, c=HUMAN_COLOR,
               alpha=0.65, linewidths=0, label="human")
    ax.scatter(xy[~human_mask, 0], xy[~human_mask, 1], s=6, c=AI_COLOR,
               alpha=0.65, linewidths=0, label="AI")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white",
              fontsize=13, loc="best")
    fig.savefig(OUT_DIR / "umap_ground_truth.png", dpi=300,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # 2. Model confidence (hard-negative zone shows up in the middle)
    cmap = LinearSegmentedColormap.from_list("human_to_ai", [HUMAN_COLOR, "#f8f9fa", AI_COLOR])
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    style(ax)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=6, c=probs, cmap=cmap,
                    vmin=0, vmax=1, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("P(AI)", color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("#333333")
    fig.savefig(OUT_DIR / "umap_confidence.png", dpi=300,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # ---------------------------------------------------------------- export
    pd.DataFrame({
        "source": src_names,
        "label": labels_arr,
        "ai_prob": probs,
        "umap_x": xy[:, 0],
        "umap_y": xy[:, 1],
    }).to_csv(OUT_DIR / "umap_data.csv", index=False)

    WEIGHTS_VOLUME.commit()
    print("Saved to the volume:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"   {f}")


@app.local_entrypoint()
def main(
    action: str = "umap",
    samples_per_source: int = 1500,
    batch_size: int = 16,
    seed: int = 42,
    n_humans: int = 60,
    top_k: int = 3,
):
    """Run the logo extraction. Actions: umap | mirror | all."""
    if action in ("umap", "all"):
        print(f"Triggering embedding extraction (samples/source={samples_per_source}) ...")
        extract.remote(samples_per_source=samples_per_source, batch_size=batch_size, seed=seed)
        print("UMAP done. Download with:")
        print("  modal volume get weightsandotherstuff logo/umap_ground_truth.png")
        print("  modal volume get weightsandotherstuff logo/umap_confidence.png")
        print("  modal volume get weightsandotherstuff logo/umap_data.csv")
    if action in ("mirror", "all"):
        print(f"Triggering mirror graph ({n_humans} humans, top-{top_k}) ...")
        mirror_graph.remote(n_humans=n_humans, top_k=top_k, seed=seed)
        print("Mirror graph done. Download with:")
        print("  modal volume get weightsandotherstuff logo/mirror_graph.png")
        print("  modal volume get weightsandotherstuff logo/mirror_pairs.csv")
