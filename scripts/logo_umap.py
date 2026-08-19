"""
UMAP of the detector's [CLS] embeddings - the logo core.

Runs the model over held-out human + AI texts, takes the last hidden
layer's [CLS] vector per text, projects to 2D with UMAP, and produces:

  1. umap_ground_truth.png - human vs AI clusters (blue = human, orange = AI)
  2. umap_confidence.png   - the same points colored by the model's own
                             P(AI); the overlap zone in the middle is the
                             hard-negative region the curriculum loop mines.

Slower than the mirror graph: ~15-20 minutes on an A10G (6,000 texts
through DeBERTa-large).

Run it inside a Modal notebook (volumes attached, GPU selected in the
sidebar) by pasting this file's contents into a cell, or:

    exec(open("/mnt/gitstuff/we-have-pangram-at-home/scripts/logo_umap.py").read())

Dependencies (install in a cell first):
    %uv pip install umap-learn

Outputs:
    /mnt/weightsandotherstuff/logo/umap_ground_truth.png
    /mnt/weightsandotherstuff/logo/umap_confidence.png
    /mnt/weightsandotherstuff/logo/umap_data.csv
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datasets import load_dataset

# ----------------------------------------------------------------- paths
# Modal volume layout. rglob does NOT work on volume mounts, so use direct
# paths (matches the eval scripts' defaults).
ckpt_dir = Path("/mnt/weightsandotherstuff/pangram_final/pangram_best")
if not ckpt_dir.exists():
    ckpt_dir = Path("/mnt/weightsandotherstuff/pangram_best")
out_dir = Path("/mnt/weightsandotherstuff/logo")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"checkpoint: {ckpt_dir} (exists: {ckpt_dir.exists()})")
print(f"output dir: {out_dir}")
assert ckpt_dir.exists(), "checkpoint not found - attach the weightsandotherstuff volume"

BG = "#0d0d0d"
HUMAN_COLOR = "#4dabf7"
AI_COLOR = "#ffa94d"
MAX_LENGTH = 512

# Held-out essay benchmark sources (same as scripts/eval_essays.py)
BENCHMARK_SOURCES = {
    "human": [
        {"name": "HC3-Human", "text_field": "human_answers", "is_list_field": True,
         "data_files": ["https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/all/train/0000.parquet"]},
        {"name": "Reddit-Writing", "text_field": "content", "is_list_field": False,
         "filter_fn": lambda x: len(x.get("content", "").split()) > 150,
         "data_files": [f"https://huggingface.co/datasets/webis/tldr-17/resolve/refs%2Fconvert%2Fparquet/default/partial-train/{i:04d}.parquet" for i in range(10)]},
    ],
    "ai": [
        {"name": "HC3-ChatGPT", "text_field": "chatgpt_answers", "is_list_field": True,
         "data_files": ["https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/all/train/0000.parquet"]},
        {"name": "GPT-Wiki-Intro", "text_field": "generated_intro", "is_list_field": False,
         "data_files": ["https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"]},
    ],
}

def load_source(src, max_samples):
    ds = load_dataset("parquet", data_files=src["data_files"], split="train", streaming=True)
    texts, field, is_list = [], src["text_field"], src.get("is_list_field", False)
    filt = src.get("filter_fn", lambda x: True)
    for sample in ds:
        if len(texts) >= max_samples:
            break
        if not filt(sample):
            continue
        text = (sample.get(field, []) or [""])[0] if is_list else sample.get(field, "")
        if text and len(text.strip()) >= 100:
            texts.append(text.strip())
    print(f"   {src['name']}: {len(texts):,} samples")
    return texts

# ------------------------------------------------------------------ model
from transformers import DebertaV2ForSequenceClassification, DebertaV2TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
model = DebertaV2ForSequenceClassification.from_pretrained(str(ckpt_dir), num_labels=2)
tokenizer = DebertaV2TokenizerFast.from_pretrained(str(ckpt_dir))
model.eval().to(device)
model.config.output_hidden_states = True
print("model loaded")

def embed_cls(texts, batch_size=16):
    """[CLS] embeddings + P(AI) probs, halving the batch on CUDA OOM."""
    embs, probs = [], []
    bs = batch_size
    i = 0
    while i < len(texts):
        chunk = texts[i:i + bs]
        try:
            inputs = tokenizer(chunk, truncation=True, max_length=MAX_LENGTH,
                               padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                cls = out.hidden_states[-1][:, 0, :].float().cpu().numpy()
                p = torch.softmax(out.logits, dim=-1)[:, 1].float().cpu().numpy()
            embs.append(cls)
            probs.append(p)
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
    return np.concatenate(embs), np.concatenate(probs)

def style(ax):
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# ------------------------------------------------------------------- data
texts, labels, src_names = [], [], []
for key, label in [("human", 0), ("ai", 1)]:
    for src in BENCHMARK_SOURCES[key]:
        t = load_source(src, 1500)
        texts += t
        labels += [label] * len(t)
        src_names += [src["name"]] * len(t)
n_h = sum(1 for l in labels if l == 0)
print(f"total: {len(texts):,} ({n_h:,} human, {len(labels) - n_h:,} AI)")
assert 0 < n_h < len(labels), "single-class set"

print("extracting [CLS] embeddings ...")
emb, probs = embed_cls(texts)
print("embeddings:", emb.shape)

# ------------------------------------------------------------------- umap
print("running UMAP (this is the slow step) ...")
import umap

xy = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
               metric="cosine", random_state=42).fit_transform(emb)

labels_arr = np.array(labels)
human = labels_arr == 0

# 1. Ground truth clusters
fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
style(ax)
ax.scatter(xy[human, 0], xy[human, 1], s=6, c=HUMAN_COLOR, alpha=0.65,
           linewidths=0, label="human")
ax.scatter(xy[~human, 0], xy[~human, 1], s=6, c=AI_COLOR, alpha=0.65,
           linewidths=0, label="AI")
ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white", fontsize=13)
fig.savefig(out_dir / "umap_ground_truth.png", dpi=300, bbox_inches="tight", facecolor=BG)
plt.show()
plt.close(fig)

# 2. Model confidence (hard-negative zone shows up in the middle)
cmap = LinearSegmentedColormap.from_list("human_to_ai", [HUMAN_COLOR, "#f8f9fa", AI_COLOR])
fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
style(ax)
sc = ax.scatter(xy[:, 0], xy[:, 1], s=6, c=probs, cmap=cmap, vmin=0, vmax=1, linewidths=0)
cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("P(AI)", color="white")
cbar.ax.tick_params(colors="white")
cbar.outline.set_edgecolor("#333333")
fig.savefig(out_dir / "umap_confidence.png", dpi=300, bbox_inches="tight", facecolor=BG)
plt.show()
plt.close(fig)

# ----------------------------------------------------------------- export
pd.DataFrame({"source": src_names, "label": labels_arr, "ai_prob": probs,
              "umap_x": xy[:, 0], "umap_y": xy[:, 1]}
             ).to_csv(out_dir / "umap_data.csv", index=False)

print("\nALL DONE. Outputs in:", out_dir)
print("  umap_ground_truth.png, umap_confidence.png, umap_data.csv")
