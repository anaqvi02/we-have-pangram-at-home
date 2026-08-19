"""
Hard-negative mirror graph for the logo.

Draws the retrieval pairs at the heart of the hard-negative mining:
human texts on the left, their nearest AI "mirrors" from the usearch
index on the right, edges between them. Human node size = the
detector's own P(AI) score (bigger = harder negative).

Fast: ~5 minutes on an A10G (only 60 human texts).

Run it inside a Modal notebook (volumes attached, GPU selected in the
sidebar) by pasting this file's contents into a cell, or:

    exec(open("/mnt/gitstuff/we-have-pangram-at-home/scripts/logo_mirror.py").read())

Dependencies (install in a cell first):
    %uv pip install usearch sentence-transformers

Outputs:
    /mnt/weightsandotherstuff/logo/mirror_graph.png
    /mnt/weightsandotherstuff/logo/mirror_pairs.csv
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset

# ----------------------------------------------------------------- paths
# Modal volume layout. rglob does NOT work on volume mounts, so use direct
# paths (matches the eval scripts' defaults).
ckpt_dir = Path("/mnt/weightsandotherstuff/pangram_final/pangram_best")
if not ckpt_dir.exists():
    ckpt_dir = Path("/mnt/weightsandotherstuff/pangram_best")
index_path = Path("/mnt/dataset/ai_mirrors.usearch")
ai_dir = Path("/mnt/dataset/ai_corpus")
out_dir = Path("/mnt/weightsandotherstuff/logo")
out_dir.mkdir(parents=True, exist_ok=True)

for label, p in [("checkpoint", ckpt_dir), ("index", index_path), ("ai corpus", ai_dir)]:
    print(f"{label}: {p} (exists: {p.exists()})")
assert ckpt_dir.exists(), "checkpoint not found - attach the weightsandotherstuff volume"
assert index_path.exists(), "index not found - attach the dataset volume"

BG = "#0d0d0d"
HUMAN_COLOR = "#4dabf7"
AI_COLOR = "#ffa94d"
MAX_LENGTH = 512

# Human sources (held-out, same as scripts/eval_essays.py)
HUMAN_SOURCES = [
    {"name": "HC3-Human", "text_field": "human_answers", "is_list_field": True,
     "data_files": ["https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/all/train/0000.parquet"]},
    {"name": "Reddit-Writing", "text_field": "content", "is_list_field": False,
     "filter_fn": lambda x: len(x.get("content", "").split()) > 150,
     "data_files": [f"https://huggingface.co/datasets/webis/tldr-17/resolve/refs%2Fconvert%2Fparquet/default/partial-train/{i:04d}.parquet" for i in range(10)]},
]

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

def p_ai(texts, batch_size=16):
    """P(AI) per text, halving the batch on CUDA OOM."""
    probs = []
    bs = batch_size
    i = 0
    while i < len(texts):
        chunk = texts[i:i + bs]
        try:
            inputs = tokenizer(chunk, truncation=True, max_length=MAX_LENGTH,
                               padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy().tolist())
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
    return np.array(probs)

# ----------------------------------------------------------------- humans
n_humans, top_k = 60, 3
texts, src_names = [], []
for src in HUMAN_SOURCES:
    t = load_source(src, n_humans)
    texts += t
    src_names += [src["name"]] * len(t)
texts, src_names = texts[:n_humans], src_names[:n_humans]
print(f"{len(texts)} human texts")

print("scoring humans with the detector ...")
probs = p_ai(texts)

# ---------------------------------------------------------------- mirrors
print("searching AI mirrors ...")
from usearch.index import Index
from sentence_transformers import SentenceTransformer

st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = Index(ndim=st.get_sentence_embedding_dimension(), metric="cos", dtype="f16")
index.load(str(index_path))
parquet_files = sorted(str(p) for p in ai_dir.glob("*.parquet"))
print(f"corpus: {len(parquet_files)} parquet files")
ds = load_dataset("parquet", data_files=parquet_files, split="train")

embs = st.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                 batch_size=256, show_progress_bar=False)
matches = index.search(embs, top_k)
keys = np.atleast_2d(np.asarray(matches.keys))
dists = np.atleast_2d(np.asarray(matches.distances))
if keys.shape[0] == 1 and keys.shape[1] != top_k:
    keys, dists = keys.T, dists.T

mirror_id_of_text, mirror_texts, pairs = {}, {}, []
for h in range(len(texts)):
    for k_i in range(keys.shape[1]):
        key = int(keys[h, k_i])
        if key == -1:
            continue
        try:
            mtext = ds[key]["text"]
        except (IndexError, KeyError):
            continue
        if mtext not in mirror_id_of_text:
            mirror_id_of_text[mtext] = len(mirror_id_of_text)
            mirror_texts[mirror_id_of_text[mtext]] = mtext
        sim = 1.0 - float(dists[h, k_i])
        pairs.append((h, mirror_id_of_text[mtext], sim))
print(f"{len(pairs)} pairs, {len(mirror_texts)} unique mirrors")
assert pairs, "no pairs - index/corpus mismatch?"

# ---------------------------------------------------------------- layout
# Humans sorted by P(AI): the hardest (most AI-looking) sit in the middle.
rng = np.random.default_rng(42)
order = np.argsort(probs, kind="stable")
human_y = {h: rank + rng.uniform(-0.25, 0.25) for rank, h in enumerate(order)}
mirror_y = {m: float(np.mean([human_y[h] for h, mm, _ in pairs if mm == m])) for m in mirror_texts}
mirror_ids = sorted(mirror_texts, key=lambda m: mirror_y[m])

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
for h, m, sim in pairs:
    ax.plot([0, 1], [human_y[h], mirror_y[m]], color="#888888", lw=0.6,
            alpha=0.15 + 0.55 * sim, zorder=1)
ax.scatter([0.0] * len(texts), [human_y[h] for h in range(len(texts))],
           s=[60 + 140 * float(probs[h]) for h in range(len(texts))],
           c=HUMAN_COLOR, alpha=0.9, edgecolors="none", zorder=2)
ax.scatter([1.0] * len(mirror_ids), [mirror_y[m] for m in mirror_ids],
           s=22, c=AI_COLOR, alpha=0.7, edgecolors="none", zorder=2)
fig.savefig(out_dir / "mirror_graph.png", dpi=300, bbox_inches="tight", facecolor=BG)
plt.show()

# ----------------------------------------------------------------- export
pd.DataFrame([{"human_source": src_names[h], "human_text": texts[h],
               "human_p_ai": float(probs[h]), "mirror_text": mirror_texts[m],
               "similarity": float(sim)} for h, m, sim in pairs]
             ).to_csv(out_dir / "mirror_pairs.csv", index=False)

print("\nALL DONE. Outputs in:", out_dir)
print("  mirror_graph.png, mirror_pairs.csv")
