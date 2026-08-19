"""
Hard-negative mirror graph for the logo.

Draws the retrieval pairs at the heart of the hard-negative mining:
human texts on the left, their nearest AI "mirrors" from the usearch
index on the right, with gradient bezier edges between them. Human
x-position and node size encode the detector's own P(AI) (hard negatives
hug the decision axis at x = 0.5). Mirrors are either reflected across
that axis (MEASURED_MIRRORS=False) or placed at their own measured P(AI)
(MEASURED_MIRRORS=True).

~5-7 minutes on an A10G at 1,000 human texts (adds ~2-3 min when
MEASURED_MIRRORS=True, which scores ~2,000 mirror texts).

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
n_humans, top_k = 1000, 3
# True = y is random, so only x carries the confidence signal (no cone/bottle
# shape). False = y is sorted by P(AI), which correlates with x and produces
# the mirrored-cone silhouette.
SHUFFLE_Y = False
# True = run the detector over the mirror texts too and place them at their
# own measured P(AI) (breaks the symmetry; mirrors cluster near the right
# edge). False = reflect the paired human's position across the axis (the
# "mirror" metaphor). Adds ~2-3 min of inference when True.
MEASURED_MIRRORS = True
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

mirror_probs = np.array([])
if MEASURED_MIRRORS:
    print(f"scoring {len(mirror_texts)} mirror texts with the detector ...")
    mirror_probs = p_ai([mirror_texts[m] for m in mirror_texts])

# ---------------------------------------------------------------- layout
# The mirror metaphor: x-position = the model's confidence. Humans live on
# the left half, their AI mirrors are reflected across the decision axis at
# x = 0.5. A hard negative (P(AI) near 1) sits right next to the axis - the
# model nearly reflected it across.
rng = np.random.default_rng(42)
order = np.argsort(probs, kind="stable")
if SHUFFLE_Y:
    human_y = {h: rng.uniform(-0.5, len(texts) - 0.5) for h in range(len(texts))}
else:
    human_y = {h: rank + rng.uniform(-0.3, 0.3) for rank, h in enumerate(order)}

def x_human(p):
    return 0.08 + 0.39 * float(p)  # 0.08 = sure human .. 0.47 = hard negative

human_x = {h: x_human(probs[h]) for h in range(len(texts))}

if MEASURED_MIRRORS:
    # Measured P(AI) saturates near 1 for AI text (almost every mirror scores
    # ~0.99+), so a linear x-map collapses the whole orange side into a flat
    # line. Spread mirrors across the right half by their RANK in P(AI)
    # instead - keeps the ordering meaning while making the spread visible.
    mirror_ids_by_p = sorted(mirror_texts, key=lambda m: mirror_probs[m])
    n_mir = max(len(mirror_ids_by_p) - 1, 1)
    mirror_x = {m: 0.53 + 0.39 * (rank / n_mir) for rank, m in enumerate(mirror_ids_by_p)}
else:
    mirror_x = {m: float(np.mean([1.0 - human_x[h] for h, mm, _ in pairs if mm == m]))
                for m in mirror_texts}
mirror_y = {m: float(np.mean([human_y[h] for h, mm, _ in pairs if mm == m]))
            for m in mirror_texts}
mirror_ids = sorted(mirror_texts, key=lambda m: mirror_y[m])

# ------------------------------------------------------------------ plot
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

def bezier(p0, p1, ctrl, n=32):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1

fig, ax = plt.subplots(figsize=(12, 12), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

# the decision axis: where a text "reflects" from human to AI
ax.axvline(0.5, color="#555555", lw=1.0, ls=(0, (4, 4)), alpha=0.55, zorder=0)

# edges: bezier arcs that fade from human-blue to AI-orange along the way
segments, seg_colors = [], []
for h, m, sim in pairs:
    p0 = np.array([human_x[h], human_y[h]])
    p1 = np.array([mirror_x[m], mirror_y[m]])
    ctrl = np.array([(p0[0] + p1[0]) / 2, max(p0[1], p1[1]) + 0.6])
    pts = bezier(p0, p1, ctrl)
    for i in range(len(pts) - 1):
        t = i / (len(pts) - 1)
        col = np.array(mcolors.to_rgb(HUMAN_COLOR)) * (1 - t) + \
              np.array(mcolors.to_rgb(AI_COLOR)) * t
        segments.append([pts[i], pts[i + 1]])
        seg_colors.append((*col, min(1.0, max(0.03, 0.05 + 0.3 * float(sim)))))
ax.add_collection(LineCollection(segments, colors=seg_colors, lw=0.4, zorder=1))

# human nodes (size = P(AI), the harder the bigger) with a soft glow
hx = np.array([human_x[h] for h in range(len(texts))])
hy = np.array([human_y[h] for h in range(len(texts))])
sizes = np.array([6 + 18 * float(probs[h]) for h in range(len(texts))])
ax.scatter(hx, hy, s=sizes * 5, c=HUMAN_COLOR, alpha=0.10, edgecolors="none", zorder=2)
ax.scatter(hx, hy, s=sizes, c=HUMAN_COLOR, alpha=0.9, edgecolors="none", zorder=3)

# AI mirror nodes (reflections, uniform) with a soft glow
mx = np.array([mirror_x[m] for m in mirror_ids])
my = np.array([mirror_y[m] for m in mirror_ids])
ax.scatter(mx, my, s=4 * 5, c=AI_COLOR, alpha=0.08, edgecolors="none", zorder=2)
ax.scatter(mx, my, s=4, c=AI_COLOR, alpha=0.8, edgecolors="none", zorder=3)

ax.set_xlim(-0.06, 1.06)
ax.set_ylim(-3, len(texts) + 3)
fig.savefig(out_dir / "mirror_graph.png", dpi=300, bbox_inches="tight", facecolor=BG)
plt.show()

# ----------------------------------------------------------------- export
pd.DataFrame([{"human_source": src_names[h], "human_text": texts[h],
               "human_p_ai": float(probs[h]), "mirror_text": mirror_texts[m],
               "similarity": float(sim)} for h, m, sim in pairs]
             ).to_csv(out_dir / "mirror_pairs.csv", index=False)

print("\nALL DONE. Outputs in:", out_dir)
print("  mirror_graph.png, mirror_pairs.csv")
