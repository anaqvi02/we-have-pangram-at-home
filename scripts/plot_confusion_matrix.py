#!/usr/bin/env python3
"""Plot a confusion matrix heatmap for the AI-text detector eval runs.

Reads the counts either from a saved eval JSON (essay_eval_*.json or
raid_eval_*.json) or from --tp/--fp/--fn/--tn flags. Matches the dark
terminal aesthetic used across the project docs.

Usage:
    python scripts/plot_confusion_matrix.py --tp 1659 --fp 242 --fn 341 --tn 1758
    python scripts/plot_confusion_matrix.py --json checkpoints/essay_eval_20260818_205103.json
    python scripts/plot_confusion_matrix.py --json checkpoints/raid_eval_....json --title "RAID (clean)"

Output: confusion_matrix.png (or --out path).
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp", type=int, help="true positives (AI correctly flagged)")
    p.add_argument("--fp", type=int, help="false positives (human flagged as AI)")
    p.add_argument("--fn", type=int, help="false negatives (AI missed)")
    p.add_argument("--tn", type=int, help="true negatives (human correctly left alone)")
    p.add_argument("--json", dest="json_path", help="eval JSON with a confusion_matrix field")
    p.add_argument("--out", default="confusion_matrix.png", help="output PNG path")
    p.add_argument("--title", default="Confusion Matrix", help="plot title")
    args = p.parse_args()

    if args.json_path:
        data = json.load(open(args.json_path, encoding="utf-8"))
        cm = None
        for key in ("overall", "confusion_matrix"):
            if key in data:
                cm = data[key].get("confusion_matrix") if isinstance(data[key], dict) else None
                if cm:
                    break
        if cm is None:
            cm = data.get("confusion_matrix")
        if cm is None:
            sys.exit(f"no confusion_matrix found in {args.json_path}")
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    elif None not in (args.tp, args.fp, args.fn, args.tn):
        tp, fp, fn, tn = args.tp, args.fp, args.fn, args.tn
    else:
        sys.exit("pass --tp/--fp/--fn/--tn or --json")

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array([[tn, fp], [fn, tp]])
    total = tn + fp + fn + tp

    fig, ax = plt.subplots(figsize=(6, 5.2), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(tn, tp) * 1.25)

    labels = ["Human", "AI"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, color="white")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actually", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = 100.0 * count / total
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", color="white", fontsize=15)

    ax.set_title(args.title, color="white", pad=14, fontsize=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("#333333")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"saved {out}  (TN {tn:,} / FP {fp:,} / FN {fn:,} / TP {tp:,}, "
          f"n={total:,})")


if __name__ == "__main__":
    main()
