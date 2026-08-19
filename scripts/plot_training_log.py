#!/usr/bin/env python3
"""Plot training curves from the curriculum-training log (training_log.csv).

The trainer writes one row per epoch with columns:
    epoch,train_loss,val_loss,val_acc,dataset_size,mined_samples

Shows the curriculum story on one figure: loss dropping (left) while the
training set grows as hard-negative mining adds samples (right). Matches
the dark terminal aesthetic used across the project docs.

Usage:
    python scripts/plot_training_log.py --csv checkpoints/pangram_final/training_log.csv
    python scripts/plot_training_log.py --csv training_log.csv --out training_curves.png

Output: training_curves.png (default --out).
"""
import argparse
import csv
import sys
from pathlib import Path

EXPECTED = ("epoch", "train_loss", "val_loss", "val_acc", "dataset_size",
            "mined_samples")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="training_log.csv", help="path to training_log.csv")
    p.add_argument("--out", default="training_curves.png", help="output PNG path")
    p.add_argument("--title", default="Curriculum Training", help="plot title")
    args = p.parse_args()

    path = Path(args.csv)
    if not path.exists():
        sys.exit(f"no training log at {path}")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} is empty")

    missing = [c for c in EXPECTED if c not in rows[0]]
    if missing:
        sys.exit(f"{path} missing columns: {', '.join(missing)}; "
                 f"expected {', '.join(EXPECTED)}")

    epoch = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]
    dataset_size = [int(r["dataset_size"]) for r in rows]
    mined = [int(r["mined_samples"]) for r in rows]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), facecolor="#0d0d0d",
                                   sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333333")

    # top: loss
    ax1.plot(epoch, train_loss, "-o", color="#4dabf7", label="train loss", lw=2, ms=4)
    ax1.plot(epoch, val_loss, "-s", color="#ffa94d", label="val loss", lw=2, ms=4)
    ax1.set_ylabel("loss", color="white")
    ax1.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")
    ax1.set_title(args.title, color="white", pad=12, fontsize=14)

    # bottom: curriculum growth + accuracy
    ax2.plot(epoch, dataset_size, "-^", color="#69db7c", label="dataset size",
             lw=2, ms=4)
    ax2.plot(epoch, mined, "--", color="#b197fc", label="mined samples", lw=2)
    ax2.set_xlabel("epoch", color="white")
    ax2.set_ylabel("samples", color="white")
    ax2.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

    # annotate val accuracy on the top panel (secondary y axis)
    ax1b = ax1.twinx()
    ax1b.set_facecolor("#0d0d0d")
    ax1b.plot(epoch, [a * 100 for a in val_acc], "x:", color="#fcc419",
              label="val acc %", ms=5)
    ax1b.set_ylabel("val acc (%)", color="#fcc419")
    ax1b.tick_params(colors="#fcc419")
    ax1b.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#fcc419",
                loc="lower left")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"saved {out}  ({len(rows)} epoch(s) logged)")


if __name__ == "__main__":
    main()
