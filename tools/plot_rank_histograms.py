"""Per-pixel rank-histogram figure at 240h (appendix C4).

Consumes tools/compute_rank_histograms.py output (rank_hist_<baseline>.npz) and
draws the per-pixel verification-rank (Talagrand) histogram for each baseline,
pooled over the seven paper variables (3D at 500+850 hPa) on the 112-init
production grid. Flat (dashed) = calibrated; U = under-, dome = over-dispersed.

Left column: the four post-hoc weight-perturbation ensembles (this work); right
column: the trained-probabilistic baselines and the IFS-ENS classical reference
(50 members subsampled to 10). Every panel shares the same y-limits. The
domain-mean (spatial-mean) calibration story is carried by the spatial-mean SSR
(Fig. tier1b), not by a rank histogram of a single aggregated scalar.

Usage:  python tools/plot_rank_histograms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_colors import color_for  # noqa: E402

INDIR = Path("/iopsstor/scratch/cscs/sadamov")
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/rank_histograms_240h"

# Column-grouped: post-hoc perturbation (this work) left, trained-probabilistic
# baselines + the IFS-ENS classical reference right.
LEFT = [
    ("aurora_encoder", "Aurora encoder"),
    ("graphcast_all", "GraphCast all"),
    ("sfno_modes10", "SFNO modes10"),
    ("aifs_perturbed", "AIFS decoder"),
]
RIGHT = [
    ("aifsens", "AIFS-ENS"),
    ("atlas", "Atlas"),
    ("fcn3", "FCN3"),
    ("ifs_ens", "IFS-ENS"),
]


def perpixel(key):
    f = INDIR / f"rank_hist_{key}.npz"
    if not f.is_file():
        return None, None
    d = np.load(f)
    h = d["perpixel_counts"].sum(axis=0).astype(float)
    s = h.sum()
    return (h / s if s else h), int(d["M"])


def main():
    cols = [LEFT, RIGHT]
    nrow = max(len(LEFT), len(RIGHT))
    hs, M = {}, None
    for col in cols:
        for key, _ in col:
            h, m = perpixel(key)
            if h is not None:
                hs[key], M = h, m
    if not hs:
        print("no rank_hist_*.npz found yet (job pending)")
        return
    nb = M + 1
    flat = 1.0 / nb
    ymax = max(max(h.max() for h in hs.values()), flat) * 1.12

    fig, axs = plt.subplots(nrow, 2, figsize=(6.6, 1.2 * nrow + 0.5), sharex=True, sharey=True)
    for c, col in enumerate(cols):
        for r, (key, lab) in enumerate(col):
            ax = axs[r, c]
            h = hs.get(key)
            if h is None:
                ax.axis("off")
                continue
            ax.bar(np.arange(nb), h, color=color_for(key), alpha=0.85, width=0.9)
            ax.axhline(flat, color="black", lw=0.8, ls="--", alpha=0.6)
            ax.set_ylim(0, ymax)
            ax.set_yticks([])
            ax.set_xlim(-0.6, M + 0.6)
            ax.set_xticks([0, M // 2, M])
            ax.set_xlabel(lab, fontsize=11)
    fig.supxlabel("rank of truth among the 10 members", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
