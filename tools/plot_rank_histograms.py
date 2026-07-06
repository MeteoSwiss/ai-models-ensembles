"""Rank-histogram figure at 240h (reviewer M6/M5).

Consumes tools/compute_rank_histograms.py output (rank_hist_<baseline>.npz) and
draws, one row per baseline, the per-pixel and spatial-mean rank (Talagrand)
histograms pooled over the seven paper variables. Flat = calibrated; U =
under-dispersed; dome = over-dispersed. Expectation: the frozen post-hoc
spatial mean is domed for the over-dispersed backbones (graphcast_all,
aifs_perturbed, aurora_encoder) and flatter for sfno_modes10 and the trained
baselines, confirming the near-constant whole-field offset of
sec. results-spatial.

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

ORDER = [
    ("aurora_encoder", "Aurora encoder"),
    ("graphcast_all", "GraphCast all"),
    ("sfno_modes10", "SFNO modes10"),
    ("aifs_perturbed", "AIFS decoder"),
    ("aifsens", "AIFS-ENS"),
    ("atlas", "Atlas"),
    ("fcn3", "FCN3"),
]


def main():
    rows = [(k, lab) for k, lab in ORDER if (INDIR / f"rank_hist_{k}.npz").is_file()]
    if not rows:
        print("no rank_hist_*.npz found yet (job pending)")
        return
    M = int(np.load(INDIR / f"rank_hist_{rows[0][0]}.npz")["M"])
    nb = M + 1
    flat = 1.0 / nb

    fig, axs = plt.subplots(
        len(rows), 2, figsize=(6.8, 0.82 * len(rows)), sharex=True, squeeze=False
    )
    for r, (key, lab) in enumerate(rows):
        d = np.load(INDIR / f"rank_hist_{key}.npz")
        pp = d["perpixel_counts"].sum(axis=0).astype(float)
        pp = pp / pp.sum()
        sm_raw = d["spatialmean_ranks"]
        sm_vals = sm_raw[sm_raw >= 0]
        sm = np.bincount(sm_vals, minlength=nb)[:nb].astype(float)
        sm = sm / sm.sum() if sm.sum() else sm
        col = color_for(key)
        for c, (h, title) in enumerate([(pp, "per-pixel"), (sm, "spatial-mean")]):
            ax = axs[r, c]
            ax.bar(np.arange(nb), h, color=col, alpha=0.85, width=0.9)
            ax.axhline(flat, color="black", lw=0.8, ls="--", alpha=0.6)
            ax.set_ylim(0, max(h.max(), flat) * 1.35)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=12)
            if c == 0:
                ax.set_ylabel(lab, fontsize=10, rotation=0, ha="right", va="center")
    for c in range(2):
        axs[-1, c].set_xlabel("truth rank among members", fontsize=10)
        axs[-1, c].set_xticks([0, M // 2, M])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
