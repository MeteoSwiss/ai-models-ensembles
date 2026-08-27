"""Spread-error diagram: does a large forecast spread predict a large error?

Reads the weighted spread histogram from tools/spread_error_binned.py, collapses
it into equal-weight spread deciles, and plots the binned ensemble-mean RMSE
against the binned spread. A calibrated, state-dependent ensemble follows the
1:1 line; a spread that carries no case-to-case information gives a flat curve.

Usage:
  python tools/plot_spread_error.py --csv <binned.csv> --out figures/spread_error.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_colors import color_for, style_for  # noqa: E402

CSV = "/iopsstor/scratch/cscs/sadamov/spread_error_binned.csv"
PANEL_VARS = [
    ("2m_temperature", "", "2 m temperature [K]"),
    ("geopotential", "500", "z500 [m$^2$ s$^{-2}$]"),
]
ORDER = [
    "aurora_encoder",
    "graphcast_all",
    "sfno_modes10",
    "aifs_perturbed",
    "aifsens",
    "atlas",
    "fcn3",
    "ifs_ens",
]


def deciles(edges_lo, edges_hi, w, ws, we, n_bins=10):
    """Collapse the fine histogram into n_bins equal-weight spread bins."""
    order = np.argsort(edges_lo)
    w, ws, we = w[order], ws[order], we[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.array([]), np.array([])
    targets = np.linspace(0, cw[-1], n_bins + 1)[1:-1]
    cuts = np.searchsorted(cw, targets)
    groups = np.split(np.arange(len(w)), cuts)
    s, r = [], []
    for g in groups:
        if len(g) == 0 or w[g].sum() <= 0:
            continue
        s.append(ws[g].sum() / w[g].sum())
        r.append(np.sqrt(we[g].sum() / w[g].sum()))
    return np.array(s), np.array(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--leads", nargs="+", type=int, default=[24, 120, 240])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default="figures/spread_error.pdf")
    args = ap.parse_args()

    data = defaultdict(lambda: defaultdict(list))
    with open(args.csv) as f:
        header = f.readline().rstrip("\n").split(",")
        ix = {k: i for i, k in enumerate(header)}
        for line in f:
            p = line.rstrip("\n").split(",")
            key = (p[ix["baseline"]], p[ix["variable"]], p[ix["level"]], int(p[ix["lead"]]))
            for col in ("bin_lo", "bin_hi", "w_count", "w_sum_spread", "w_sum_sq_err"):
                data[key][col].append(float(p[ix[col]]))

    nrow, ncol = len(args.leads), len(PANEL_VARS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow), squeeze=False)
    for i, lead in enumerate(args.leads):
        for j, (var, lvl, xlabel) in enumerate(PANEL_VARS):
            ax = axes[i][j]
            lim = [np.inf, -np.inf]
            for name in ORDER:
                d = data.get((name, var, lvl, lead))
                if not d:
                    continue
                s, r = deciles(
                    np.array(d["bin_lo"]),
                    np.array(d["bin_hi"]),
                    np.array(d["w_count"]),
                    np.array(d["w_sum_spread"]),
                    np.array(d["w_sum_sq_err"]),
                    args.bins,
                )
                if s.size == 0:
                    continue
                ax.plot(
                    s,
                    r,
                    style_for(name),
                    color=color_for(name),
                    marker="o",
                    ms=3,
                    lw=1.4,
                    label=name if (i == 0 and j == 0) else None,
                )
                lim[0] = min(lim[0], s.min(), r.min())
                lim[1] = max(lim[1], s.max(), r.max())
            if np.isfinite(lim[0]):
                ax.plot(lim, lim, color="0.4", lw=0.8, ls=(0, (4, 3)), zorder=0)
            ax.set_xlabel(f"ensemble spread, {xlabel}")
            ax.set_ylabel("ensemble-mean RMSE")
            name = xlabel.split(" [")[0]
            ax.set_title(
                f"{name}{'' if not lvl else ' ' + lvl + ' hPa'}, lead {lead} h", fontsize=10
            )
            ax.grid(alpha=0.25, lw=0.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
