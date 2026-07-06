"""Consolidated PIT / verification-rank histogram figure at 240h (reviewer M6).

Uses the SwissClim eval's own per-baseline grid PIT output
  $STORE/baselines/<baseline>/eval/probabilistic/pit_hist_<var>_grid_ensprob.npz
(counts shape n_leads x n_bins), so the figure is consistent with every other
number in the paper. One panel per baseline; counts pooled over the seven paper
variables (3D at 500+850 hPa) at lead 240h and normalised. Flat dashed line =
calibrated; a right/left tilt is a mean bias, a U under-dispersion, a dome
over-dispersion of the pointwise field.

Usage:  python tools/plot_pit_histograms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_colors import color_for  # noqa: E402

STORE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/pit_histograms_240h"
LEAD = 240

ORDER = [
    ("aurora_encoder", "aurora_encoder"),
    ("graphcast_all", "graphcast_all"),
    ("sfno_modes10", "sfno_modes10"),
    ("aifs_perturbed", "aifs_perturbed"),
    ("aifsens", "AIFS-ENS"),
    ("atlas", "Atlas"),
    ("fcn3", "FCN3"),
]
# paper variable-level PIT files
VARFILES = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "geopotential_500",
    "geopotential_850",
    "temperature_500",
    "temperature_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
    "specific_humidity_500",
    "specific_humidity_850",
]


def pooled_pit_240(baseline: str):
    d = Path(f"{STORE}/baselines/{baseline}/eval/probabilistic")
    total = None
    for v in VARFILES:
        f = d / f"pit_hist_{v}_grid_ensprob.npz"
        if not f.is_file():
            continue
        z = np.load(f, allow_pickle=True)
        lh = z["lead_hours"]
        i = int(np.argmin(np.abs(lh - LEAD)))
        c = z["counts"][i].astype(float)
        total = c if total is None else total + c
    if total is None:
        return None
    return total / total.sum()


def main():
    rows = [(k, lab) for k, lab in ORDER if pooled_pit_240(k) is not None]
    if not rows:
        print("no PIT npz found")
        return
    nb = len(pooled_pit_240(rows[0][0]))
    flat = 1.0 / nb

    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, (key, lab) in zip(axs.flat, rows):
        h = pooled_pit_240(key)
        ax.bar(np.arange(nb), h, color=color_for(key), alpha=0.85, width=0.92)
        ax.axhline(flat, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_title(lab, fontsize=13)
        ax.set_ylim(0, max(h.max(), flat) * 1.35)
        ax.set_xticks([0, (nb - 1) // 2, nb - 1])
        ax.set_xticklabels(["0", "", str(nb - 1)])
        ax.tick_params(labelsize=11)
        ax.set_yticks([])
    for ax in axs.flat[len(rows) :]:
        ax.axis("off")
    for ax in axs[1]:
        ax.set_xlabel("verification rank", fontsize=12)
    fig.suptitle(
        "Per-pixel verification-rank (PIT) histograms at 240 h, 7-variable pool", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Wrote {OUT}.{ext}")

    print("\n240h pooled PIT (first/last bin excess over flat):")
    for key, lab in rows:
        h = pooled_pit_240(key)
        print(
            f"  {lab:16} bin0={h[0]:.3f} binL={h[-1]:.3f} flat={flat:.3f} "
            f"peak@{int(np.argmax(h))}"
        )


if __name__ == "__main__":
    main()
