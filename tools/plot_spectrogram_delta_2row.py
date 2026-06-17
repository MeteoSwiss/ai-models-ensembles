"""z500 delta-spectrogram, 2-row 8-panel layout.

Row 1: IFS-ENS + the three trained-probabilistic baselines.
Row 2: the four post-hoc weight-perturbed baselines (incl aifs_perturbed).
Replicates the SwissClim energy_spectra delta-spectrogram processing
(eps-relative log ratio, Gaussian smoothing sigma=[0.5,1.0], 4dx wavenumber
cutoff, log y-axis, coolwarm) so the look matches the rest of the paper.
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
import model_colors  # noqa: F401  (sets shared font rcParams on import)

BASE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines"
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/spectrogram_delta_z500_7way"

ROW1 = [("ifs_ens", "IFS-ENS"), ("aifsens", "AIFS-ENS"), ("fcn3", "FCN3"), ("atlas", "Atlas")]
ROW2 = [
    ("aurora_encoder", "aurora_encoder"),
    ("graphcast_all", "graphcast_all"),
    ("sfno_modes10", "sfno_modes10"),
    ("aifs_perturbed", "aifs_perturbed"),
]


def load(b: str):
    f = glob.glob(f"{BASE}/{b}/eval/energy_spectra/*geopotential_500*lead000h-360h*enspooled.npz")[
        0
    ]
    d = np.load(f)
    return d["lead_hours"], d["wavenumber"], d["energy_prediction"], d["energy_target"]


def delta(pred, target):
    t = target[target > 0]
    tmed = float(np.nanmedian(t)) if t.size else 1e-10
    eps = max(tmed * 1e-6, 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.log10(pred + eps) - np.log10(target + eps)
    return gaussian_filter(diff, sigma=[0.5, 1.0])


def main() -> None:
    panels = ROW1 + ROW2
    lead, wn0, _, _ = load(panels[0][0])
    kmax = float(np.nanmax(wn0))
    cm = np.ones(len(wn0), dtype=bool)
    cm[:2] = False
    cm &= wn0 <= kmax / 2.0
    wn = wn0[cm]

    diffs = {}
    for b, _ in panels:
        _, _, pred, target = load(b)
        diffs[b] = delta(pred, target)[:, cm]
    vmax = float(np.nanpercentile(np.abs(np.stack(list(diffs.values()))), 98))

    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True)
    im = None
    for ax, (b, lab) in zip(axs.flat, panels, strict=False):
        im = ax.pcolormesh(
            lead, wn, diffs[b].T, shading="gouraud", cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        ax.set_yscale("log")
        ax.set_title(lab, fontsize=11)
    for ax in axs[1]:
        ax.set_xlabel("Lead time (h)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Wavenumber (cycles/km)")
    fig.suptitle(
        r"Delta spectrogram of $Z_{500}$ ($\Delta\log_{10}$ energy, forecast $-$ truth)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cax = fig.add_axes([0.94, 0.12, 0.013, 0.76])
    fig.colorbar(im, cax=cax, label=r"$\Delta\log_{10}$ energy (forecast $-$ truth)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Wrote {OUT}.{ext}  (vmax={vmax:.3f})")
    plt.close()


if __name__ == "__main__":
    main()
