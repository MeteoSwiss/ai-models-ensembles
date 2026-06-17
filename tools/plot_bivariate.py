"""Combined bivariate (joint-distribution) figures for appendix C.

Reads the per-baseline cached multivariate npz (forecast hist + truth hist)
and lays out one panel per baseline (4 post-hoc, then 3 trained + IFS-ENS).
Includes aifs_perturbed (reviewer request). Two figures:
  bivariate_Tq_500hPa_7way.pdf      - T vs q with Clausius-Clapeyron envelope
  bivariate_geostrophic_500hPa.pdf  - |grad Z| vs wind speed (geostrophic)
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
import model_colors  # noqa: F401  (shared font)

BASE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines"
FIGDIR = "/users/sadamov/pyprojects/ai-models-ensembles/figures"

PANELS = [
    ("aurora_encoder", "aurora_encoder"),
    ("graphcast_all", "graphcast_all"),
    ("sfno_modes10", "sfno_modes10"),
    ("aifs_perturbed", "aifs_perturbed"),
    ("aifsens", "AIFS-ENS"),
    ("fcn3", "FCN3"),
    ("atlas", "Atlas"),
    ("ifs_ens", "IFS-ENS"),
]


def find(b: str, pattern: str) -> str:
    g = glob.glob(f"{BASE}/{b}/eval/multivariate/*{pattern}*level500*.npz")
    return g[0] if g else ""


def qsat_bolton(T_K, p_hpa=500.0):
    Tc = T_K - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))  # hPa (Bolton 1980)
    return 0.622 * es / (p_hpa - 0.378 * es)


def panel(ax, npz, cc_curve=False):
    d = np.load(npz, allow_pickle=True)
    hx, hy = d["bins_x"], d["bins_y"]
    xc = 0.5 * (hx[:-1] + hx[1:])
    yc = 0.5 * (hy[:-1] + hy[1:])
    X, Y = np.meshgrid(xc, yc)
    H = d["hist"].T.astype(float)
    HT = d["hist_target"].T.astype(float)
    Hl = np.log10(H + 1.0)
    ax.contourf(X, Y, Hl, levels=12, cmap="viridis")
    ax.contour(X, Y, np.log10(HT + 1.0), levels=6, colors="0.7", linewidths=0.5)
    if cc_curve:
        Tg = np.linspace(xc.min(), xc.max(), 200)
        qs = qsat_bolton(Tg)
        ax.fill_between(Tg, qs, yc.max(), color="pink", alpha=0.35, lw=0)
        ax.axhspan(yc.min(), 0.0, color="pink", alpha=0.35, lw=0)
        ax.plot(Tg, qs, "r--", lw=1.2)
        ax.set_ylim(yc.min(), yc.max())
        ax.set_xlim(xc.min(), xc.max())
    return d["var_x"], d["var_y"]


def make(pattern, out, xlabel, ylabel, suptitle, cc=False):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for ax, (b, lab) in zip(axs.flat, PANELS, strict=False):
        f = find(b, pattern)
        if not f:
            ax.set_visible(False)
            continue
        panel(ax, f, cc_curve=cc)
        ax.set_title(lab, fontsize=11)
    for ax in axs[1]:
        ax.set_xlabel(xlabel)
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/{out}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Wrote {FIGDIR}/{out}.{ext}")
    plt.close()


def main() -> None:
    make(
        "temperature_specific_humidity",
        "bivariate_Tq_500hPa_7way",
        "Temperature (K)",
        "Specific humidity (kg/kg)",
        "Joint $T$-$q$ distribution at 500 hPa (forecast filled, truth grey)",
        cc=True,
    )
    make(
        "geopotential_height_gradient_wind_speed",
        "bivariate_geostrophic_500hPa",
        "Geopotential-height gradient",
        "Wind speed (m/s)",
        "Geostrophic balance at 500 hPa (forecast filled, truth grey)",
        cc=False,
    )


if __name__ == "__main__":
    main()
