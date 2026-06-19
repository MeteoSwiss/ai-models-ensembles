"""Combined bivariate (joint-distribution) figures for appendix C.

Reads the per-baseline cached multivariate npz (forecast hist + truth hist)
and lays out one panel per baseline (4 post-hoc, then 3 trained + IFS-ENS).
Includes aifs_perturbed (reviewer request). Two figures:
  bivariate_Tq_500hPa_7way.pdf      - T vs q with Clausius-Clapeyron envelope
  bivariate_geostrophic_500hPa.pdf  - |grad Z| vs wind speed (geostrophic)

Styling is delegated to the SwissClim research-branch renderer
``plot_bivariate_histogram`` (plasma filled contours, truncated-Greys truth
contour lines, physical-constraint overlays), with a single shared horizontal
colorbar built here exactly like the intercomparison ``multivariate()`` driver
(log locator + ``cbar.add_lines`` truth-density level marks). The per-baseline
name is kept only as a small corner annotation, not a matplotlib title bar, and
there is no figure suptitle (AMS: no redundant titles in the graphic).
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
import model_colors  # noqa: F401  (shared font)

from swissclim_evaluations.plots.bivariate_histograms import plot_bivariate_histogram

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


def make(pattern, out, xlabel, ylabel):
    # Collect the available baselines and their cached histograms.
    entries = []
    for b, lab in PANELS:
        f = find(b, pattern)
        if not f:
            continue
        d = np.load(f, allow_pickle=True)
        entries.append(
            {
                "label": lab,
                "hist": np.asarray(d["hist"]),
                "hist_target": np.asarray(d["hist_target"]),
                "bins_x": np.asarray(d["bins_x"]),
                "bins_y": np.asarray(d["bins_y"]),
                "var_x": str(d["var_x"]),
                "var_y": str(d["var_y"]),
                "level_hpa": float(d["level_hpa"]) if d["level_hpa"].size else None,
                "coriolis_parameter": float(d["coriolis_parameter"]),
            }
        )
    if not entries:
        print(f"No cached histograms found for {pattern}; skipped")
        return

    # Shared axis limits across panels (research-branch convention: zoom out 25%).
    all_x_min = min(e["bins_x"].min() for e in entries)
    all_x_max = max(e["bins_x"].max() for e in entries)
    all_y_min = min(e["bins_y"].min() for e in entries)
    all_y_max = max(e["bins_y"].max() for e in entries)
    x_range, y_range = all_x_max - all_x_min, all_y_max - all_y_min
    x_center, y_center = (all_x_max + all_x_min) / 2.0, (all_y_max + all_y_min) / 2.0
    shared_xlim = (x_center - 0.625 * x_range, x_center + 0.625 * x_range)
    shared_ylim = (y_center - 0.625 * y_range, y_center + 0.625 * y_range)

    # Global log-norm across all histograms so the shared colorbar is consistent.
    all_vals = []
    for e in entries:
        for key in ("hist", "hist_target"):
            pos = e[key][e[key] > 0]
            if pos.size:
                all_vals.append(pos)
    combined = np.concatenate(all_vals) if all_vals else np.array([1e-10, 1.0])
    global_vmin = max(float(combined.min()), 1e-10)
    global_vmax = float(combined.max())
    global_norm = LogNorm(vmin=global_vmin, vmax=global_vmax)

    # Shared truth histogram (first model) so the truth contour lines are
    # byte-identical across panels, matching the intercomparison driver.
    ref_target = entries[0]["hist_target"]

    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    axs_flat = axs.flatten()
    shared_target_cs = None
    for idx, e in enumerate(entries):
        ax = axs_flat[idx]
        is_bottom = idx >= 4
        result = plot_bivariate_histogram(
            hist_1=e["hist"],
            hist_2=ref_target,
            bins_x=e["bins_x"],
            bins_y=e["bins_y"],
            label_1="Prediction",
            label_2="Target",
            var_x=e["var_x"],
            var_y=e["var_y"],
            level_hpa=e["level_hpa"],
            ax=ax,
            xlabel=xlabel if is_bottom else None,
            ylabel=ylabel,
            xlim=shared_xlim,
            ylim=shared_ylim,
            show_colorbar=False,
            show_legend=(idx == 0),
            coriolis_parameter=e["coriolis_parameter"],
            return_contour_sets=True,
        )
        # Drop the per-panel matplotlib title bar; keep the baseline name as a
        # small corner annotation instead (AMS: no redundant titles). Pin it to
        # the top-right corner: the renderer's per-panel legend (drawn on idx==0)
        # sits upper-left for T-q and lower-right for the geostrophic pair, so
        # top-right is clear of the legend in both figures.
        ax.set_title("")
        ax.text(
            0.97,
            0.97,
            e["label"],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )
        if not is_bottom:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if idx % 4 != 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        if shared_target_cs is None and isinstance(result, tuple):
            shared_target_cs = result[2]

    # Hide surplus axes.
    for idx in range(len(entries), len(axs_flat)):
        axs_flat[idx].set_visible(False)

    # Single shared horizontal colorbar at the bottom, with truth-density level
    # marks via cbar.add_lines (research-branch multivariate() styling).
    n = len(entries)
    sm = ScalarMappable(cmap="plasma", norm=global_norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axs_flat[:n].tolist(),
        orientation="horizontal",
        location="bottom",
        pad=0.08,
        fraction=0.06,
        shrink=0.6,
    )
    cbar.ax.xaxis.set_major_locator(mticker.LogLocator())
    cbar.ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    cbar.set_label("Density (log scale)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    if shared_target_cs is not None:
        cbar.add_lines(shared_target_cs)

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
    )
    make(
        "geopotential_height_gradient_wind_speed",
        "bivariate_geostrophic_500hPa",
        "Geopotential-height gradient",
        "Wind speed (m/s)",
    )


if __name__ == "__main__":
    main()
