"""Generate Phase 6a publication figure: spatial-mean SSR vs lead, fresh vs frozen, per variable.

Reads per-variant CSVs from
  $STORE/ablation/phase3/graphcast_operational/eval/*/spatial_mean_ssr/spatial_ssr.csv
and produces a 2x3 multi-panel plot covering the 6 paper-headline variables.
Saves PDF + PNG into figures/phase6a_spatial_mean_ssr.{pdf,png}.

Usage:
  python tools/plot_spatial_mean_ssr.py

Hypothesis test ([[phase6-fresh-per-step-weight]]): fresh-per-step weight
perturbation produces spatial-mean spread that grows as sqrt(T) instead of
the persistent-noise T scaling. At T=60 steps the predicted ratio is ~7.7x
at the variance level (~5.4x at spread level). Confirmed direction (frozen
> fresh) with magnitude ~3.7x at lead 240 h on geopotential and others.
"""

from __future__ import annotations

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = (
    "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation/phase3/"
    "graphcast_operational/eval"
)
OUT_ROOT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/phase6a_spatial_mean_ssr"
VARS_PLOT = [
    "2m_temperature",
    "geopotential",
    "temperature",
    "specific_humidity",
    "mean_sea_level_pressure",
    "u_component_of_wind",
]


def main() -> None:
    csvs = sorted(glob.glob(f"{BASE}/*/spatial_mean_ssr/spatial_ssr.csv"))
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    colors_fresh = plt.cm.viridis(np.linspace(0.1, 0.9, 7))

    for ax, var in zip(axs.flat, VARS_PLOT, strict=False):
        sub = df[df["variable"] == var].copy()
        sub_lvl = sub.groupby(["model", "lead_time_hours"])["ssr"].mean().reset_index()

        fresh = sorted(
            [m for m in sub["model"].unique() if "frozen" not in m],
            key=lambda x: float(x.split("_")[1]),
        )
        for i, m in enumerate(fresh):
            d = sub_lvl[sub_lvl["model"] == m].sort_values("lead_time_hours")
            sig = float(m.split("_")[1])
            ax.plot(
                d["lead_time_hours"],
                d["ssr"],
                "o-",
                color=colors_fresh[i],
                lw=1.4,
                label=f"fresh sigma={sig:g}",
                alpha=0.8,
            )

        frozen = sub_lvl[sub_lvl["model"].str.contains("frozen")].sort_values("lead_time_hours")
        ax.plot(
            frozen["lead_time_hours"],
            frozen["ssr"],
            "s--",
            color="crimson",
            lw=2.5,
            label="frozen sigma=1.0",
            markersize=7,
        )

        ax.set_title(var, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)

    axs[0, 0].legend(fontsize=7, loc="upper left", ncol=2)
    for ax in axs[1]:
        ax.set_xlabel("Lead time (h)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Spatial-mean SSR")

    fig.suptitle(
        "Phase 6a: Spatial-mean SSR -- fresh-per-step vs frozen " "(GraphCast n_coarse=42)",
        fontsize=12,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{OUT_ROOT}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close()


if __name__ == "__main__":
    main()
