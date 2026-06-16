"""Phase 6 publication figure for SFNO: spatial-mean SSR vs lead, fresh vs frozen.

Reads per-variant CSVs from
  $STORE/diagnostics/sfno_phase6_ssr/{frozen,fresh}_<sigma>.csv
and produces a 2x3 multi-panel plot covering the 6 paper-headline variables.
Saves PDF + PNG into figures/phase6_sfno_spatial_mean_ssr.{pdf,png}.

Hypothesis test ([[phase6-fresh-per-step-weight]]): fresh-per-step weight
perturbation produces spatial-mean spread that grows as random-walk sqrt(T)
instead of the persistent-noise T scaling. At T=60 6-h steps the predicted
ratio is ~7.7x at the variance level (~5.4x at spread level).

The matched variance-budget anchor is sigma_persistent x sqrt(T) = 0.25 x
sqrt(60) = 1.93, so fresh@1.93 should produce ~similar pointwise spread to
frozen@0.25 while having dramatically smaller spatial-mean spread.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/diagnostics/sfno_phase6_ssr"
OUT_ROOT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/phase6_sfno_spatial_mean_ssr"

VARS_PLOT = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "specific_humidity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistence-json", type=Path, default=None)
    parser.add_argument("--climatology-json", type=Path, default=None)
    args = parser.parse_args()

    csvs = sorted(glob.glob(f"{BASE}/*.csv"))
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df_lvl = df.groupby(["model", "variable", "lead_time_hours"])["ssr"].mean().reset_index()

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)

    frozen_sigmas = sorted(
        [m for m in df_lvl["model"].unique() if m.startswith("frozen_")],
        key=lambda x: float(x.split("_")[1]),
    )
    fresh_sigmas = sorted(
        [m for m in df_lvl["model"].unique() if m.startswith("fresh_")],
        key=lambda x: float(x.split("_")[1]),
    )
    refresh_models = sorted(
        [m for m in df_lvl["model"].unique() if m.startswith("refresh")],
        key=lambda x: float(x.split("_")[1]),
    )
    colors_frozen = plt.cm.Reds(np.linspace(0.35, 0.95, len(frozen_sigmas)))
    colors_fresh = plt.cm.Blues(np.linspace(0.45, 0.95, len(fresh_sigmas)))
    colors_refresh = plt.cm.Greens(np.linspace(0.5, 0.95, max(len(refresh_models), 1)))

    for ax, var in zip(axs.flat, VARS_PLOT, strict=False):
        sub = df_lvl[df_lvl["variable"] == var]
        for i, m in enumerate(frozen_sigmas):
            d = sub[sub["model"] == m].sort_values("lead_time_hours")
            sig = float(m.split("_")[1])
            ax.plot(
                d["lead_time_hours"],
                d["ssr"],
                "s--",
                color=colors_frozen[i],
                lw=1.6,
                label=f"frozen s={sig:g}",
                alpha=0.9,
                markersize=5,
            )
        for i, m in enumerate(fresh_sigmas):
            d = sub[sub["model"] == m].sort_values("lead_time_hours")
            sig = float(m.split("_")[1])
            ax.plot(
                d["lead_time_hours"],
                d["ssr"],
                "o-",
                color=colors_fresh[i],
                lw=2.0,
                label=f"fresh s={sig:g}",
                alpha=0.9,
                markersize=6,
            )
        for i, m in enumerate(refresh_models):
            d = sub[sub["model"] == m].sort_values("lead_time_hours")
            n = "".join(c for c in m.split("_")[0] if c.isdigit())
            sig = float(m.split("_")[1])
            ax.plot(
                d["lead_time_hours"],
                d["ssr"],
                "^-",
                color=colors_refresh[i],
                lw=2.0,
                label=f"refresh-{n} s={sig:g}",
                alpha=0.9,
                markersize=6,
            )

        if args.climatology_json is not None:
            ax.axhline(1.0, color="black", lw=1.0, ls="-", alpha=0.6, label="Climatology SSR=1")
        else:
            ax.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.6)
        if args.persistence_json is not None:
            ax.axhline(0.0, color="black", lw=1.0, ls=":", alpha=0.7, label="Persistence SSR=0")
        ax.set_title(var, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    axs[0, 0].legend(fontsize=7, loc="best", ncol=2)
    for ax in axs[1]:
        ax.set_xlabel("Lead time (h)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Spatial-mean SSR (log)")

    fig.suptitle(
        "Phase 6 SFNO: spatial-mean SSR -- fresh-per-step vs frozen, modes10",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_ROOT), exist_ok=True)
    for ext in ("pdf", "png"):
        out = f"{OUT_ROOT}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close()

    print()
    print("Variant-mean spatial-mean SSR at lead 240h (variable-mean across 7 paper vars):")
    sub240 = df_lvl[(df_lvl["lead_time_hours"] == 240) & (df_lvl["variable"].isin(VARS_PLOT))]
    summary = sub240.groupby("model")["ssr"].mean().sort_index()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
