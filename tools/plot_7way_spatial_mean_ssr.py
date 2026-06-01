"""Tier 1b figure: spatial-mean SSR across the 7-way baseline study.

Reads per-baseline CSVs from
  $STORE/baselines/<baseline>/spatial_mean_ssr/spatial_ssr.csv
covering aurora_encoder, graphcast_all, sfno_modes10, aifsens, fcn3, atlas
(plus aifs_perturbed once the production run lands), and produces a multi-panel
plot of spatial-mean SSR vs lead per variable, split by baseline family
(post-hoc weight perturbation vs trained probabilistic).

Hypothesis: post-hoc weight perturbation produces order-of-magnitude larger
spatial-mean spread than trained probabilistic models, even when pointwise SSR
is comparable. See [[spatial-mean-vs-pointwise-ssr]].
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines"
OUT_ROOT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/tier1b_7way_spatial_mean_ssr"

PERTURBED = {
    "aurora_encoder": "Aurora encoder s=0.025",
    "graphcast_all": "GraphCast all s=0.01",
    "sfno_modes10": "SFNO modes10 s=0.25",
}
TRAINED = {
    "aifsens": "AIFS-ENS",
    "fcn3": "FCN3",
    "atlas": "Atlas",
}

VARS_PLOT = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]


def load_all() -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(f"{BASE}/*/spatial_mean_ssr/spatial_ssr.csv")):
        baseline = d.split("/")[-3]
        df = pd.read_csv(d)
        df["baseline"] = baseline
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = load_all()
    df_lvl = df.groupby(["baseline", "variable", "lead_time_hours"])["ssr"].mean().reset_index()

    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    colors_pert = plt.cm.Blues(np.linspace(0.45, 0.95, len(PERTURBED)))
    colors_trnd = plt.cm.Oranges(np.linspace(0.45, 0.95, len(TRAINED)))

    for ax, var in zip(axs.flat[: len(VARS_PLOT)], VARS_PLOT, strict=False):
        sub = df_lvl[df_lvl["variable"] == var]
        for i, (b, lab) in enumerate(PERTURBED.items()):
            d = sub[sub["baseline"] == b].sort_values("lead_time_hours")
            if len(d):
                ax.plot(
                    d["lead_time_hours"],
                    d["ssr"],
                    "o-",
                    color=colors_pert[i],
                    lw=1.6,
                    label=lab,
                    alpha=0.9,
                )
        for i, (b, lab) in enumerate(TRAINED.items()):
            d = sub[sub["baseline"] == b].sort_values("lead_time_hours")
            if len(d):
                ax.plot(
                    d["lead_time_hours"],
                    d["ssr"],
                    "s--",
                    color=colors_trnd[i],
                    lw=1.6,
                    label=lab,
                    alpha=0.9,
                )

        ax.axhline(1.0, color="black", lw=0.7, ls=":", alpha=0.6)
        ax.set_title(var, fontsize=10)
        ax.grid(alpha=0.3)

    # 8th panel becomes the legend
    for ax in axs.flat[len(VARS_PLOT) :]:
        ax.axis("off")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs.flat[len(VARS_PLOT)].legend(handles, labels, fontsize=10, loc="center", frameon=False)
    for ax in axs[1]:
        ax.set_xlabel("Lead time (h)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Spatial-mean SSR")

    fig.suptitle(
        "Tier 1b: Spatial-mean SSR -- post-hoc weight perturbation vs trained probabilistic",
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
    print("Per-baseline mean spatial-mean SSR at lead 240h:")
    summary = (
        df_lvl[df_lvl["lead_time_hours"] == 240]
        .groupby("baseline")["ssr"]
        .agg(["mean", "min", "max", "count"])
    )
    print(summary.round(3))


if __name__ == "__main__":
    main()
