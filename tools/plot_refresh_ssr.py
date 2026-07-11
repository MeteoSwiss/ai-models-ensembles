"""Frozen vs refresh-every-20 spatial-mean SSR.

Shows the refresh fix on the over-dispersed weight-space backbones (Aurora,
AIFS, both pulled DOWN toward 1) alongside the mildly under-dispersed SFNO
(pulled UP toward 1) - the opposite direction, which is the interesting point.
7-variable mean per lead from the cached spatial_mean_ssr CSVs. Replaces the
SFNO-only Phase-6 sweep figure as the headline refresh figure (Fig. fig:phase6).

All three frozen+refresh pairs are now on the 112-init production grid
($STORE/baselines/<b>/spatial_mean_ssr/spatial_ssr.csv). SFNO refresh is
sfno_p6c (refresh-every-20, sigma_N = 0.25*sqrt(2) ~ 0.35), run on the
production grid to match Aurora/AIFS (tools/submit_sfno_p6c_production.sh).
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
from model_colors import color_for

STORE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
BASE = f"{STORE}/baselines"
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/refresh_frozen_vs_refresh_ssr"

# (frozen CSV path, refresh CSV path, label, colour-key)
# All three on the 112-init production grid.
PAIRS = [
    (
        f"{BASE}/aurora_encoder/spatial_mean_ssr/spatial_ssr.csv",
        f"{BASE}/aurora_p6c_reseed/spatial_mean_ssr/spatial_ssr.csv",
        "Aurora",
        "aurora_encoder",
    ),
    (
        f"{BASE}/aifs_perturbed/spatial_mean_ssr/spatial_ssr.csv",
        f"{BASE}/aifs_p6c_reseed/spatial_mean_ssr/spatial_ssr.csv",
        "AIFS",
        "aifs_perturbed",
    ),
    (
        f"{BASE}/sfno_modes10/spatial_mean_ssr/spatial_ssr.csv",
        f"{BASE}/sfno_p6c_reseed/spatial_mean_ssr/spatial_ssr.csv",
        "SFNO",
        "sfno_modes10",
    ),
]


def var_mean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.groupby("lead_time_hours")["ssr"].mean().reset_index()


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhline(1.0, color="0.5", lw=1.2, ls="-", zorder=0)
    color_handles = []
    for frozen, refresh, label, ckey in PAIRS:
        c = color_for(ckey)
        d = var_mean(frozen).sort_values("lead_time_hours")
        ax.plot(
            d["lead_time_hours"],
            d["ssr"],
            color=c,
            lw=2.0,
            ls="-",
            marker="o",
            ms=5,
        )
        r = var_mean(refresh).sort_values("lead_time_hours")
        ax.plot(
            r["lead_time_hours"],
            r["ssr"],
            color=c,
            lw=2.0,
            ls="--",
            marker="s",
            ms=5,
        )
        color_handles.append(Line2D([], [], color=c, lw=2.5, label=label))

    style_handles = [
        Line2D([], [], color="0.2", lw=2.0, ls="-", marker="o", ms=5, label="frozen"),
        Line2D([], [], color="0.2", lw=2.0, ls="--", marker="s", ms=5, label="refresh-20"),
    ]

    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Spatial-mean SSR (7-variable mean)")
    ax.grid(alpha=0.3)

    # Two compact keys outside the data area: model colour (top right of the
    # margin) and frozen/refresh line style (below it). Keeps the axes clear.
    leg_model = ax.legend(
        handles=color_handles,
        title="Model",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(leg_model)
    ax.legend(
        handles=style_handles,
        title="Weight noise",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Wrote {OUT}.{ext}")
    plt.close()
    print("\n240h values:")
    for frozen, refresh, label, _ in PAIRS:
        f240 = var_mean(frozen).query("lead_time_hours==240")["ssr"].iloc[0]
        r240 = var_mean(refresh).query("lead_time_hours==240")["ssr"].iloc[0]
        print(f"  {label}: frozen {f240:.2f} -> refresh {r240:.2f}")


if __name__ == "__main__":
    main()
