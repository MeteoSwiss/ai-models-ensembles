"""Frozen vs refresh-every-20 spatial-mean SSR on the production grid.

Shows the refresh fix on the over-dispersed weight-space backbones (Aurora,
AIFS) pulled toward 1, with SFNO (already near 1) frozen for reference.
7-variable mean per lead from the cached spatial_mean_ssr CSVs. Replaces the
SFNO-only Phase-6 sweep figure as the headline refresh figure
(Fig. fig:phase6). Reads:
  $STORE/baselines/<b>/spatial_mean_ssr/spatial_ssr.csv
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
from model_colors import color_for

BASE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines"
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/refresh_frozen_vs_refresh_ssr"

# (frozen baseline, refresh run, label, colour-key)
PAIRS = [
    ("aurora_encoder", "aurora_p6c", "Aurora", "aurora_encoder"),
    ("aifs_perturbed", "aifs_p6c", "AIFS", "aifs_perturbed"),
    ("sfno_modes10", None, "SFNO", "sfno_modes10"),
]


def var_mean(baseline: str) -> pd.DataFrame:
    df = pd.read_csv(f"{BASE}/{baseline}/spatial_mean_ssr/spatial_ssr.csv")
    return df.groupby("lead_time_hours")["ssr"].mean().reset_index()


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhline(1.0, color="0.5", lw=1.2, ls="-", zorder=0)
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
            ms=4,
            label=f"{label} (frozen)",
        )
        if refresh is not None:
            r = var_mean(refresh).sort_values("lead_time_hours")
            ax.plot(
                r["lead_time_hours"],
                r["ssr"],
                color=c,
                lw=2.0,
                ls="--",
                marker="s",
                ms=4,
                label=f"{label} (refresh-20)",
            )
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Spatial-mean SSR (7-variable mean)")
    ax.set_title("Refresh-every-20 corrects the over-dispersed spatial mean")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Wrote {OUT}.{ext}")
    plt.close()
    print("\n240h values:")
    for frozen, refresh, label, _ in PAIRS:
        f240 = var_mean(frozen).query("lead_time_hours==240")["ssr"].iloc[0]
        msg = f"  {label}: frozen {f240:.2f}"
        if refresh:
            r240 = var_mean(refresh).query("lead_time_hours==240")["ssr"].iloc[0]
            msg += f" -> refresh {r240:.2f}"
        print(msg)


if __name__ == "__main__":
    main()
