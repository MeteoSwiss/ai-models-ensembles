"""Block-bootstrap 95% CIs on the headline CRPSS (reviewer M1).

Consumes per_init_crps_production.csv (tools/compute_per_init_crps.py) and the
cached WB2 climatology denominator (tools/data/crps_clim_eval_1990_2019.json).

1. Validation: variable-mean CRPSS per baseline per lead (3D vars averaged over
   500/850 first, then mean over the 7 paper variables) must reproduce
   figures/headline_8way_table.tex; printed side by side. Only a clean match
   licenses the CI.
2. Block bootstrap over the 8 weekly init-blocks (2 years x Jan/Apr/Jul/Oct x
   days 2-8, 14 inits each), resampled with replacement B times. The
   climatology denominator is held at its all-init value (its sampling
   variability is second-order and shared). Reports each baseline's CRPSS with
   a 95% percentile CI and the gap to AIFS-ENS with a 95% CI (paired: the same
   resampled blocks for both baselines), and the bootstrap probability that the
   gap has the observed sign.

Usage:
  python tools/block_bootstrap_crpss.py --lead 240 --B 10000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

CSV = "/iopsstor/scratch/cscs/sadamov/per_init_crps_production.csv"
CLIM = Path(__file__).resolve().parent / "data" / "crps_clim_eval_1990_2019.json"

PAPER_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS_3D = (500, 850)
# headline_8way_table.tex variable-mean CRPSS for cross-check
TABLE = {
    120: {
        "aifsens": 0.580,
        "atlas": 0.558,
        "fcn3": 0.534,
        "aifs_perturbed": 0.542,
        "graphcast_all": 0.489,
        "aurora_encoder": 0.471,
        "sfno_modes10": 0.388,
        "ifs_ens": 0.518,
    },
    240: {
        "aifsens": 0.256,
        "atlas": 0.238,
        "fcn3": 0.218,
        "aifs_perturbed": 0.221,
        "graphcast_all": 0.167,
        "aurora_encoder": 0.134,
        "sfno_modes10": 0.125,
        "ifs_ens": 0.207,
    },
}


def block_of(init_iso: str) -> tuple[int, int]:
    dt = datetime.fromisoformat(init_iso)
    return (dt.year, dt.month)  # 8 (year, month) weekly blocks


def clim_crps(clim: dict, var: str, level, lead: int) -> float:
    key = var if level is None else f"{var}_{level}"
    return float(clim[key][str(lead)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=240)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=CSV)
    args = ap.parse_args()

    clim = json.loads(CLIM.read_text())
    rng = np.random.default_rng(args.seed)

    # rows: baseline -> block -> init -> {(var,level): crps}
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for ln in Path(args.csv).read_text().splitlines()[1:]:
        b, init, var, lvl, lead, crps, nm = ln.split(",")
        if int(lead) != args.lead:
            continue
        level = None if lvl == "" else int(lvl)
        data[b][block_of(init)][init][(var, level)] = float(crps)

    baselines = [
        x
        for x in [
            "aifsens",
            "atlas",
            "fcn3",
            "aifs_perturbed",
            "graphcast_all",
            "aurora_encoder",
            "sfno_modes10",
            "ifs_ens",
        ]
        if x in data
    ]

    def var_mean_crpss(per_init_crps: dict[str, dict]) -> float:
        """Given {init: {(var,level): crps}}, return variable-mean CRPSS over inits."""
        vscores = []
        for var in PAPER_VARS:
            levels = (
                [None] if var in ("2m_temperature", "mean_sea_level_pressure") else list(LEVELS_3D)
            )
            lvl_crpss = []
            for level in levels:
                num = np.nanmean(
                    [
                        per_init_crps[i][(var, level)]
                        for i in per_init_crps
                        if (var, level) in per_init_crps[i]
                    ]
                )
                den = clim_crps(clim, var, level, args.lead)
                lvl_crpss.append(1.0 - num / den)
            vscores.append(np.mean(lvl_crpss))
        return float(np.mean(vscores))

    # point estimates
    all_inits = {b: {i: v for blk in data[b].values() for i, v in blk.items()} for b in baselines}
    point = {b: var_mean_crpss(all_inits[b]) for b in baselines}

    print(f"=== validation vs headline table (lead {args.lead}h) ===")
    print(f"{'baseline':16} {'recomputed':>10} {'table':>8} {'d':>7}")
    for b in baselines:
        t = TABLE.get(args.lead, {}).get(b, float("nan"))
        print(f"{b:16} {point[b]:10.3f} {t:8.3f} {point[b]-t:+7.3f}")

    # block bootstrap
    blocks = sorted({blk for b in baselines for blk in data[b]})
    nb = len(blocks)
    boot = {b: np.empty(args.B) for b in baselines}
    for r in range(args.B):
        pick = rng.integers(0, nb, size=nb)
        chosen = [blocks[k] for k in pick]
        for b in baselines:
            merged = {}
            for j, blk in enumerate(chosen):
                for i, v in data[b].get(blk, {}).items():
                    merged[f"{i}#{j}"] = v  # unique key so repeated blocks count twice
            boot[b][r] = var_mean_crpss(merged)

    def ci(x):
        return np.percentile(x, 2.5), np.percentile(x, 97.5)

    print(f"\n=== CRPSS with 95% block-bootstrap CI (B={args.B}, {nb} weekly blocks) ===")
    print(f"{'baseline':16} {'CRPSS':>7} {'95% CI':>18}")
    for b in baselines:
        lo, hi = ci(boot[b])
        print(f"{b:16} {point[b]:7.3f}   [{lo:6.3f}, {hi:6.3f}]")

    ref = "aifsens"
    if ref in boot:
        print("\n=== gap to AIFS-ENS (paired) : AIFS-ENS - baseline, 95% CI ===")
        print(f"{'baseline':16} {'gap':>7} {'95% CI':>18} {'P(gap>0)':>9}")
        for b in baselines:
            if b == ref:
                continue
            d = boot[ref] - boot[b]
            lo, hi = ci(d)
            p = float(np.mean(d > 0))
            print(f"{b:16} {point[ref]-point[b]:7.3f}   [{lo:6.3f}, {hi:6.3f}] {p:9.3f}")


if __name__ == "__main__":
    main()
