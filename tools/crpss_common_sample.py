"""CRPSS on the common (IFS-ENS-valid) subsample of initialisations.

The WeatherBench-2 IFS-ENS archive is gappy: whole (variable, init, lead) map
slabs are missing, so the IFS-ENS CRPSS numerator averages a smaller sample than
the gap-free ML rows, while the climatology denominator averages all 112 inits.
This pairs both sides on exactly the same cases and reports, per lead:

  * CRPSS over the full sample each baseline actually has (the paper's numbers)
  * CRPSS over the common subsample where IFS-ENS is valid, numerator AND
    denominator restricted identically
  * the ranking under both, so a rank change is visible at a glance

Inputs:
  --per-init  per-init CRPS numerators (tools/compute_per_init_crps.py)
  --clim      per-init climatology denominators
              (tools/data/crps_clim_eval_1990_2019_per_init.json)

Usage:
  python tools/crpss_common_sample.py --leads 240 --out-csv <path>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _env import SCRATCH

PER_INIT = str(SCRATCH / "per_init_crps_production.csv")
CLIM = Path(__file__).resolve().parent / "data" / "crps_clim_eval_1990_2019_per_init.json"

VARS_3D = {
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
}


def label_of(var: str, level: str) -> str:
    return var if level in ("", None) else f"{var}_{int(float(level))}"


def load_per_init(path: str):
    """(baseline, var, level, lead) -> {init: crps}, finite entries only."""
    rows: dict = defaultdict(dict)
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
        ix = {k: i for i, k in enumerate(header)}
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < len(header):
                continue
            try:
                crps = float(p[ix["crps"]])
            except ValueError:
                continue
            if not np.isfinite(crps):
                continue
            if int(float(p[ix["n_members"]])) < 2:
                continue
            key = (p[ix["baseline"]], p[ix["variable"]], p[ix["level"]], int(p[ix["lead"]]))
            rows[key][p[ix["init"]]] = crps
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-init", default=PER_INIT)
    ap.add_argument("--clim", default=str(CLIM))
    ap.add_argument("--leads", nargs="+", type=int, default=[240])
    ap.add_argument("--reference", default="ifs_ens")
    ap.add_argument("--out-csv", default=str(SCRATCH / "crpss_common_sample.csv"))
    args = ap.parse_args()

    per_init = load_per_init(args.per_init)
    clim = json.loads(Path(args.clim).read_text())

    baselines = sorted({k[0] for k in per_init})
    channels = sorted({(k[1], k[2]) for k in per_init})

    out = ["lead,baseline,sample,n_inits_min,crpss"]
    for lead in args.leads:
        # common sample per channel = inits where the reference is valid
        common: dict[tuple[str, str], set] = {}
        for var, lvl in channels:
            ref = per_init.get((args.reference, var, lvl, lead), {})
            common[(var, lvl)] = set(ref)

        summary = {}
        for name in baselines:
            for tag in ("full", "common"):
                per_var: dict[str, list] = defaultdict(list)
                n_min = 10**6
                for var, lvl in channels:
                    num = per_init.get((name, var, lvl, lead), {})
                    if not num:
                        continue
                    inits = set(num) & common[(var, lvl)] if tag == "common" else set(num)
                    den_all = clim.get(label_of(var, lvl), {}).get(str(lead), {})
                    inits = {i for i in inits if i in den_all}
                    if len(inits) < 2:
                        continue
                    n_min = min(n_min, len(inits))
                    num_mean = float(np.mean([num[i] for i in inits]))
                    den_mean = float(np.mean([den_all[i] for i in inits]))
                    if den_mean <= 0:
                        continue
                    per_var[var].append(1.0 - num_mean / den_mean)
                if not per_var:
                    continue
                # levels averaged within a variable first, then variables
                score = float(np.mean([np.mean(v) for v in per_var.values()]))
                summary[(name, tag)] = (score, n_min)
                out.append(f"{lead},{name},{tag},{n_min},{score:.6f}")

        print(f"\n=== lead {lead} h")
        print(f"{'baseline':22s} {'full':>9s} {'common':>9s} {'delta':>8s}  {'n_min':>6s}")
        order_full = sorted(
            (b for b in baselines if (b, "full") in summary),
            key=lambda b: -summary[(b, "full")][0],
        )
        order_common = sorted(
            (b for b in baselines if (b, "common") in summary),
            key=lambda b: -summary[(b, "common")][0],
        )
        for b in order_full:
            f = summary[(b, "full")][0]
            c, n = summary.get((b, "common"), (float("nan"), 0))
            print(f"{b:22s} {f:9.4f} {c:9.4f} {c-f:8.4f}  {n:6d}")
        print(f"rank full  : {' > '.join(order_full)}")
        print(f"rank common: {' > '.join(order_common)}")
        print("RANKING UNCHANGED" if order_full == order_common else "RANKING CHANGES")

    Path(args.out_csv).write_text("\n".join(out) + "\n")
    print(f"\n-> {args.out_csv}")


if __name__ == "__main__":
    main()
