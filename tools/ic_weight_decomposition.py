"""Attribute ensemble spread to weight perturbation versus initial conditions.

The paper runs weight-only and weight+IC but no IC-only control, so the share of
the short-lead dispersion that the weight perturbation actually contributes is
not separable (reviewer item 1). With the `*_ic_only` arms in place this scores
all three arms of each backbone on the SAME initialisations and reports, per
lead, the cos(lat)-weighted ensemble spread and the fair CRPS.

Only initialisations present in all three arms are used, so every comparison is
paired.

Output CSV: baseline,arm,variable,level,lead,spread,crps,n_inits

Usage:
  python tools/ic_weight_decomposition.py --backbones all --leads 6 24 72 120 240
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

sys.stdout.reconfigure(line_buffering=True)

STORE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
TRUTH_SRC = {
    2023: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    2024: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
}
OUT_CSV = Path("/iopsstor/scratch/cscs/sadamov/ic_weight_decomposition.csv")

# backbone -> {arm: baseline dir name}
TRIPLETS = {
    "aurora": {
        "weight": "aurora_encoder",
        "ic": "aurora_ic_only",
        "weight+ic": "aurora_encoder_ic",
    },
    "graphcast": {
        "weight": "graphcast_all",
        "ic": "graphcast_ic_only",
        "weight+ic": "graphcast_all_ic",
    },
    "sfno": {
        "weight": "sfno_modes10",
        "ic": "sfno_ic_only",
        "weight+ic": "sfno_modes10_ic",
    },
    "aifs": {
        "weight": "aifs_perturbed",
        "ic": "aifs_ic_only",
        "weight+ic": "aifs_perturbed_ic",
    },
}

VARS = [
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
    ("geopotential", 500),
    ("temperature", 850),
    ("u_component_of_wind", 850),
]


_TRUTH_DS: dict[int, xr.Dataset] = {}


def _truth(year: int) -> xr.Dataset:
    """Opened truth stores are reused: this scorer would otherwise reopen the
    multi-TB WB2 zarr once per (arm, variable, init)."""
    if year not in _TRUTH_DS:
        _TRUTH_DS[year] = xr.open_zarr(TRUTH_SRC[year], consolidated=True, decode_timedelta=True)
    return _TRUTH_DS[year]


def fair_crps_pixel(members: np.ndarray, o: np.ndarray) -> np.ndarray:
    M = members.shape[0]
    skill = np.abs(members - o[None]).mean(axis=0)
    s = np.sort(members, axis=0)
    coef = 2.0 * np.arange(1, M + 1) - M - 1.0
    spread = np.tensordot(coef, s, axes=([0], [0])) / (M * (M - 1))
    return skill - spread


def wmean(field: np.ndarray, wlat: np.ndarray) -> float:
    W = np.broadcast_to(wlat[:, None], field.shape)
    m = np.isfinite(field)
    den = np.sum(np.where(m, W, 0.0))
    return float(np.sum(np.where(m, W * field, 0.0)) / den) if den else float("nan")


def score_one(path: Path, var: str, level, lead: int, init: datetime):
    """(spread, crps) for one run/variable/lead, or None if unusable."""
    try:
        ds = xr.open_zarr(path, consolidated=True)
        da = ds[var].sel(lead_time=np.timedelta64(lead, "h"))
    except Exception:
        return None
    if level is not None:
        da = da.sel(level=level)
    if "init_time" in da.dims:
        da = da.isel(init_time=0)
    v = da.values.astype(np.float64)
    if v.ndim != 3 or not np.isfinite(v).all():
        return None
    valid = init + timedelta(hours=lead)
    t = _truth(valid.year)[var]
    if level is not None:
        t = t.sel(level=level)
    o = (
        t.sel(time=np.datetime64(valid, "ns"))
        .sel(latitude=ds["latitude"], longitude=ds["longitude"])
        .values
    )
    wlat = np.cos(np.deg2rad(ds["latitude"].values))
    M = v.shape[0]
    spread = np.sqrt((M + 1.0) / M * v.var(axis=0, ddof=1))
    return wmean(spread, wlat), wmean(fair_crps_pixel(v, o), wlat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["all"])
    ap.add_argument("--leads", nargs="+", type=int, default=[6, 24, 72, 120, 240])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument(
        "--require-inits",
        type=int,
        default=112,
        help="fail unless every backbone has this many paired inits (0 disables)",
    )
    args = ap.parse_args()

    names = sorted(TRIPLETS) if args.backbones == ["all"] else args.backbones
    rows = ["baseline,arm,variable,level,lead,spread,crps,n_inits"]

    for bb in names:
        arms = TRIPLETS[bb]
        dirs = {a: Path(f"{STORE}/baselines/{d}") for a, d in arms.items()}
        common = None
        for d in dirs.values():
            have = {p.parent.name for p in d.glob("*/forecast.zarr")}
            common = have if common is None else (common & have)
        common = sorted(common or [])
        print(f"{bb}: {len(common)} initialisations present in all three arms", flush=True)
        if args.require_inits and len(common) < args.require_inits:
            missing = {
                a: args.require_inits - len({p.parent.name for p in d.glob("*/forecast.zarr")})
                for a, d in dirs.items()
            }
            raise SystemExit(
                f"{bb}: only {len(common)}/{args.require_inits} paired initialisations "
                f"(short by arm: {missing}). Fill the gaps or pass --require-inits 0."
            )
        if not common:
            continue

        for var, level in VARS:
            for lead in args.leads:
                t0 = time.time()
                for arm, d in dirs.items():

                    def one(tag, _d=d, _var=var, _lv=level, _lead=lead):
                        init = datetime.strptime(tag, "%Y%m%d_%H%M")
                        return score_one(_d / tag / "forecast.zarr", _var, _lv, _lead, init)

                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        res = [r for r in ex.map(one, common) if r is not None]
                    if not res:
                        continue
                    sp = float(np.mean([r[0] for r in res]))
                    cr = float(np.mean([r[1] for r in res]))
                    lvl = "" if level is None else level
                    rows.append(f"{bb},{arm},{var},{lvl},{lead},{sp:.8g},{cr:.8g},{len(res)}")
                print(
                    f"  {bb} {var}{'' if level is None else '_' + str(level)} "
                    f"lead {lead}h done ({time.time()-t0:.0f}s)",
                    flush=True,
                )
                Path(args.out).write_text("\n".join(rows) + "\n")

    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"DONE -> {args.out} ({len(rows)-1} rows)", flush=True)


if __name__ == "__main__":
    main()
