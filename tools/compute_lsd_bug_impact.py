"""Exact impact of the member-0 align bug on the paper's LSD column.

SwissClim's energy_spectra collapses the prediction ensemble to member 0
(_compute_spectra_pair -> xr.align(join="inner") against a target that carries a
singleton ensemble dim). The calibration table's LSD column reads
lsd_metrics_3d_lead_time_combined.csv, whose values are LSD computed per
(init, lead, level) and then averaged over init and level.

This recomputes both variants on the production baselines - member 0 as stored,
and the mean of per-member spectra as intended - so the difference is exact
rather than estimated. Reduction order matches the eval exactly.

Usage (compute node):
    python tools/compute_lsd_bug_impact.py --out tools/data/lsd_bug_impact.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
from _env import IFS_ENS, STORE, WB2_2022, WB2_2024  # noqa: E402

from swissclim_evaluations.plots.energy_spectra import (  # noqa: E402
    _compute_lsd_da,
    calculate_energy_spectra,
)

VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS = [500, 850]
LEADS = [120, 240]
ML_BASELINES = [
    "aifsens",
    "atlas",
    "fcn3",
    "aurora_encoder",
    "graphcast_all",
    "sfno_modes10",
]
IFS_MEMBERS = list(range(0, 50, 5))  # the stratified 10 the eval config selects


def truth_for(year: int) -> xr.Dataset:
    return xr.open_zarr(WB2_2022 if year < 2024 else WB2_2024)


def _open(model: str, init: str) -> xr.Dataset:
    """Prediction for one init, subset to the eval's leads and members."""
    leads = [np.timedelta64(h, "h") for h in LEADS]
    if model == "ifs_ens":
        ds = xr.open_zarr(IFS_ENS).sel(init_time=np.datetime64(init))
        ds = ds.isel(ensemble=IFS_MEMBERS)
    else:
        tag = init.replace("-", "").replace("T", "_").replace(":", "")[:13]
        ds = xr.open_zarr(STORE / "baselines" / model / tag / "forecast.zarr")
        if "init_time" in ds.dims:
            ds = ds.squeeze("init_time", drop=True)
    return ds.sel(lead_time=leads)


def one_init(job: tuple[str, str]) -> list[dict]:
    model, init = job
    fc = _open(model, init)
    valid = np.datetime64(init) + fc["lead_time"].values
    tr = (
        truth_for(int(init[:4]))
        .sel(time=valid)
        .sel(latitude=fc.latitude, longitude=fc.longitude, method="nearest")
        .rename({"time": "lead_time"})
        .assign_coords(lead_time=fc["lead_time"])
    )
    rows = []
    for var in VARS_3D:
        for lvl in LEVELS:
            da_p = fc[var].sel(level=lvl, drop=True).load()
            da_t = tr[var].sel(level=lvl, drop=True).load()
            sp = calculate_energy_spectra(da_p)  # keeps ensemble
            st = calculate_energy_spectra(da_t)
            s0 = sp.isel(ensemble=0, drop=True)  # what the eval stored
            spool = sp.mean(dim="ensemble")  # what it should have been
            lsd0 = _compute_lsd_da(st, s0)
            lsdp = _compute_lsd_da(st, spool)
            for i, h in enumerate(LEADS):
                rows.append(
                    {
                        "model": model,
                        "variable": var,
                        "level": lvl,
                        "lead": h,
                        "init": init,
                        "lsd_member0": float(np.asarray(lsd0.values).ravel()[i]),
                        "lsd_pooled": float(np.asarray(lsdp.values).ravel()[i]),
                    }
                )
    return rows


def inits_for(model: str) -> list[str]:
    if model == "ifs_ens":
        d = xr.open_zarr(IFS_ENS)
        return [str(t)[:16] for t in d.init_time.values]
    dirs = sorted(p.name for p in (STORE / "baselines" / model).iterdir() if p.name[:2] == "20")
    return [f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[9:11]}:{d[11:13]}" for d in dirs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ML_BASELINES + ["ifs_ens"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-inits", type=int, default=None)
    ap.add_argument("--out", default="tools/data/lsd_bug_impact.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_rows: list[dict] = []
    for model in args.models:
        inits = inits_for(model)
        if args.limit_inits:
            inits = inits[: args.limit_inits]
        print(f"[{model}] {len(inits)} inits", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, rows in enumerate(ex.map(one_init, [(model, i) for i in inits]), 1):
                all_rows.extend(rows)
                if n % 20 == 0:
                    print(f"  {model} {n}/{len(inits)}", flush=True)
        pd.DataFrame(all_rows).to_csv(args.out, index=False)
        print(f"  {model} done -> {args.out} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
