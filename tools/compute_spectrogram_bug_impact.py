"""Exact impact of the member-0 align bug on the paper's delta spectrogram.

figures/spectrogram_delta_z500_7way.pdf is built by plot_spectrogram_delta_2row.py
from the eval's *enspooled.npz bundles, whose energy_prediction is the
init-averaged spectrum of member 0 alone (see compute_lsd_bug_impact.py).

This recomputes the same quantity as the init-averaged mean of per-member spectra
and reports the difference in the plotted delta field. Unlike the LSD column, the
init average here happens before the nonlinearity, so the two should be much
closer; this measures by how much.

Runs on a lead subset (every Nth of the 61 six-hourly steps) to fit the debug
partition.

Usage (compute node):
    python tools/compute_spectrogram_bug_impact.py --lead-stride 4
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
from _env import IFS_ENS, STORE, WB2_2022, WB2_2024  # noqa: E402

from swissclim_evaluations.plots.energy_spectra import (  # noqa: E402
    calculate_energy_spectra,
)

PANELS = [
    "ifs_ens",
    "aifsens",
    "fcn3",
    "atlas",
    "aurora_encoder",
    "graphcast_all",
    "sfno_modes10",
    "aifs_perturbed",
]
IFS_MEMBERS = list(range(0, 50, 5))
_STRIDE = 4


def truth_for(year: int) -> xr.Dataset:
    return xr.open_zarr(WB2_2022 if year < 2024 else WB2_2024)


def _open(model: str, init: str) -> xr.Dataset:
    if model == "ifs_ens":
        ds = xr.open_zarr(IFS_ENS).sel(init_time=np.datetime64(init))
        ds = ds.isel(ensemble=IFS_MEMBERS)
    else:
        tag = init.replace("-", "").replace("T", "_").replace(":", "")[:13]
        ds = xr.open_zarr(STORE / "baselines" / model / tag / "forecast.zarr")
        if "init_time" in ds.dims:
            ds = ds.squeeze("init_time", drop=True)
    return ds.isel(lead_time=slice(None, None, _STRIDE))


def one_init(job: tuple[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (member0 spectrum, pooled spectrum, target spectrum, lead_hours)."""
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
    da_p = fc["geopotential"].sel(level=500, drop=True).load()
    da_t = tr["geopotential"].sel(level=500, drop=True).load()
    sp = calculate_energy_spectra(da_p)
    st = calculate_energy_spectra(da_t)
    hours = (fc["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    return (
        sp.isel(ensemble=0, drop=True).values,
        sp.mean(dim="ensemble").values,
        st.values,
        hours,
    )


def inits_for(model: str) -> list[str]:
    if model == "ifs_ens":
        return [str(t)[:16] for t in xr.open_zarr(IFS_ENS).init_time.values]
    dirs = sorted(p.name for p in (STORE / "baselines" / model).iterdir() if p.name[:2] == "20")
    return [f"{d[:4]}-{d[4:6]}-{d[6:8]}T{d[9:11]}:{d[11:13]}" for d in dirs]


def main() -> None:
    global _STRIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=PANELS)
    ap.add_argument("--lead-stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-inits", type=int, default=None)
    ap.add_argument("--out", default="tools/data/spectrogram_bug_impact.npz")
    args = ap.parse_args()
    _STRIDE = args.lead_stride

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out: dict[str, np.ndarray] = {}
    for model in args.models:
        inits = inits_for(model)
        if args.limit_inits:
            inits = inits[: args.limit_inits]
        print(f"[{model}] {len(inits)} inits", flush=True)
        acc0, accp, acct, hours = [], [], [], None
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, (s0, sp, st, h) in enumerate(ex.map(one_init, [(model, i) for i in inits]), 1):
                acc0.append(s0)
                accp.append(sp)
                acct.append(st)
                hours = h
                if n % 20 == 0:
                    print(f"  {model} {n}/{len(inits)}", flush=True)
        out[f"{model}|member0"] = np.nanmean(np.stack(acc0), axis=0)
        out[f"{model}|pooled"] = np.nanmean(np.stack(accp), axis=0)
        out[f"{model}|target"] = np.nanmean(np.stack(acct), axis=0)
        out["lead_hours"] = hours
        np.savez_compressed(args.out, **out)
        print(f"  {model} done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
