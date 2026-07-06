"""Rank (Talagrand) histograms at 240h for reviewer M5.

The spatial-mean spread-skill pathology (sec. results-spatial) predicts a
specific rank-histogram shape: the over-dispersed post-hoc spatial mean should
give a domed histogram (truth too often central), the mildly under-dispersed
per-pixel field a shallow U, and the trained baselines a flatter histogram.
This computes both from the production forecast.zarr member fields.

For each baseline and (variable, level) at lead 240h:
  * per-pixel rank = #(members < truth) over the M members, pooled over all
    inits and pixels -> M+1 bins.
  * spatial-mean rank = rank of the cos-lat truth spatial mean among the M
    member spatial means, one per init.

Output: /iopsstor/scratch/cscs/sadamov/rank_hist_<baseline>.npz
  perpixel_counts (nvarlev, M+1), spatialmean_ranks (nvarlev, ninits), labels.

Usage (CPU sbatch): python tools/compute_rank_histograms.py --baselines all
"""

from __future__ import annotations

import argparse
import sys
import time
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
OUTDIR = Path("/iopsstor/scratch/cscs/sadamov")

BASELINES = {
    "aurora_encoder": f"{STORE}/baselines/aurora_encoder",
    "graphcast_all": f"{STORE}/baselines/graphcast_all",
    "sfno_modes10": f"{STORE}/baselines/sfno_modes10",
    "aifs_perturbed": f"{STORE}/baselines/aifs_perturbed",
    "aifsens": f"{STORE}/baselines/aifsens",
    "atlas": f"{STORE}/baselines/atlas",
    "fcn3": f"{STORE}/baselines/fcn3",
    "ifs_ens": "/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr",
}
# IFS-ENS is a single WeatherBench-2 consolidated zarr (init_time dim, 50 members)
# rather than the per-init forecast.zarr tree, and carries archive NaN gaps in the
# upper-air fields. Subsample its 50 members to a fixed 10 so its Talagrand
# histogram shares the M+1=11 bin grid of the 10-member baselines (the pooled
# per-pixel shape is insensitive to which 10).
IFS_ENS_MEMBERS = 10
IFS_ENS_SEED = 0
VARS = [
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
    ("geopotential", 500),
    ("geopotential", 850),
    ("temperature", 500),
    ("temperature", 850),
    ("u_component_of_wind", 500),
    ("u_component_of_wind", 850),
    ("v_component_of_wind", 500),
    ("v_component_of_wind", 850),
    ("specific_humidity", 500),
    ("specific_humidity", 850),
]
INIT_DATES = [
    datetime(y, m, d, h)
    for y in (2023, 2024)
    for m in (1, 4, 7, 10)
    for d in range(2, 9)
    for h in (0, 12)
]


def cos_lat_weights(lat):
    return np.cos(np.deg2rad(lat))


def wmean(field, wlat):
    W = np.broadcast_to(wlat[:, None], field.shape)
    m = np.isfinite(field)
    den = np.sum(np.where(m, W, 0.0))
    return float(np.sum(np.where(m, W * field, 0.0)) / den) if den else np.nan


def truth_field(var, level, valid, cache):
    k = (var, level, valid.isoformat())
    if k not in cache:
        ds = xr.open_zarr(TRUTH_SRC[valid.year], consolidated=True, decode_timedelta=True)
        da = ds[var]
        if level is not None:
            da = da.sel(level=level)
        da = da.sel(time=np.datetime64(valid, "ns"))
        cache[k] = (da.values, da["latitude"].values, da["longitude"].values)
    return cache[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["all"])
    ap.add_argument("--lead", type=int, default=240)
    ap.add_argument("--limit-inits", type=int, default=None)
    args = ap.parse_args()

    names = list(BASELINES) if args.baselines == ["all"] else args.baselines
    inits = INIT_DATES[: args.limit_inits] if args.limit_inits else INIT_DATES
    cache: dict = {}

    for name in names:
        is_ifs = name == "ifs_ens"
        base = None if is_ifs else Path(BASELINES[name])
        ifs_ds = ifs_sel = None
        if is_ifs:
            ifs_ds = xr.open_zarr(BASELINES[name], consolidated=True, chunks={})
            rng = np.random.default_rng(IFS_ENS_SEED)
            ifs_sel = np.sort(rng.choice(ifs_ds.sizes["ensemble"], IFS_ENS_MEMBERS, replace=False))
        t0 = time.time()
        M_ref = 10
        pp = np.zeros((len(VARS), M_ref + 1), dtype=np.int64)
        sm = [[] for _ in VARS]
        for init in inits:
            if is_ifs:
                try:
                    sub = ifs_ds.sel(
                        init_time=np.datetime64(init, "ns"),
                        lead_time=np.timedelta64(args.lead, "h"),
                    ).isel(ensemble=ifs_sel)
                except Exception:
                    continue
                flat, flon = ifs_ds["latitude"].values, ifs_ds["longitude"].values
            else:
                zp = base / f"{init:%Y%m%d_%H%M}" / "forecast.zarr"
                if not zp.is_dir():
                    continue
                fc = xr.open_zarr(zp, consolidated=True, chunks={})
                flat, flon = fc["latitude"].values, fc["longitude"].values
                try:
                    sub = fc.sel(lead_time=np.timedelta64(args.lead, "h"))
                except Exception:
                    continue
                if "init_time" in sub.dims:
                    sub = sub.isel(init_time=0)
            wlat = cos_lat_weights(flat)
            valid = init + timedelta(hours=args.lead)
            for vi, (var, level) in enumerate(VARS):
                da = sub[var]
                if level is not None:
                    da = da.sel(level=level)
                if is_ifs:
                    da = da.transpose("ensemble", "latitude", "longitude")
                members = da.values  # (M, lat, lon)
                if members.ndim != 3:
                    continue
                M = members.shape[0]
                o_full, tlat, tlon = truth_field(var, level, valid, cache)
                if o_full.shape != members.shape[1:]:
                    o = (
                        xr.DataArray(
                            o_full,
                            coords={"latitude": tlat, "longitude": tlon},
                            dims=["latitude", "longitude"],
                        )
                        .sel(latitude=flat, longitude=flon)
                        .values
                    )
                else:
                    o = o_full
                # per-pixel rank = #(members < obs), 0..M
                fin = np.isfinite(members).all(axis=0) & np.isfinite(o)
                rk = (members < o[None]).sum(axis=0)
                counts = np.bincount(rk[fin].ravel(), minlength=M + 1)
                if M == M_ref:
                    pp[vi] += counts[: M_ref + 1]
                # spatial-mean rank
                mem_sm = np.array([wmean(members[m], wlat) for m in range(M)])
                o_sm = wmean(o, wlat)
                if np.isfinite(o_sm) and np.isfinite(mem_sm).all():
                    sm[vi].append(int((mem_sm < o_sm).sum()))
        labels = [v if lv is None else f"{v}_{lv}" for v, lv in VARS]
        maxlen = max((len(s) for s in sm), default=0)
        sm_arr = np.full((len(VARS), maxlen), -1, dtype=np.int64)
        for i, s in enumerate(sm):
            sm_arr[i, : len(s)] = s
        outp = OUTDIR / f"rank_hist_{name}.npz"
        np.savez(
            outp, perpixel_counts=pp, spatialmean_ranks=sm_arr, labels=np.array(labels), M=M_ref
        )
        print(f"{name}: {time.time()-t0:.0f}s -> {outp}", flush=True)


if __name__ == "__main__":
    main()
