"""Raw ensemble member values at single gridpoints, all 112 production inits.

Companion to compute_ens_anomalies.py (which samples 2048 points x 28 inits): here
single locations, every init, so the member distribution at each point can be shown
in physical units. Several points share one pass (every chunk read is a global field). Also stores the ERA5 probabilistic climatology there (1990-2019,
years as members, matched by calendar month/day/hour, as in the headline CRPSS) and
IFS-ENS (all 50 perturbed members, and members 0, 5, ..., 45; missing WB2 slabs are NaN).

Output, per point: $AIENS_STORE/analysis/ens_anom_gaussianity/point_<name>.npz
  <baseline> (nvarlev, nlead, 112, M) float32 native units, era5_clim (nvarlev, nlead,
  112, 30), ifs_ens (..., 50), ifs_ens10 (..., 10), labels, lead_hours, inits, lat, lon,
  clim_years.

Default points: the four of figures/ens_anom/point_grid_values_* (north to south) and their
opposite meridians (point_grid_opposite_values_*), see tools/plot_point_ensemble.py.

Usage (CPU sbatch): python tools/compute_point_ensemble.py [--points name,lat,lon ...]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from _env import BASELINES, IFS_ENS, WB2_ORIGINAL
from compute_ens_anomalies import (
    ALL_BASELINES,
    CLIM_NAME,
    CLIM_YEARS,
    IFS10_MEMBERS,
    IFS10_NAME,
    IFS_NAME,
    OUTDIR,
    VARS,
)

sys.stdout.reconfigure(line_buffering=True)

POINTS = [
    "ny_alesund,79,12",
    "zurich,47.5,8.5",
    "equator_pacific,0,180",
    "mcmurdo,-77.85,166.67",
    "arctic_168w,79,192",
    "npacific_171w,47.5,188.5",
    "equator_0e,0,0",
    "antarctic_13w,-77.75,346.75",
]


def nearest(ds: xr.Dataset, lats, lons) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = ds.latitude.values, ds.longitude.values
    return (
        np.array([np.abs(lat - la).argmin() for la in lats]),
        np.array([np.abs(lon - lo % 360).argmin() for lo in lons]),
    )


def process_init(path: str, lats, lons, leads: list[int]) -> np.ndarray:
    """(npoint, nvarlev, nlead, M)."""
    ds = xr.open_zarr(path, chunks=None)
    ilat, ilon = nearest(ds, lats, lons)
    lead_h = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int)
    out = np.empty((len(lats), len(VARS), len(leads), ds.sizes["ensemble"]), np.float32)
    for k, (var, lev) in enumerate(VARS):
        da = ds[var].isel(init_time=0)
        if lev is not None:
            da = da.sel(level=lev)
        for j, h in enumerate(leads):
            x = da.isel(lead_time=int(np.flatnonzero(lead_h == h)[0]))
            out[:, k, j] = x.transpose("ensemble", "latitude", "longitude").values[:, ilat, ilon].T
    return out


def process_ifs_init(init: str, lats, lons, leads: list[int]) -> np.ndarray:
    """(npoint, nvarlev, nlead, 50) from the IFS-ENS zarr; missing slabs are NaN."""
    ds = xr.open_zarr(IFS_ENS, chunks=None).sel(init_time=datetime.strptime(init, "%Y%m%d_%H%M"))
    ilat, ilon = nearest(ds, lats, lons)
    out = np.empty((len(lats), len(VARS), len(leads), ds.sizes["ensemble"]), np.float32)
    for k, (var, lev) in enumerate(VARS):
        da = ds[var] if lev is None else ds[var].sel(level=lev)
        for j, h in enumerate(leads):
            x = da.sel(lead_time=np.timedelta64(h, "h"))
            out[:, k, j] = x.transpose("ensemble", "latitude", "longitude").values[:, ilat, ilon].T
    return out


def process_clim_time(month: int, day: int, hour: int, lats, lons) -> np.ndarray:
    """(npoint, nvarlev, nyears) at one calendar day and hour."""
    ds = xr.open_zarr(WB2_ORIGINAL, consolidated=True, chunks=None)
    ilat, ilon = nearest(ds, lats, lons)
    out = np.empty((len(lats), len(VARS), len(CLIM_YEARS)), np.float32)
    for y, year in enumerate(CLIM_YEARS):
        t = np.datetime64(f"{year}-{month:02d}-{day:02d}T{hour:02d}")
        for var in dict.fromkeys(v for v, _ in VARS):
            rows = [k for k, (v, _) in enumerate(VARS) if v == var]
            levs = [VARS[k][1] for k in rows]
            da = ds[var].sel(time=t)
            if levs[0] is None:
                out[:, rows[0], y] = da.values[ilat, ilon]
            else:
                vals = da.sel(level=levs).values  # one read of the 37-level chunk
                for r, k in enumerate(rows):
                    out[:, k, y] = vals[r][ilat, ilon]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", default=POINTS, help="name,lat,lon")
    ap.add_argument("--leads", type=int, nargs="+", default=[240])
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()
    names = [p.split(",")[0] for p in args.points]
    lats = [float(p.split(",")[1]) for p in args.points]
    lons = [float(p.split(",")[2]) for p in args.points]

    root0 = BASELINES / ALL_BASELINES[0]
    inits = sorted(p.name for p in root0.iterdir() if (p / "forecast.zarr").is_dir())
    if len(inits) != 112:
        raise RuntimeError(f"expected 112 inits, found {len(inits)}")

    data: dict[str, np.ndarray] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name in ALL_BASELINES:
            futs = {
                ex.submit(
                    process_init,
                    str(BASELINES / name / it / "forecast.zarr"),
                    lats,
                    lons,
                    args.leads,
                ): j
                for j, it in enumerate(inits)
            }
            res = {futs[f]: f.result() for f in as_completed(futs)}
            data[name] = np.stack([res[j] for j in range(len(inits))], axis=3)
            print(f"  {name}: done ({time.time() - t0:.0f}s)")

        futs = {
            ex.submit(process_ifs_init, it, lats, lons, args.leads): j for j, it in enumerate(inits)
        }
        res = {futs[f]: f.result() for f in as_completed(futs)}
        data[IFS_NAME] = np.stack([res[j] for j in range(len(inits))], axis=3)
        data[IFS10_NAME] = data[IFS_NAME][..., IFS10_MEMBERS]
        print(f"  {IFS_NAME}: done ({time.time() - t0:.0f}s)")

        valid = {}
        for j, it in enumerate(inits):
            for li, h in enumerate(args.leads):
                t = datetime.strptime(it, "%Y%m%d_%H%M") + timedelta(hours=h)
                valid[(j, li)] = (t.month, t.day, t.hour)
        times = sorted(set(valid.values()))
        futs = {ex.submit(process_clim_time, *mdh, lats, lons): mdh for mdh in times}
        res = {futs[f]: f.result() for f in as_completed(futs)}
        clim = np.empty(
            (len(names), len(VARS), len(args.leads), len(inits), len(CLIM_YEARS)), np.float32
        )
        for (j, li), mdh in valid.items():
            clim[:, :, li, j] = res[mdh]
        data[CLIM_NAME] = clim
        print(f"  {CLIM_NAME}: {len(times)} calendar times ({time.time() - t0:.0f}s)")

    ds0 = xr.open_zarr(BASELINES / ALL_BASELINES[1] / inits[0] / "forecast.zarr", chunks=None)
    ilat, ilon = nearest(ds0, lats, lons)
    for i, pname in enumerate(names):
        out = OUTDIR / f"point_{pname}.npz"
        np.savez(
            out,
            labels=np.array([f"{v}" if lev is None else f"{v}_{lev}" for v, lev in VARS]),
            lead_hours=np.array(args.leads),
            inits=np.array(inits),
            lat=float(ds0.latitude.values[ilat[i]]),
            lon=float(ds0.longitude.values[ilon[i]]),
            clim_years=np.array(CLIM_YEARS),
            **{k: v[i] for k, v in data.items()},
        )
        print(f"wrote {out}: " + ", ".join(f"{k} {v[i].shape}" for k, v in data.items()))


if __name__ == "__main__":
    main()
