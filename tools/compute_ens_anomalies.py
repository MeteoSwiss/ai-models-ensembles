"""Per-gridpoint standardized ensemble anomalies for the marginal-Gaussianity check.

Does Gaussian weight noise give Gaussian ensemble marginals? The pooled field
distribution (the standard wd_kde output) cannot answer that: it is dominated
by climatology. The relevant sample is the ensemble's conditional distribution
at each gridpoint, made comparable across points by standardizing:

    z_m = (x_m - mean_m x) / std_m(x, ddof=1)

For each baseline, (variable, level), lead and init this reads the full
M-member slab and keeps z at a fixed set of cos-lat-weighted random gridpoints.
Lead 0 is skipped (weight-only ensembles have zero spread at t=0).

Reference "era5_clim": the WB2 probabilistic climatology used for the headline
CRPSS (ERA5 1990-2019, years as members, matched by calendar month/day/hour, no
smoothing) at the same valid times. Its 30 years are split into three disjoint
10-member ensembles (year index mod 3) so the standardization uses the same M=10
as the forecasts; the three groups are stacked on the init axis.

Output: $AIENS_STORE/analysis/ens_anom_gaussianity/<baseline>.npz
  z (nvarlev, nlead, ninit, M, npoint) float32, mean / std (nvarlev, nlead, ninit, npoint)
  float32 ensemble mean and ddof=1 std in native units (x_m = mean + z * std), labels,
  lead_hours, inits, lat, lon.
  For era5_clim ninit = 3 x 28 (init-major, year group minor).

Usage (CPU sbatch): python tools/compute_ens_anomalies.py --baselines all era5_clim
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from _env import BASELINES, STORE, WB2_ORIGINAL

sys.stdout.reconfigure(line_buffering=True)

OUTDIR = STORE / "analysis" / "ens_anom_gaussianity"
ALL_BASELINES = [
    "aurora_encoder",
    "graphcast_all",
    "sfno_modes10",
    "aifs_perturbed",
    "aifsens",
    "atlas",
    "fcn3",
]
VARS = [
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
    ("10m_u_component_of_wind", None),
    ("10m_v_component_of_wind", None),
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
LEAD_HOURS = list(range(24, 361, 24))
INIT_STRIDE = 4  # every 4th of the 112 production inits -> 28, all 8 season blocks
N_POINTS = 2048
POINT_SEED = 0
CLIM_NAME = "era5_clim"
CLIM_YEARS = list(range(1990, 2020))
CLIM_GROUPS = 3


def sample_points(lat: np.ndarray, nlon: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(POINT_SEED)
    w = np.repeat(np.cos(np.deg2rad(lat)).clip(min=0.0), nlon)
    flat = rng.choice(w.size, size=N_POINTS, replace=False, p=w / w.sum())
    return np.unravel_index(np.sort(flat), (lat.size, nlon))


def standardize(x: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z, mean, std along the member axis (mean/std drop that axis)."""
    x = x.astype(np.float64)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, ddof=1, keepdims=True)
    sd[sd == 0] = np.nan
    z = (x - mu) / sd
    return (
        z.astype(np.float32),
        mu.squeeze(axis).astype(np.float32),
        sd.squeeze(axis).astype(np.float32),
    )


def production_inits() -> list[str]:
    root = BASELINES / ALL_BASELINES[0]
    inits = sorted(p.name for p in root.iterdir() if (p / "forecast.zarr").is_dir())
    if len(inits) != 112:
        raise RuntimeError(f"expected 112 inits, found {len(inits)}")
    return inits[::INIT_STRIDE]


def save(name, z, mu, sd, inits: list[str], lat: np.ndarray, lon: np.ndarray) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{name}.npz"
    np.savez(
        out,
        z=z,
        mean=mu,
        std=sd,
        labels=np.array([f"{v}" if lev is None else f"{v}_{lev}" for v, lev in VARS]),
        lead_hours=np.array(LEAD_HOURS),
        inits=np.array(inits),
        lat=lat,
        lon=lon,
    )
    nan_frac = float(np.isnan(z).mean())
    print(f"{name}: wrote {out} shape={z.shape} nan_frac={nan_frac:.2e}")


def process_init(path: str, ilat: np.ndarray, ilon: np.ndarray) -> tuple[np.ndarray, ...]:
    ds = xr.open_zarr(path, chunks=None)
    lead_h = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int)
    li = [int(np.flatnonzero(lead_h == h)[0]) for h in LEAD_HOURS]
    M = ds.sizes["ensemble"]
    out = np.full((len(VARS), len(LEAD_HOURS), M, N_POINTS), np.nan, dtype=np.float32)
    mu = np.full((len(VARS), len(LEAD_HOURS), N_POINTS), np.nan, dtype=np.float32)
    sd = np.full_like(mu, np.nan)
    for k, (var, lev) in enumerate(VARS):
        da = ds[var].isel(init_time=0)
        if lev is not None:
            da = da.sel(level=lev)
        for j, lt in enumerate(li):
            x = da.isel(lead_time=lt).transpose("ensemble", "latitude", "longitude").values
            out[k, j], mu[k, j], sd[k, j] = standardize(x[:, ilat, ilon], axis=0)
    return out, mu, sd


def process_clim_day(
    month: int, day: int, ilat: np.ndarray, ilon: np.ndarray
) -> tuple[np.ndarray, ...]:
    """z (nvarlev, CLIM_GROUPS, M, npoint), mean/std (nvarlev, CLIM_GROUPS, npoint), one day 00 UTC."""
    ds = xr.open_zarr(WB2_ORIGINAL, consolidated=True, chunks=None)
    x = np.empty((len(VARS), len(CLIM_YEARS), N_POINTS), np.float32)
    for y, year in enumerate(CLIM_YEARS):
        t = np.datetime64(f"{year}-{month:02d}-{day:02d}T00")
        for var in dict.fromkeys(v for v, _ in VARS):
            rows = [k for k, (v, _) in enumerate(VARS) if v == var]
            levs = [VARS[k][1] for k in rows]
            da = ds[var].sel(time=t)
            if levs[0] is None:
                x[rows[0], y] = da.values[ilat, ilon]
            else:
                vals = da.sel(level=levs).values  # one read of the 37-level chunk
                for r, k in enumerate(rows):
                    x[k, y] = vals[r][ilat, ilon]
    groups = np.stack([x[:, g::CLIM_GROUPS] for g in range(CLIM_GROUPS)], axis=1)
    return standardize(groups, axis=2)


def run_climatology(workers: int) -> None:
    inits = production_inits()
    valid = {}
    for j, it in enumerate(inits):
        t0 = datetime.strptime(it, "%Y%m%d_%H%M")
        for li, h in enumerate(LEAD_HOURS):
            t = t0 + timedelta(hours=h)
            if t.hour != 0:
                raise RuntimeError(f"valid time {t} is not 00 UTC")
            valid[(j, li)] = (t.month, t.day)
    days = sorted(set(valid.values()))
    ds0 = xr.open_zarr(WB2_ORIGINAL, consolidated=True)
    lat, lon = ds0.latitude.values, ds0.longitude.values
    ilat, ilon = sample_points(lat, lon.size)

    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_clim_day, m, d, ilat, ilon): (m, d) for m, d in days}
        for n, fut in enumerate(as_completed(futs), 1):
            res[futs[fut]] = fut.result()
            print(f"  {CLIM_NAME}: {n}/{len(days)} calendar days ({time.time() - t0:.0f}s)")

    M = len(CLIM_YEARS) // CLIM_GROUPS
    z = np.empty((len(VARS), len(LEAD_HOURS), len(inits) * CLIM_GROUPS, M, N_POINTS), np.float32)
    mu = np.empty((len(VARS), len(LEAD_HOURS), len(inits) * CLIM_GROUPS, N_POINTS), np.float32)
    sd = np.empty_like(mu)
    for (j, li), md in valid.items():
        g = slice(j * CLIM_GROUPS, (j + 1) * CLIM_GROUPS)
        z[:, li, g], mu[:, li, g], sd[:, li, g] = res[md]
    save(CLIM_NAME, z, mu, sd, inits, lat[ilat], lon[ilon])


def run_baseline(name: str, workers: int) -> None:
    root = BASELINES / name
    inits = production_inits()
    ds0 = xr.open_zarr(root / inits[0] / "forecast.zarr", chunks=None)
    lat = ds0.latitude.values
    lon = ds0.longitude.values
    M = ds0.sizes["ensemble"]
    ilat, ilon = sample_points(lat, lon.size)

    z = np.full((len(VARS), len(LEAD_HOURS), len(inits), M, N_POINTS), np.nan, np.float32)
    mu = np.full((len(VARS), len(LEAD_HOURS), len(inits), N_POINTS), np.nan, np.float32)
    sd = np.full_like(mu, np.nan)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(process_init, str(root / it / "forecast.zarr"), ilat, ilon): j
            for j, it in enumerate(inits)
        }
        for n, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            z[:, :, j], mu[:, :, j], sd[:, :, j] = fut.result()
            print(f"  {name}: {n}/{len(inits)} inits ({time.time() - t0:.0f}s)")
    save(name, z, mu, sd, inits, lat[ilat], lon[ilon])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["all"])
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()
    names = ALL_BASELINES if args.baselines == ["all"] else args.baselines
    for name in names:
        if name == CLIM_NAME:
            run_climatology(args.workers)
        else:
            run_baseline(name, args.workers)


if __name__ == "__main__":
    main()
