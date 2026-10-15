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

Reference "ifs_ens" / "ifs_ens10": the WB2 IFS-ENS zarr with all 50 perturbed members
(no control) and with members 0, 5, ..., 45, from one read. Its missing slabs (whole
init/lead/field, ~20%) stay NaN.

Output: $AIENS_STORE/analysis/ens_anom_gaussianity/<baseline>[_<tag>].npz
  z (nvarlev, nlead, ninit, M, npoint) float32, mean / std (nvarlev, nlead, ninit, npoint)
  float32 ensemble mean and ddof=1 std in native units (x_m = mean + z * std), p (nvarlev,
  nlead, ninit, npoint) float32 Shapiro-Wilk p-value of the M members, labels, lead_hours,
  inits, lat, lon.
  For era5_clim ninit = 3 x ninit (init-major, year group minor).

Usage (CPU sbatch): python tools/compute_ens_anomalies.py --baselines all era5_clim
  Shapiro-Wilk sample (figures/ens_anom/member_shapiro_rate): --baselines all era5_clim ifs_ens
  --init-stride 1 --n-points 8192 --vars 2m_temperature specific_humidity_850
  geopotential_500 --tag sw --p-only
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy.stats import shapiro

from _env import BASELINES, IFS_ENS, STORE, WB2_ORIGINAL

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
IFS_NAME = "ifs_ens"  # all 50 perturbed members of the WB2 IFS-ENS zarr (no control)
IFS10_NAME = "ifs_ens10"
IFS10_MEMBERS = list(range(0, 50, 5))  # the 10-member IFS-ENS of the CRPSS / SSR figures
FIELDS = ("z", "mean", "std", "p")


def sample_points(lat: np.ndarray, nlon: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(POINT_SEED)
    w = np.repeat(np.cos(np.deg2rad(lat)).clip(min=0.0), nlon)
    flat = rng.choice(w.size, size=n, replace=False, p=w / w.sum())
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


def sw_pvalue(z: np.ndarray, axis: int) -> np.ndarray:
    """Shapiro-Wilk p-value of the members on `axis` (NaN where the members are constant)."""
    return shapiro(np.moveaxis(z, axis, -1), axis=-1).pvalue.astype(np.float32)


def var_label(var: str, lev) -> str:
    return f"{var}" if lev is None else f"{var}_{lev}"


def production_inits(stride: int) -> list[str]:
    root = BASELINES / ALL_BASELINES[0]
    inits = sorted(p.name for p in root.iterdir() if (p / "forecast.zarr").is_dir())
    if len(inits) != 112:
        raise RuntimeError(f"expected 112 inits, found {len(inits)}")
    return inits[::stride]


def save(name: str, arrays: dict, inits, lat, lon, vars_) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{name}.npz"
    np.savez(
        out,
        **arrays,
        labels=np.array([var_label(v, lev) for v, lev in vars_]),
        lead_hours=np.array(LEAD_HOURS),
        inits=np.array(inits),
        lat=lat,
        lon=lon,
    )
    nan_frac = float(np.isnan(arrays["p"]).mean())
    print(f"{name}: wrote {out} p shape={arrays['p'].shape} p nan_frac={nan_frac:.2e}")


def allocate(nv: int, ni: int, M: int, n_points: int, p_only: bool) -> dict[str, np.ndarray]:
    """Output arrays (nvarlev, nlead, ninit, [M,] npoint), NaN-filled."""
    shape = (nv, len(LEAD_HOURS), ni)
    keep = ("p",) if p_only else FIELDS
    return {
        k: np.full(shape + ((M,) if k == "z" else ()) + (n_points,), np.nan, np.float32)
        for k in keep
    }


def members_out(x: np.ndarray, axis: int) -> tuple[np.ndarray, ...]:
    """z, mean, std, Shapiro-Wilk p of the members on `axis`."""
    z, mu, sd = standardize(x, axis=axis)
    return z, mu, sd, sw_pvalue(z, axis=axis)


def process_init(path: str, ilat: np.ndarray, ilon: np.ndarray, vars_) -> tuple[np.ndarray, ...]:
    ds = xr.open_zarr(path, chunks=None)
    lead_h = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int)
    li = [int(np.flatnonzero(lead_h == h)[0]) for h in LEAD_HOURS]
    x = np.empty((len(vars_), len(LEAD_HOURS), ds.sizes["ensemble"], ilat.size), np.float32)
    for k, (var, lev) in enumerate(vars_):
        da = ds[var].isel(init_time=0)
        if lev is not None:
            da = da.sel(level=lev)
        for j, lt in enumerate(li):
            v = da.isel(lead_time=lt).transpose("ensemble", "latitude", "longitude").values
            x[k, j] = v[:, ilat, ilon]
    return members_out(x, axis=2)


def process_ifs_init(init: str, ilat: np.ndarray, ilon: np.ndarray, vars_) -> tuple:
    """(all 50 members, IFS10_MEMBERS) outputs of one IFS-ENS init; missing slabs stay NaN."""
    ds = xr.open_zarr(IFS_ENS, chunks=None).sel(
        init_time=datetime.strptime(init, "%Y%m%d_%H%M"),
        lead_time=[np.timedelta64(h, "h") for h in LEAD_HOURS],
    )
    x = np.empty((len(vars_), len(LEAD_HOURS), ds.sizes["ensemble"], ilat.size), np.float32)
    for k, (var, lev) in enumerate(vars_):
        da = ds[var] if lev is None else ds[var].sel(level=lev)
        for j in range(len(LEAD_HOURS)):
            v = da.isel(lead_time=j).transpose("ensemble", "latitude", "longitude").values
            x[k, j] = v[:, ilat, ilon]
    return members_out(x, axis=2), members_out(x[:, :, IFS10_MEMBERS], axis=2)


def process_clim_time(
    month: int, day: int, hour: int, ilat: np.ndarray, ilon: np.ndarray, vars_
) -> tuple[np.ndarray, ...]:
    """z (nvarlev, CLIM_GROUPS, M, npoint), mean/std/p (nvarlev, CLIM_GROUPS, npoint)."""
    ds = xr.open_zarr(WB2_ORIGINAL, consolidated=True, chunks=None)
    x = np.empty((len(vars_), len(CLIM_YEARS), ilat.size), np.float32)
    for y, year in enumerate(CLIM_YEARS):
        t = np.datetime64(f"{year}-{month:02d}-{day:02d}T{hour:02d}")
        for var in dict.fromkeys(v for v, _ in vars_):
            rows = [k for k, (v, _) in enumerate(vars_) if v == var]
            levs = [vars_[k][1] for k in rows]
            da = ds[var].sel(time=t)
            if levs[0] is None:
                x[rows[0], y] = da.values[ilat, ilon]
            else:
                vals = da.sel(level=levs).values  # one read of the 37-level chunk
                for r, k in enumerate(rows):
                    x[k, y] = vals[r][ilat, ilon]
    groups = np.stack([x[:, g::CLIM_GROUPS] for g in range(CLIM_GROUPS)], axis=1)
    return members_out(groups, axis=2)


def run_climatology(
    workers: int, inits: list[str], vars_, n_points: int, tag: str, p_only: bool
) -> None:
    valid = {}
    for j, it in enumerate(inits):
        t0 = datetime.strptime(it, "%Y%m%d_%H%M")
        for li, h in enumerate(LEAD_HOURS):
            t = t0 + timedelta(hours=h)
            valid[(j, li)] = (t.month, t.day, t.hour)
    times = sorted(set(valid.values()))
    ds0 = xr.open_zarr(WB2_ORIGINAL, consolidated=True)
    lat, lon = ds0.latitude.values, ds0.longitude.values
    ilat, ilon = sample_points(lat, lon.size, n_points)

    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_clim_time, *mdh, ilat, ilon, vars_): mdh for mdh in times}
        for n, fut in enumerate(as_completed(futs), 1):
            res[futs[fut]] = dict(zip(FIELDS, fut.result(), strict=True))
            print(f"  {CLIM_NAME}: {n}/{len(times)} calendar times ({time.time() - t0:.0f}s)")

    M = len(CLIM_YEARS) // CLIM_GROUPS
    out = allocate(len(vars_), len(inits) * CLIM_GROUPS, M, n_points, p_only)
    for (j, li), mdh in valid.items():
        g = slice(j * CLIM_GROUPS, (j + 1) * CLIM_GROUPS)
        for k, a in out.items():
            a[:, li, g] = res[mdh][k]
    save(CLIM_NAME + tag, out, inits, lat[ilat], lon[ilon], vars_)


def run_baseline(
    name: str, workers: int, inits: list[str], vars_, n_points: int, tag: str, p_only: bool
) -> None:
    root = BASELINES / name
    ds0 = xr.open_zarr(root / inits[0] / "forecast.zarr", chunks=None)
    lat = ds0.latitude.values
    lon = ds0.longitude.values
    ilat, ilon = sample_points(lat, lon.size, n_points)

    out = allocate(len(vars_), len(inits), ds0.sizes["ensemble"], n_points, p_only)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(process_init, str(root / it / "forecast.zarr"), ilat, ilon, vars_): j
            for j, it in enumerate(inits)
        }
        for n, fut in enumerate(as_completed(futs), 1):
            r = dict(zip(FIELDS, fut.result(), strict=True))
            for k, a in out.items():
                a[:, :, futs[fut]] = r[k]
            print(f"  {name}: {n}/{len(inits)} inits ({time.time() - t0:.0f}s)")
    save(name + tag, out, inits, lat[ilat], lon[ilon], vars_)


def run_ifs(workers: int, inits: list[str], vars_, n_points: int, tag: str, p_only: bool) -> None:
    """IFS_NAME (50 members) and IFS10_NAME from one read of the IFS-ENS zarr."""
    ds0 = xr.open_zarr(IFS_ENS, chunks=None)
    lat, lon = ds0.latitude.values, ds0.longitude.values
    ilat, ilon = sample_points(lat, lon.size, n_points)
    outs = [
        allocate(len(vars_), len(inits), ds0.sizes["ensemble"], n_points, p_only),
        allocate(len(vars_), len(inits), len(IFS10_MEMBERS), n_points, p_only),
    ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_ifs_init, it, ilat, ilon, vars_): j for j, it in enumerate(inits)}
        for n, fut in enumerate(as_completed(futs), 1):
            for out, res in zip(outs, fut.result(), strict=True):
                r = dict(zip(FIELDS, res, strict=True))
                for k, a in out.items():
                    a[:, :, futs[fut]] = r[k]
            print(f"  {IFS_NAME}: {n}/{len(inits)} inits ({time.time() - t0:.0f}s)")
    for name, out in zip((IFS_NAME, IFS10_NAME), outs, strict=True):
        save(name + tag, out, inits, lat[ilat], lon[ilon], vars_)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["all"])
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--init-stride", type=int, default=INIT_STRIDE)
    ap.add_argument("--n-points", type=int, default=N_POINTS)
    ap.add_argument("--vars", nargs="+", default=None, help="labels, e.g. geopotential_500")
    ap.add_argument("--tag", default="", help="output <baseline>_<tag>.npz")
    ap.add_argument("--p-only", action="store_true", help="store only the Shapiro-Wilk p")
    args = ap.parse_args()
    names = [n for b in args.baselines for n in (ALL_BASELINES if b == "all" else [b])]
    labels = [var_label(*v) for v in VARS]
    vars_ = VARS if args.vars is None else [VARS[labels.index(lab)] for lab in args.vars]
    inits = production_inits(args.init_stride)
    tag = f"_{args.tag}" if args.tag else ""
    for name in names:
        if name == CLIM_NAME:
            run_climatology(args.workers, inits, vars_, args.n_points, tag, args.p_only)
        elif name == IFS_NAME:
            run_ifs(args.workers, inits, vars_, args.n_points, tag, args.p_only)
        else:
            run_baseline(name, args.workers, inits, vars_, args.n_points, tag, args.p_only)


if __name__ == "__main__":
    main()
