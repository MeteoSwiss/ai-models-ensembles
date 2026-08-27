"""Fixed climatological per-channel scale for the multivariate/path scores.

ES, VS and SIGK standardise each (variable, level) channel before taking a
multivariate norm, otherwise geopotential (~1e4 m^2 s^-2) swamps temperature
(~1e2 K). Standardising by the *verifying* field's own spread makes the scale
truth-dependent, which costs the scores their propriety. This computes the same
quantity once, from ERA5 1990-2019, so the scale is a fixed constant known
before any forecast is scored:

    scale(v, p) = mean over sampled valid times of the cos(lat)-weighted
                  spatial standard deviation of the ERA5 field

matching the semantics of the per-init spatial std it replaces (not a per-pixel
temporal sigma, which weights the channels very differently).

Output: tools/data/channel_scale_1990_2019.json  {"<var>[_<level>]": scale}

The WB2-original store packs all 37 pressure levels into one chunk, so a single
`sel(level=500)` read costs ~40 s while a surface field costs 0.05 s. Each
timestep is therefore read once as a full-level slab, both paper levels are
taken from it, and the timesteps are read through a thread pool (zarr releases
the GIL while decompressing).

Usage (CPU sbatch, ~30 min):
  python tools/compute_channel_scale.py --max-times 100 --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

sys.stdout.reconfigure(line_buffering=True)

WB2_LOCAL = "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original"
OUT_JSON = Path(__file__).resolve().parent / "data" / "channel_scale_1990_2019.json"

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS_3D = (500, 850)


def weighted_std(field: np.ndarray, wlat: np.ndarray) -> float:
    W = np.broadcast_to(wlat[:, None], field.shape)
    m = np.isfinite(field)
    den = np.sum(np.where(m, W, 0.0))
    if den <= 0:
        return float("nan")
    mean = np.sum(np.where(m, W * field, 0.0)) / den
    var = np.sum(np.where(m, W * (field - mean) ** 2, 0.0)) / den
    return float(np.sqrt(var))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-start", type=int, default=1990)
    ap.add_argument("--year-end", type=int, default=2019)
    ap.add_argument("--stride-days", type=int, default=7)
    ap.add_argument("--hour", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--max-times", type=int, default=100, help="cap the sample (0 = no cap)")
    ap.add_argument("--workers", type=int, default=16, help="parallel timestep reads")
    args = ap.parse_args()

    print(f"opening {WB2_LOCAL} ...", flush=True)
    t0 = time.time()
    # chunks=None keeps this out of dask entirely: the whole job is a stream of
    # single-timestep reads, and a dask graph over the 1959-2021 x 37-level
    # store costs far more to build than the reads themselves (a 6 h no-output
    # hang on job 3189637, 2026-08-25).
    ds = xr.open_zarr(WB2_LOCAL, consolidated=True, chunks=None)
    print(f"opened in {time.time()-t0:.0f}s, {ds.sizes.get('time')} timesteps", flush=True)
    t = ds["time"]
    sel = (
        (t.dt.year >= args.year_start)
        & (t.dt.year <= args.year_end)
        & (t.dt.hour == args.hour)
        & (t.dt.dayofyear % args.stride_days == 1)
    )
    idx = np.where(sel.values)[0]
    if args.max_times and len(idx) > args.max_times:
        idx = idx[:: int(np.ceil(len(idx) / args.max_times))]
    print(f"{len(idx)} sampled valid times ({args.year_start}-{args.year_end})", flush=True)

    lat = ds["latitude"].values
    wlat = np.cos(np.deg2rad(lat))

    out: dict[str, float] = {}

    def write_out():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

    for var in VARS_2D:
        t0 = time.time()
        da = ds[var].isel(time=idx)
        vals = [weighted_std(da.isel(time=k).values, wlat) for k in range(da.sizes["time"])]
        out[var] = float(np.nanmean(vals))
        print(f"  {var}: scale={out[var]:.6g} ({time.time()-t0:.0f}s)", flush=True)
        write_out()

    for var in VARS_3D:
        t0 = time.time()
        da = ds[var]
        lev = [float(x) for x in da["level"].values.tolist()]
        pos = [lev.index(float(lv)) for lv in LEVELS_3D]

        def one(k, _da=da, _pos=pos):
            slab = _da.isel(time=int(k)).values  # (level, lat, lon), one chunk
            return [weighted_std(slab[p], wlat) for p in _pos]

        acc: list[list[float]] = [[] for _ in LEVELS_3D]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for n, res in enumerate(ex.map(one, idx), start=1):
                for j, v in enumerate(res):
                    acc[j].append(v)
                if n % 20 == 0:
                    print(f"    {var}: {n}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)
        for j, lv in enumerate(LEVELS_3D):
            key = f"{var}_{lv}"
            out[key] = float(np.nanmean(acc[j]))
            print(f"  {key}: scale={out[key]:.6g} ({time.time()-t0:.0f}s)", flush=True)
        write_out()

    print(f"DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
