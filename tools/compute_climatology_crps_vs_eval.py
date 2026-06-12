"""Exact WeatherBench2 probabilistic-climatology CRPS denominator for CRPSS.

Replaces the analytical-Gaussian workaround (CRPS_clim = sigma_clim/sqrt(pi)).
Follows the WB2 / ECMWF procedure (Rasp et al. 2024, arXiv:2308.15560):

  * Probabilistic climatology = the 30 years 1990-2019 as ensemble members,
    matched by calendar (month, day, hour-of-day), WITHOUT smoothing.
  * Scored with the fair (M-1) CRPS estimator against the actual evaluation
    truth at each forecast valid time -- NOT a leave-one-out estimate over the
    reference period. This is the key difference from
    compute_climatology_1990_2019.py: the denominator sees the same 2023/2024
    observations as the model forecasts, so climate trend over the window is
    handled exactly as WB2 does.
  * cos(lat) spatial weighting, identical to scores.create_latitude_weights
    used for the forecast CRPS in SwissClim_Evaluations.

For each (variable, level, lead) the climatology forecast is time-invariant,
so its CRPS varies with lead only through the shifted valid-time sample. We
therefore compute one fair-CRPS scalar per unique valid time, then average
over the 112 initialisations to get CRPS_clim(variable, level, lead).

  fair CRPS(t) = <  (1/M) sum_m |f_m - o_t|
                  - 1/(2 M (M-1)) sum_{i,j} |f_i - f_j|  >_{cos-lat}

with M = 30 climatology members. The pairwise spread uses the Gini sort
identity  sum_{i,j}|f_i-f_j| = 2 sum_i (2i - M - 1) f_(i).

Output: crps_clim_eval_1990_2019.json
  { "2m_temperature": {"0": v, "6": v, ...}, "geopotential_500": {...}, ... }
keyed by variable (2D) or variable_level (3D), each a lead-hour -> CRPS map.

Usage:
  python tools/compute_climatology_crps_vs_eval.py --parallel 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import dask

sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Config -- production evaluation grid (112 inits, 61 leads)
# ---------------------------------------------------------------------------
CLIM_SRC = "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original"
TRUTH_SRC = {
    2022: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    2023: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    2024: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
    2025: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
}

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS_3D = (500, 850)

CLIM_YEAR_START = 1990
CLIM_YEAR_END = 2019

# Evaluation grids. Each CRPSS denominator must be scored over the SAME
# (init, lead) valid-time sample as its numerator, so we keep one preset per
# evaluation grid in the paper.
#   production : 112-init headline grid (Tab. headline8way, Fig. headline-crpss)
#   ablation   : 4 mid-season inits, 240 h (Tab. calibration / uplift)
GRIDS = {
    "production": {
        # 2023+2024, week-starts Jan/Apr/Jul/Oct 2-8, 00+12 UTC -> 112 inits.
        "init_dates": [
            datetime(y, m, d, h)
            for y in (2023, 2024)
            for m in (1, 4, 7, 10)
            for d in range(2, 9)
            for h in (0, 12)
        ],
        "leads": tuple(range(0, 361, 6)),  # 0..360 -> 61 steps
        "tag": "1990_2019",
    },
    "ablation": {
        # 4 mid-season inits at 00 UTC, 240 h rollout -> 41 leads.
        "init_dates": [
            datetime(2023, 5, 15, 0),
            datetime(2023, 8, 15, 0),
            datetime(2024, 2, 15, 0),
            datetime(2024, 11, 15, 0),
        ],
        "leads": tuple(range(0, 241, 6)),  # 0..240 -> 41 steps
        "tag": "ablation_1990_2019",
    },
}

OUTDIR = Path("/iopsstor/scratch/cscs/sadamov")


def out_paths(tag: str):
    return {
        "json": OUTDIR / f"crps_clim_eval_{tag}.json",
        "prov": OUTDIR / f"crps_clim_eval_{tag}_provenance.json",
        "ckpt": OUTDIR / f"crps_clim_eval_{tag}_checkpoint.jsonl",
        "repo": Path(__file__).resolve().parent / "data" / f"crps_clim_eval_{tag}.json",
    }


# ---------------------------------------------------------------------------
# Eval-grid helpers
# ---------------------------------------------------------------------------
def init_times(grid: dict) -> list[datetime]:
    return list(grid["init_dates"])


def valid_records(grid: dict) -> tuple[list[tuple[int, datetime]], set[datetime]]:
    """Return [(lead_hours, valid_dt)] for all init x lead, plus unique valids."""
    recs = []
    uniq = set()
    for init in init_times(grid):
        for L in grid["leads"]:
            vt = init + timedelta(hours=L)
            recs.append((L, vt))
            uniq.add(vt)
    return recs, uniq


def cos_lat_weights(lat: np.ndarray) -> np.ndarray:
    """Match scores.create_latitude_weights: cos(lat) (unnormalised)."""
    return np.cos(np.deg2rad(lat))


def weighted_spatial_mean(field: np.ndarray, wlat: np.ndarray) -> float:
    """cos(lat)-weighted, skipna spatial mean of a (lat, lon) field."""
    W = np.broadcast_to(wlat[:, None], field.shape)
    mask = np.isfinite(field)
    den = np.sum(np.where(mask, W, 0.0))
    if den == 0:
        return float("nan")
    num = np.sum(np.where(mask, W * field, 0.0))
    return float(num / den)


def gini_spread_term(ens: np.ndarray) -> np.ndarray:
    """Fair spread 1/(2 M (M-1)) sum_{i,j}|f_i-f_j| per pixel via sort identity.

    ens: (M, lat, lon). Returns (lat, lon).
    """
    M = ens.shape[0]
    s = np.sort(ens, axis=0)  # ascending along members
    coef = (2.0 * np.arange(1, M + 1) - M - 1.0).astype(np.float64)  # (M,)
    acc = np.tensordot(coef, s, axes=([0], [0]))  # (lat, lon)
    return acc / (M * (M - 1))


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def open_clim(time_chunk: int = 200) -> xr.Dataset:
    return xr.open_zarr(CLIM_SRC, consolidated=True, chunks={"time": time_chunk, "level": 1})


def _da_field(ds: xr.Dataset, var: str, level: int | None) -> xr.DataArray:
    da = ds[var]
    if level is not None:
        da = da.sel(level=level)
    return da


def load_clim_ensemble(
    var: str, level: int | None, buckets: set[tuple[int, int, int]]
) -> tuple[dict[tuple[int, int, int], np.ndarray], np.ndarray]:
    """Per (month, day, hour) bucket, the 30-member (year) climatology ensemble.

    Pulled one calendar year at a time (memory-safe, mirrors
    compute_climatology_1990_2019.py). Returns {bucket: (M, lat, lon)} and lat.
    """
    ds = open_clim()
    da = _da_field(ds, var, level)
    tvals = da["time"]
    years = tvals.dt.year.values
    months = tvals.dt.month.values
    days = tvals.dt.day.values
    hours = tvals.dt.hour.values
    lat = da["latitude"].values

    need_months = {b[0] for b in buckets}
    need_days = {b[1] for b in buckets}
    need_hours = {b[2] for b in buckets}

    lat_n, lon_n = da.sizes["latitude"], da.sizes["longitude"]
    # member slot = (year - CLIM_YEAR_START)
    n_years = CLIM_YEAR_END - CLIM_YEAR_START + 1
    ens = {b: np.full((n_years, lat_n, lon_n), np.nan, dtype=np.float32) for b in buckets}
    filled = {b: 0 for b in buckets}

    for yr in range(CLIM_YEAR_START, CLIM_YEAR_END + 1):
        sel_mask = (
            (years == yr)
            & np.isin(months, list(need_months))
            & np.isin(days, list(need_days))
            & np.isin(hours, list(need_hours))
        )
        idx = np.where(sel_mask)[0]
        if idx.size == 0:
            continue
        sub = da.isel(time=idx)
        block = sub.load().values  # (k, lat, lon)
        bm = sub["time"].dt.month.values
        bd = sub["time"].dt.day.values
        bh = sub["time"].dt.hour.values
        slot = yr - CLIM_YEAR_START
        for j in range(block.shape[0]):
            b = (int(bm[j]), int(bd[j]), int(bh[j]))
            if b in ens:
                ens[b][slot] = block[j]
                filled[b] += 1
        print(
            f"     [{var}{'' if level is None else '_'+str(level)}] year {yr}: "
            f"{idx.size} steps",
            flush=True,
        )

    # Drop NaN member slots (e.g. a missing date) so M reflects real members.
    cleaned = {}
    for b, arr in ens.items():
        good = np.isfinite(arr).all(axis=(1, 2))
        cleaned[b] = arr[good]
    return cleaned, lat


def load_truth(
    var: str, level: int | None, valids: set[datetime]
) -> dict[np.datetime64, np.ndarray]:
    """Load ERA5 truth (lat, lon) for each unique valid time, from the right zarr."""
    by_year: dict[int, list[datetime]] = {}
    for vt in valids:
        by_year.setdefault(vt.year, []).append(vt)

    out: dict[np.datetime64, np.ndarray] = {}
    for yr, vts in by_year.items():
        path = TRUTH_SRC[yr]
        ds = xr.open_zarr(path, consolidated=True, decode_timedelta=True)
        da = _da_field(ds, var, level)
        times = np.array([np.datetime64(vt, "ns") for vt in sorted(vts)])
        sub = da.sel(time=times).load()
        vals = sub.values  # (k, lat, lon)
        for k, t in enumerate(times):
            out[t] = vals[k]
    return out


# ---------------------------------------------------------------------------
# Per-field driver
# ---------------------------------------------------------------------------
def compute_field(var: str, level: int | None, grid: dict) -> dict:
    label = var if level is None else f"{var}_{level}"
    t0 = time.time()
    recs, uniq = valid_records(grid)
    buckets = {(vt.month, vt.day, vt.hour) for vt in uniq}
    print(f"  -> [{label}] {len(uniq)} unique valids, {len(buckets)} clim buckets", flush=True)

    ens, lat = load_clim_ensemble(var, level, buckets)
    wlat = cos_lat_weights(lat)
    spread = {b: gini_spread_term(ens[b]) for b in ens}
    print(f"     [{label}] clim+spread ready ({time.time()-t0:.0f}s)", flush=True)

    truth = load_truth(var, level, uniq)
    print(f"     [{label}] truth loaded ({time.time()-t0:.0f}s)", flush=True)

    # One fair-CRPS scalar per unique valid time.
    crps_vt: dict[datetime, float] = {}
    for vt in uniq:
        b = (vt.month, vt.day, vt.hour)
        members = ens[b]
        o = truth[np.datetime64(vt, "ns")]
        skill = np.abs(members - o[None]).mean(axis=0)  # (lat, lon)
        crps_pix = skill - spread[b]
        crps_vt[vt] = weighted_spatial_mean(crps_pix, wlat)

    # Average over inits per lead.
    per_lead: dict[str, float] = {}
    inits = init_times(grid)
    for L in grid["leads"]:
        vals = [crps_vt[init + timedelta(hours=L)] for init in inits]
        per_lead[str(L)] = float(np.nanmean(vals))

    print(
        f"     [{label}] done ({time.time()-t0:.0f}s) " f"CRPS@240h={per_lead.get('240'):.6g}",
        flush=True,
    )
    n_members = int(np.median([ens[b].shape[0] for b in ens]))
    return {"label": label, "leads": per_lead, "n_members": n_members, "n_inits": len(inits)}


def _worker(var: str, level: int | None, grid: dict) -> dict:
    for env in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env, "1")
    return compute_field(var, level, grid)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default="production")
    ap.add_argument("--variables", nargs="+", default=None)
    ap.add_argument("--levels", nargs="+", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16, help="dask threads per worker")
    ap.add_argument("--parallel", type=int, default=2, help="concurrent (var, level) workers")
    args = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=args.workers)

    grid = GRIDS[args.grid]
    paths = out_paths(grid["tag"])
    print(
        f"Grid '{args.grid}': {len(grid['init_dates'])} inits, {len(grid['leads'])} leads "
        f"-> {paths['json'].name}",
        flush=True,
    )

    req_vars = args.variables or (VARS_2D + VARS_3D)
    req_levels = tuple(args.levels) if args.levels else LEVELS_3D

    work: list[tuple[str, int | None]] = []
    for var in req_vars:
        if var in VARS_2D:
            work.append((var, None))
        elif var in VARS_3D:
            for lvl in req_levels:
                work.append((var, int(lvl)))
        else:
            print(f"WARN: unknown variable {var!r}, skipping", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    def key_of(var, level):
        return var if level is None else f"{var}_{level}"

    out: dict[str, dict] = {}
    rows: list[dict] = []
    done: set[str] = set()
    if paths["ckpt"].exists():
        with paths["ckpt"].open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                done.add(rec["label"])
                out[rec["label"]] = rec["leads"]
                rows.append(rec)
        paths["json"].write_text(json.dumps(out, indent=2))
        print(f"Resuming; {len(done)} fields already done: {sorted(done)}", flush=True)

    todo = [(v, lv) for (v, lv) in work if key_of(v, lv) not in done]
    if not todo:
        print("Nothing to do; checkpoint complete.", flush=True)
        return
    print(f"Fields remaining: {len(todo)}/{len(work)}; parallel={args.parallel}", flush=True)

    with paths["ckpt"].open("a", buffering=1) as ckpt:
        with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as ex:
            futs = {ex.submit(_worker, v, lv, grid): (v, lv) for (v, lv) in todo}
            for fut in as_completed(futs):
                v, lv = futs[fut]
                key = key_of(v, lv)
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  FAILED {key}: {type(e).__name__}: {e}", flush=True)
                    continue
                rows.append(row)
                out[key] = row["leads"]
                ckpt.write(json.dumps(row) + "\n")
                ckpt.flush()
                os.fsync(ckpt.fileno())
                paths["json"].write_text(json.dumps(out, indent=2))
                print(f"  -> committed {key}", flush=True)

    paths["json"].write_text(json.dumps(out, indent=2))
    paths["repo"].parent.mkdir(parents=True, exist_ok=True)
    paths["repo"].write_text(json.dumps(out, indent=2))
    paths["prov"].write_text(
        json.dumps(
            {
                "method": "WB2 probabilistic climatology (Rasp2024), fair CRPS vs eval truth",
                "grid": args.grid,
                "clim_source": CLIM_SRC,
                "clim_years": [CLIM_YEAR_START, CLIM_YEAR_END],
                "truth_sources": TRUTH_SRC,
                "init_dates": [d.isoformat() for d in grid["init_dates"]],
                "leads": list(grid["leads"]),
                "n_inits": len(grid["init_dates"]),
                "lat_weighting": "cos(lat), matches scores.create_latitude_weights",
                "per_field": rows,
            },
            indent=2,
        )
    )
    print(f"\n-> {paths['json']}\n-> {paths['repo']}\n-> {paths['prov']}")


if __name__ == "__main__":
    main()
