"""Compute 1990-2019 WB2-ERA5 climatology denominators for CRPSS.

Two outputs per (variable, level):
  1. sigma_clim_1990_2019.json
     Per-pixel temporal standard deviation of the value (after sub-sampling
     daily 00 UTC), then cos-lat spatial mean. Same JSON schema as
     /iopsstor/scratch/cscs/sadamov/sigma_clim_ablation.json. Drop-in for the
     Gaussian-CRPS formula s/sqrt(pi).
  2. empirical_crps_clim_1990_2019.json
     A non-parametric, leave-one-out fair-CRPS of a per-(pixel, DOY) 30-year
     climatological ensemble against each year's truth, then cos-lat spatial
     mean. Use this when CRPS_clim assumption of Gaussianity matters (MSL, q,
     2m_t) -- replaces s/sqrt(pi) in CRPSS = 1 - CRPS / CRPS_clim.

Reads the WB2 ARCO-ERA5 zarr over HTTPS (no gcsfs needed; the bucket is
public). Designed for the project venv at
/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles. The HTTPS path
also dodges the broken cffi/_cffi_backend wheel in that venv.

Usage:
  python tools/compute_climatology_1990_2019.py \\
      --variables 2m_temperature \\
      --year-end 1990 \\
      --hours 0
The defaults compute the full 7-variable, 30-year, 4-hour-of-day product.
Pass --dry-run for a quick wall-time probe on 1990 only, no outputs written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
import dask

sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Local 0.25 deg ERA5 reanalysis 1959-2021 -- covers our 1990-2019 climatology
# window in full. All 7 paper vars (2m_t, MSL, geopotential, temperature,
# u/v_wind, specific_humidity) at 37 pressure levels. Faster than the public
# WB2 HTTPS bucket and avoids egress + cert dependence.
WB2_LOCAL = "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original"

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS_3D = (500, 850)

YEAR_START_DEFAULT = 1990
YEAR_END_DEFAULT = 2019
HOURS_DEFAULT = (0, 6, 12, 18)

OUTDIR = Path("/iopsstor/scratch/cscs/sadamov")
SIGMA_OUT = OUTDIR / "sigma_clim_1990_2019.json"
CRPS_OUT = OUTDIR / "empirical_crps_clim_1990_2019.json"
PROVENANCE_OUT = OUTDIR / "climatology_1990_2019_provenance.json"
# Per-(var, level) JSONL checkpoint -- idempotent resume. Each line is one
# completed field's row dict; on restart the script skips any keys already
# present and merges new keys into the same outputs. Lines are appended
# atomically (single write + flush + fsync) so a kill never leaves the file
# in a half-written state.
CHECKPOINT = OUTDIR / "climatology_1990_2019_checkpoint.jsonl"


# ---------------------------------------------------------------------------
# Climatology helpers
# ---------------------------------------------------------------------------
def open_wb2(time_chunk: int = 200) -> xr.Dataset:
    """Open the local WB2-original reanalysis (1959-2021, 0.25 deg, 37 levels).

    Chunking strategy: ``chunks={"time": 200, "level": 1}``. Pinning level to
    one chunk-per-level is essential for 3D variables. The on-disk zarr packs
    all 37 levels into a single chunk; without an explicit per-level chunk
    hint, ``.sel(level=lvl).load()`` materialises the FULL 37-level slab
    before slicing, blowing through memory. A naive ``chunks={"time": 1460}``
    (one calendar year per chunk) was tried and caused the clim1990 2476374
    job to OOM at 2 min 16 s because each 3D-var chunk read becomes
    1460 x 37 x 721 x 1440 x 4 = 222 GB -- two workers alone push 444 GB
    just for the chunk reads, before the staging cube is even allocated.
    With ``time=200, level=1`` each 3D-var chunk is
    200 x 1 x 721 x 1440 x 4 = 830 MB, comfortably safe.
    """
    return xr.open_zarr(WB2_LOCAL, consolidated=True, chunks={"time": time_chunk, "level": 1})


def select_subset(
    ds: xr.Dataset,
    var: str,
    level: int | None,
    year_start: int,
    year_end: int,
    hours: tuple[int, ...],
) -> xr.DataArray:
    """Slice (variable, level, year-window, hour-of-day)."""
    da = ds[var]
    if level is not None:
        da = da.sel(level=level)
    # Year window
    time_mask = (da["time"].dt.year >= year_start) & (da["time"].dt.year <= year_end)
    da = da.isel(time=time_mask)
    # Hour-of-day subsample
    hour_mask = da["time"].dt.hour.isin(list(hours))
    da = da.isel(time=hour_mask)
    return da


def cos_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def fair_crps_loo_vectorized(
    stage: np.ndarray,
    stage_count: np.ndarray,
    min_years: int = 3,
) -> tuple[np.ndarray, int, int]:
    """Per-bin mean LOO fair-CRPS via the Gini sort identity.

    For sorted x_(1) <= ... <= x_(n) per pixel, the mean over leave-one-out
    truth choices k of the fair-CRPS against the n-1 other years simplifies
    to ``sum_{i,j} |x_i - x_j| / (2 n (n-1))``, and the pairwise sum equals
    ``2 sum_i (2i - n - 1) x_(i)`` (1-indexed). Each bin becomes one tiny
    sort + one streaming weighted sum -- ~60x faster than the original
    per-bin diff-matrix path (3.7 GB allocator churn per bin) at <250 MB
    peak transient. The earlier all-bins ``stage.sort(axis=0)`` on the
    full (30, 1464, 721, 1440) cube triggered a hidden ~181 GB scratch in
    numpy's non-contiguous-axis sort path and OOM'd job 2488607 -- this
    per-bin variant avoids that path entirely.

    stage: (n_years, n_bins, lat, lon) float32, NaN where missing.
    stage_count: (n_years, n_bins) int8, 1 where present.
    Returns (crps_mean: (lat, lon) f32, n_valid_bins, n_skipped_bins).
    """
    n_bins = stage.shape[1]
    n_present = stage_count.sum(axis=0).astype(np.int64)

    crps_accum = np.zeros(stage.shape[-2:], dtype=np.float64)
    n_valid = 0
    n_skipped = 0
    log_every = 200
    t0 = time.time()
    for bi in range(n_bins):
        n_p = int(n_present[bi])
        if n_p < min_years:
            n_skipped += 1
            continue
        sub_sorted = np.sort(stage[:, bi], axis=0)  # (n_years, lat, lon)
        # All-present case (n_p == n_years) is the dominant path; NaN sinks
        # to the end so weights truncated at n_p still apply for partial bins.
        weights = 2.0 * np.arange(1, n_p + 1, dtype=np.float32) - n_p - 1.0
        # Streaming weighted sum keeps transient ~ one (lat, lon) panel
        # rather than the (n_years, lat, lon) broadcast einsum would create.
        S = np.zeros(stage.shape[-2:], dtype=np.float32)
        for j in range(n_p):
            S += weights[j] * sub_sorted[j]
        S *= 2.0
        crps_accum += (S / (2.0 * n_p * (n_p - 1))).astype(np.float64)
        n_valid += 1
        if (bi + 1) % log_every == 0:
            print(f"     ... CRPS bin {bi + 1}/{n_bins} ({time.time() - t0:.0f}s)")
    if n_valid == 0:
        raise RuntimeError("No usable DOY-hour bins.")
    return (crps_accum / n_valid).astype(np.float32), n_valid, n_skipped


# ---------------------------------------------------------------------------
# Per-variable driver
# ---------------------------------------------------------------------------
def _worker_one_field(
    var: str,
    level: int | None,
    year_start: int,
    year_end: int,
    hours: tuple[int, ...],
    dry_run: bool,
) -> dict:
    """ProcessPool worker: open WB2 fresh, compute one (var, level) field.

    Workers cannot share xarray Datasets across the multiprocessing boundary
    (zarr async stores are not picklable), so each worker re-opens WB2 on its
    own. Numpy thread fanout is pinned to 1 because the year-by-year pull
    fills 30 GB+ of NumPy arrays per worker, and a 32-thread BLAS gangs up
    badly with multiple ProcessPool workers on a 72-CPU node.
    """
    for env in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env, "1")
    ds = open_wb2()
    return compute_one_field(ds, var, level, year_start, year_end, hours, dry_run=dry_run)


def compute_one_field(
    ds: xr.Dataset,
    var: str,
    level: int | None,
    year_start: int,
    year_end: int,
    hours: tuple[int, ...],
    dry_run: bool = False,
) -> dict:
    """Return per-(var, level) scalars: sigma_clim, empirical CRPS_clim."""
    label = var if level is None else f"{var}_{level}"
    print(f"  -> [{label}] selecting {year_start}-{year_end}, hours={hours}")
    t0 = time.time()
    da = select_subset(ds, var, level, year_start, year_end, hours)
    n_t = da.sizes["time"]
    print(f"     n_time={n_t}, shape={da.shape}, dtype={da.dtype}")
    if dry_run:
        # Force one chunk to materialise to measure throughput
        slice_one = da.isel(time=slice(0, min(40, n_t))).load()
        dt = time.time() - t0
        bytes_one = slice_one.nbytes
        print(
            f"     dry-run: pulled {bytes_one/1e6:.1f} MB in {dt:.1f}s "
            f"({bytes_one/1e6/dt:.1f} MB/s eff.)"
        )
        # Estimate full-field cost in seconds.
        scale = n_t / slice_one.sizes["time"]
        est_full = dt * scale
        print(
            f"     estimated full pull for this field: {est_full:.0f}s = " f"{est_full/60:.1f} min"
        )
        return {"label": label, "dry_run_seconds": dt, "full_estimate_s": est_full, "n_time": n_t}

    # ---- Pull year-at-a-time, bin in memory ----
    # The zarr is chunked time-contiguously (100 ts per chunk).  Bin-by-bin
    # access fetches the same chunks repeatedly.  Instead, pull one calendar
    # year at a time (~10-15 zarr chunks, ~6 GB at 6-hourly 2D, ~1.5 GB at
    # 00 UTC only), then walk the (DOY, hour) bins in memory.
    lat = da["latitude"].values
    w = cos_weights(lat)

    dt_index = da["time"].dt
    years = np.unique(dt_index.year.values)
    keys_all = dt_index.dayofyear.values.astype(np.int64) * 100 + dt_index.hour.values.astype(
        np.int64
    )
    unique_keys = np.unique(keys_all)
    print(f"     {len(unique_keys)} DOY-hour bins; {len(years)} years to pull")

    # Per-(bin, year) staging: stack into shape (n_years, n_bins, lat, lon)
    n_bins = len(unique_keys)
    n_years_total = len(years)
    bin_to_idx = {int(k): i for i, k in enumerate(unique_keys)}

    sum_x = np.zeros(da.shape[-2:], dtype=np.float64)
    sumsq_x = np.zeros(da.shape[-2:], dtype=np.float64)
    n_total = 0

    # Also accumulate DOY-conditional sums for sigma_clim_doy.
    # (annual cycle removed; comparable to the LOO empirical CRPS_clim).
    doy_sum_x = np.zeros((n_bins,) + da.shape[-2:], dtype=np.float64)
    doy_sumsq_x = np.zeros((n_bins,) + da.shape[-2:], dtype=np.float64)
    doy_count = np.zeros(n_bins, dtype=np.int64)

    # Allocate the per-(year, bin, lat, lon) cube.  At 30 yrs x 1460 bins x
    # 721 x 1440 x float32 this is 181 GB -- only feasible on a fat node.
    # For 2D variables we keep float32 (181 GB at full); for 3D-per-level same.
    stage_shape = (n_years_total, n_bins) + da.shape[-2:]
    stage_bytes = int(np.prod(stage_shape)) * 4
    print(f"     staging cube shape={stage_shape} (~{stage_bytes/1e9:.1f} GB float32)")
    stage = np.full(stage_shape, np.nan, dtype=np.float32)
    stage_count = np.zeros((n_years_total, n_bins), dtype=np.int8)

    t_years = time.time()
    for yi, yr in enumerate(years):
        t_y = time.time()
        sel = da.isel(time=(dt_index.year.values == yr))
        block = sel.load().values  # (n_t_yr, lat, lon)
        block_keys = sel["time"].dt.dayofyear.values.astype(np.int64) * 100 + sel[
            "time"
        ].dt.hour.values.astype(np.int64)
        # Per-pixel running sums for sigma
        b64 = block.astype(np.float64, copy=False)
        sum_x += b64.sum(axis=0)
        sumsq_x += (b64 * b64).sum(axis=0)
        n_total += block.shape[0]
        # Bin and stage; also accumulate DOY-conditional running stats.
        for j, key in enumerate(block_keys):
            bi = bin_to_idx[int(key)]
            stage[yi, bi] = block[j]
            stage_count[yi, bi] = 1
            doy_sum_x[bi] += b64[j]
            doy_sumsq_x[bi] += b64[j] * b64[j]
            doy_count[bi] += 1
        dt_y = time.time() - t_y
        eta = (n_years_total - (yi + 1)) * dt_y
        print(f"     year {yr} pulled in {dt_y:.0f}s (eta {eta/60:.1f} min)")

    print(f"     all {n_years_total} years pulled in " f"{(time.time()-t_years)/60:.1f} min")

    # ---- Mean LOO fair-CRPS across (DOY, hour) bins, vectorised ----
    # One in-place sort + one einsum replaces the 1464-bin Python loop. See
    # ``fair_crps_loo_vectorized`` for the Gini-identity derivation.
    t_crps = time.time()
    crps_pix, bin_count, skipped = fair_crps_loo_vectorized(stage, stage_count, min_years=3)
    print(f"     CRPS over {bin_count} bins in {time.time() - t_crps:.0f}s")
    del stage  # ``stage`` is consumed in place by the sort + nan_to_num

    # ---- Sigma_clim (UNCONDITIONAL: annual cycle + interannual) ----
    mean_pix = sum_x / n_total
    var_pix = sumsq_x / n_total - mean_pix * mean_pix
    var_pix = np.clip(var_pix, 0.0, None)
    sigma_pix = np.sqrt(var_pix)
    sigma_scalar = float(np.average(sigma_pix, axis=0, weights=w).mean())
    print(f"     sigma_clim (unconditional, cos-lat spatial mean) = {sigma_scalar:.6g}")

    # ---- Sigma_clim_doy (CONDITIONAL on (DOY, hour); annual cycle removed) ----
    # Per-bin per-pixel stdev, then mean-square-pool over bins, then cos-lat mean.
    doy_mean = doy_sum_x / np.maximum(doy_count[:, None, None], 1)
    doy_var = doy_sumsq_x / np.maximum(doy_count[:, None, None], 1) - doy_mean * doy_mean
    doy_var = np.clip(doy_var, 0.0, None)
    # Average variance over bins, then sqrt; reflects mean within-bin variance.
    var_pix_doy = doy_var.mean(axis=0)
    sigma_pix_doy = np.sqrt(var_pix_doy)
    sigma_doy_scalar = float(np.average(sigma_pix_doy, axis=0, weights=w).mean())
    print(f"     sigma_clim_doy (DOY-conditional, cos-lat) = {sigma_doy_scalar:.6g}")

    crps_scalar = float(np.average(crps_pix, axis=0, weights=w).mean())
    print(f"     empirical_crps_clim (cos-lat spatial mean) = {crps_scalar:.6g}")
    if skipped:
        print(f"     ({skipped}/{len(unique_keys)} bins skipped: <3 years)")

    return {
        "label": label,
        "sigma_clim": sigma_scalar,
        "sigma_clim_doy": sigma_doy_scalar,
        "empirical_crps_clim": crps_scalar,
        "gaussian_crps_clim_unconditional": sigma_scalar / np.sqrt(np.pi),
        "gaussian_crps_clim_doy": sigma_doy_scalar / np.sqrt(np.pi),
        "n_time": n_t,
        "n_bins_used": bin_count,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Restrict to a subset of variables (default: all)",
    )
    ap.add_argument(
        "--levels", nargs="+", type=int, default=None, help="Restrict 3D levels (default 500 850)"
    )
    ap.add_argument("--year-start", type=int, default=YEAR_START_DEFAULT)
    ap.add_argument("--year-end", type=int, default=YEAR_END_DEFAULT)
    ap.add_argument("--hours", nargs="+", type=int, default=list(HOURS_DEFAULT))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe wall-time on one chunk only, do not compute or write.",
    )
    ap.add_argument("--sigma-out", default=str(SIGMA_OUT))
    ap.add_argument("--crps-out", default=str(CRPS_OUT))
    ap.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Dask thread-pool workers for HTTPS reads (default 32)",
    )
    ap.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Number of concurrent (var, level) workers via ProcessPoolExecutor. "
        "Memory-bound: each worker peaks at ~220-260 GB once dask + numpy "
        "scratch is added on top of the 181 GB staging cube. --parallel 3 was "
        "tried (job 2487050) and OOM-killed mid-pull when all three workers "
        "happened to overlap on year-load. --parallel 2 (~520 GB peak) is the "
        "safe ceiling on the 800 GB partition. The vectorised Gini-CRPS keeps "
        "per-key wall at ~120 min vs ~216 min, so 6 keys / 2 workers still "
        "finishes in ~6 h comfortably under the 12 h cap.",
    )
    args = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=args.workers)

    requested_vars = args.variables
    if requested_vars is None:
        requested_vars = VARS_2D + VARS_3D
    requested_levels = tuple(args.levels) if args.levels else LEVELS_3D

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Idempotency: build the full (var, level) work list, then skip keys
    # already present in the per-field checkpoint. Each completed field
    # appends one JSONL line with sigma + crps + dry-run sentinel.
    # ----------------------------------------------------------------------
    work: list[tuple[str, int | None]] = []
    for var in requested_vars:
        if var in VARS_2D:
            work.append((var, None))
        elif var in VARS_3D:
            for lvl in requested_levels:
                work.append((var, int(lvl)))
        else:
            print(f"WARN: unknown variable {var!r}, skipping", flush=True)

    sigma_dict: dict = {}
    crps_dict: dict = {}
    rows: list = []

    def _key_of(var: str, level: int | None) -> str:
        return var if level is None else f"{var}_{level}"

    done_keys: set[str] = set()
    if not args.dry_run and CHECKPOINT.exists():
        with CHECKPOINT.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                k = rec["label"]
                done_keys.add(k)
                sigma_dict[k] = {
                    "unconditional": rec["sigma_clim"],
                    "doy_conditional": rec["sigma_clim_doy"],
                }
                crps_dict[k] = rec["empirical_crps_clim"]
                rows.append(rec)
        # Promote partial-output JSONs so anyone reading them mid-run sees
        # the keys already on disk.
        Path(args.sigma_out).write_text(json.dumps(sigma_dict, indent=2))
        Path(args.crps_out).write_text(json.dumps(crps_dict, indent=2))
        print(
            f"Resuming with {len(done_keys)} keys already in checkpoint: " f"{sorted(done_keys)}",
            flush=True,
        )

    todo = [(v, lv) for (v, lv) in work if _key_of(v, lv) not in done_keys]
    if not todo:
        print("Nothing to do; checkpoint already complete.", flush=True)
        return

    print(f"Keys remaining: {len(todo)} / {len(work)}", flush=True)

    # Parallel: one worker per (var, level) up to ``args.parallel`` concurrent.
    # Each worker opens its own WB2 handle (zarr stores are not picklable),
    # processes the year-by-year pull + per-(DOY, hour) LOO fair-CRPS, returns
    # the row dict. Peak per-worker memory is the staging cube
    # 30 yrs x 1460 bins x 721 x 1440 x f32 = 181 GB at full ERA5 grid; with 4
    # workers concurrent the node sees ~720 GB of resident heap, comfortably
    # inside the 800 GB --mem cap. dial down ``--parallel`` if memory pressure
    # bites or for partial-grid (--levels) runs.
    par = max(1, int(args.parallel))
    print(f"Spawning ProcessPoolExecutor with {par} workers", flush=True)

    # Append-only JSONL writer for atomic per-field commits. Each worker
    # produces one row dict; main writes it on completion (single writer ->
    # no contention). Explicit flush + fsync after each line so partial
    # state survives a hard kill.
    with CHECKPOINT.open("a", buffering=1) as ckpt:
        with ProcessPoolExecutor(max_workers=par) as ex:
            futures = {
                ex.submit(
                    _worker_one_field,
                    var,
                    lvl,
                    args.year_start,
                    args.year_end,
                    tuple(args.hours),
                    args.dry_run,
                ): (var, lvl)
                for (var, lvl) in todo
            }
            for fut in as_completed(futures):
                var, lvl = futures[fut]
                key = _key_of(var, lvl)
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  FAILED {key}: {type(e).__name__}: {e}", flush=True)
                    continue
                rows.append(row)
                if args.dry_run:
                    continue
                sigma_dict[key] = {
                    "unconditional": row["sigma_clim"],
                    "doy_conditional": row["sigma_clim_doy"],
                }
                crps_dict[key] = row["empirical_crps_clim"]
                ckpt.write(json.dumps(row) + "\n")
                ckpt.flush()
                os.fsync(ckpt.fileno())
                Path(args.sigma_out).write_text(json.dumps(sigma_dict, indent=2))
                Path(args.crps_out).write_text(json.dumps(crps_dict, indent=2))
                print(f"  -> committed {key} to {CHECKPOINT}", flush=True)

    if args.dry_run:
        total = sum(r.get("full_estimate_s", 0.0) for r in rows)
        print()
        print("=" * 70)
        print(f"DRY-RUN ESTIMATE for ONE FIELD: {total:.0f}s = {total/60:.1f} min")
        # Scale up to full 12-field product (2 + 5x2).
        n_fields_full = len(VARS_2D) + len(VARS_3D) * len(LEVELS_3D)
        n_done = len(rows)
        full_pull_s = total / max(n_done, 1) * n_fields_full
        print(
            f"FULL 12-field pull estimate: {full_pull_s:.0f}s = "
            f"{full_pull_s/3600:.1f}h (pull only; add ~20% for fair-CRPS loop)"
        )
        return

    Path(args.sigma_out).write_text(json.dumps(sigma_dict, indent=2))
    Path(args.crps_out).write_text(json.dumps(crps_dict, indent=2))
    PROVENANCE_OUT.write_text(
        json.dumps(
            {
                "wb2_source": WB2_LOCAL,
                "year_start": args.year_start,
                "year_end": args.year_end,
                "hours": list(args.hours),
                "variables": requested_vars,
                "levels_3d": list(requested_levels),
                "per_field": rows,
            },
            indent=2,
        )
    )
    print(f"\n-> {args.sigma_out}")
    print(f"-> {args.crps_out}")
    print(f"-> {PROVENANCE_OUT}")


if __name__ == "__main__":
    main()
