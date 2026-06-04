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
/iopsstor/scratch/cscs/sadamov/venvs/ai-models-ensembles. The HTTPS path
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
import time
from pathlib import Path

import numpy as np
import xarray as xr
import dask


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


# ---------------------------------------------------------------------------
# Climatology helpers
# ---------------------------------------------------------------------------
def open_wb2(time_chunk: int = 400) -> xr.Dataset:
    """Open the local WB2-original reanalysis (1959-2021, 0.25 deg, 37 levels)."""
    return xr.open_zarr(WB2_LOCAL, consolidated=True, chunks={"time": time_chunk})


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


def fair_crps_leave_one_out(da_yearly: np.ndarray) -> np.ndarray:
    """Per-pixel, per-DOY empirical fair-CRPS leave-one-out.

    Input: array (n_years, ..., lat, lon) of values per pixel per DOY (one
    DOY).  Treats each year's value as truth and the other (n_years - 1)
    years as the climatological ensemble.  Returns the mean fair-CRPS over
    years.

    fair-CRPS for an M-member ensemble {x_i} and truth y:
        CRPS = (1/M) sum_i |x_i - y|
                 - (1/(2 M (M-1))) sum_{i,j} |x_i - x_j|
    With LOO M = n_years - 1.

    The implementation operates per pixel.  For 30 years it is cheap.
    """
    n_years = da_yearly.shape[0]
    if n_years < 2:
        return np.full(da_yearly.shape[1:], np.nan, dtype=np.float32)

    # Full pairwise |x_i - x_j| matrix (n_years, n_years, ...)
    diff = np.abs(da_yearly[:, None] - da_yearly[None, :])  # (n_years, n_years, ...)

    # For each leave-one-out: truth = y_k, ensemble = all except k.
    # Term A: (1/M) sum_{i != k} |x_i - y_k| = (1/M) (sum_i diff[i, k] - 0)
    # since diff[k,k] = 0.  Here M = n_years - 1.
    M = n_years - 1
    sum_abs_diff_to_truth = diff.sum(axis=0)  # (n_years, ..., lat, lon)
    term_A = sum_abs_diff_to_truth / M  # truth index along axis 0

    # Term B: spread of the LOO ensemble = (1/(2 M (M-1))) sum_{i != k, j != k} diff[i, j]
    # = (1/(2 M (M-1))) * (S_all - 2*S_row_k + 0)
    # where S_all = sum_{i,j} diff[i,j] (independent of k) and
    # S_row_k = sum_j diff[k, j] = sum_i diff[i, k] (since diff is symmetric)
    S_all = diff.sum(axis=(0, 1))  # (..., lat, lon)
    S_row = diff.sum(axis=1)  # (n_years, ..., lat, lon)
    spread_loo = (S_all[None] - 2.0 * S_row) / (2.0 * M * (M - 1))

    crps_per_year = term_A - spread_loo  # (n_years, ..., lat, lon)
    return crps_per_year.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-variable driver
# ---------------------------------------------------------------------------
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

    # ---- Per-bin LOO fair-CRPS (require >=3 years per bin) ----
    crps_pix_accum = np.zeros(da.shape[-2:], dtype=np.float64)
    bin_count = 0
    skipped = 0
    t_bins = time.time()
    for bi in range(n_bins):
        present = stage_count[:, bi].astype(bool)
        if present.sum() < 3:
            skipped += 1
            continue
        sub = stage[present, bi]  # (n_years_present, lat, lon)
        crps_pix_accum += fair_crps_leave_one_out(sub).astype(np.float64)
        bin_count += 1
        if (bi + 1) % 200 == 0:
            elapsed = time.time() - t_bins
            print(f"     ... CRPS bin {bi+1}/{n_bins} ({elapsed:.0f}s)")
    if bin_count == 0:
        raise RuntimeError("No usable DOY-hour bins.")
    del stage  # free memory before downstream

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

    crps_pix = (crps_pix_accum / bin_count).astype(np.float32)
    crps_scalar = float(np.average(crps_pix, axis=0, weights=w).mean())
    print(f"     empirical_crps_clim (cos-lat spatial mean) = {crps_scalar:.6g}")
    if skipped:
        print(f"     ({skipped}/{len(unique_keys)} bins skipped: <2 years)")

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
    args = ap.parse_args()

    dask.config.set(scheduler="threads", num_workers=args.workers)

    requested_vars = args.variables
    if requested_vars is None:
        requested_vars = VARS_2D + VARS_3D
    requested_levels = tuple(args.levels) if args.levels else LEVELS_3D

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Opening WB2 ARCO-ERA5 zarr over HTTPS ...")
    ds = open_wb2()
    print(f"  ds time range: {ds.time.min().values} .. {ds.time.max().values}")

    sigma_dict: dict[str, float] = {}
    crps_dict: dict[str, float] = {}
    rows = []

    for var in requested_vars:
        if var in VARS_2D:
            row = compute_one_field(
                ds,
                var,
                None,
                args.year_start,
                args.year_end,
                tuple(args.hours),
                dry_run=args.dry_run,
            )
            rows.append(row)
            if not args.dry_run:
                sigma_dict[var] = {
                    "unconditional": row["sigma_clim"],
                    "doy_conditional": row["sigma_clim_doy"],
                }
                crps_dict[var] = row["empirical_crps_clim"]
        elif var in VARS_3D:
            for lvl in requested_levels:
                row = compute_one_field(
                    ds,
                    var,
                    lvl,
                    args.year_start,
                    args.year_end,
                    tuple(args.hours),
                    dry_run=args.dry_run,
                )
                rows.append(row)
                if not args.dry_run:
                    key = f"{var}_{lvl}"
                    sigma_dict[key] = {
                        "unconditional": row["sigma_clim"],
                        "doy_conditional": row["sigma_clim_doy"],
                    }
                    crps_dict[key] = row["empirical_crps_clim"]
        else:
            print(f"WARN: unknown variable {var!r}, skipping")

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
