"""Spatial-mean SSR diagnostic.

Computes the Spread-Skill Ratio of the cos(lat)-weighted GLOBAL SPATIAL MEAN per
(model, variable, lead). This is the field-averaged counterpart of the
per-pixel SSR that SwissClim ssr_combined.csv reports.

Why this matters: persistent-noise weight perturbations produce members whose
biases are COHERENT across space (each member is offset by ~constant), so the
std-across-members of the spatial mean grows roughly as T, while fresh-per-step
noise produces incoherent biases that decorrelate as random-walk ~sqrt(T). The
per-pixel SSR cannot distinguish these regimes (pointwise variance is local);
the spatial-mean SSR can. See memory/phase6_fresh_per_step_weight.md and
memory/spatial_mean_vs_pointwise_ssr.md.

SSR definition (spatial-mean version):

    For each (init, variable, lead, level):
      S_n = sum_ij cos(lat_i) * field_n_ij / sum_ij cos(lat_i)   for member n
      T   = sum_ij cos(lat_i) * truth_ij    / sum_ij cos(lat_i)
      spread^2  = ((M+1)/(M-1)) * var_n(S_n)
      error^2   = ( mean_n(S_n) - T )^2

    Aggregating across inits:
      SSR(var, lead) = sqrt(  mean_init(spread^2) / mean_init(error^2)  )

This matches the standard SSR definition (unbiased variance estimator over an
M-member ensemble, then ratio of root-mean-square spread to root-mean-square
error), applied to the SCALAR time series of global spatial means rather than
to per-pixel fields.

Usage:

    python tools/spatial_mean_ssr.py \\
        --forecast-zarrs <path1> [<path2> ...] \\
        --truth-zarr <truth.zarr> \\
        --variables 2m_temperature mean_sea_level_pressure geopotential ... \\
        --leads 24 72 120 240 \\
        --out-csv spatial_ssr.csv

CSV columns: model, variable, lead_time_hours, level, n_inits, spread, error, ssr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _weighted_spatial_mean(da: xr.DataArray, lat: np.ndarray) -> xr.DataArray:
    """Cos(lat)-weighted area mean over (lat, lon).

    `da` must have dims including 'latitude' and 'longitude'. Returns a
    DataArray with those dims dropped.
    """
    weights = np.cos(np.deg2rad(lat))
    w = xr.DataArray(weights, dims="latitude", coords={"latitude": lat})
    return da.weighted(w).mean(dim=("latitude", "longitude"), skipna=True)


def _compute_for_init(
    fcst: xr.Dataset,
    truth: xr.Dataset,
    variable: str,
    lead_hours: int,
    level: float | None,
) -> dict | None:
    """Return spread^2 and error^2 for one (init, variable, lead, level)."""
    if variable not in fcst.data_vars:
        return None
    fa = fcst[variable]
    ta = truth[variable]

    if level is not None and "level" in fa.dims:
        fa = fa.sel(level=level)
        ta = ta.sel(level=level)

    # Forecast lead-time selection (truth is already time-selected upstream)
    if "lead_time" in fa.dims:
        lt = fa["lead_time"].values
        if np.issubdtype(lt.dtype, np.timedelta64):
            hours = (lt / np.timedelta64(1, "h")).astype(int)
        else:
            hours = lt.astype(int)
        if lead_hours not in hours:
            return None
        idx = int(np.where(hours == lead_hours)[0][0])
        fa = fa.isel(lead_time=idx)
    # Drop singleton init_time dim on the forecast if present
    if "init_time" in fa.dims:
        fa = fa.isel(init_time=0)

    lat = fa["latitude"].values
    s_n = _weighted_spatial_mean(fa, lat)  # dims: (ensemble,) plus any leftover
    t_scalar = float(_weighted_spatial_mean(ta, lat).values)

    s_vals = s_n.values
    s_vals = s_vals[np.isfinite(s_vals)]
    M = s_vals.size
    if M < 2:
        return None

    var_M = float(np.var(s_vals, ddof=1))  # unbiased sample variance
    spread2 = var_M * (M + 1) / (M - 1)  # SSR convention: unbiased variance
    err2 = float((s_vals.mean() - t_scalar) ** 2)
    return {"spread2": spread2, "error2": err2, "n_members": M}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--forecast-zarrs",
        nargs="+",
        required=True,
        help="One or more forecast.zarr paths (one per init).",
    )
    p.add_argument(
        "--truth-zarrs",
        nargs="+",
        required=True,
        help="One or more truth zarrs (e.g. weatherbench2 split by year). Merged via xr.open_mfdataset.",
    )
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[None],
        help="Pressure levels for 3D vars (use 'None' for 2D).",
    )
    p.add_argument("--leads", nargs="+", type=int, required=True)
    p.add_argument(
        "--model-label",
        required=True,
        help="Label written to the 'model' column of the output CSV.",
    )
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    # consolidated=True is required for WB2 2022-2023 / 2024-2025: the raw
    # zarr hierarchy is missing metadata for `mean_sea_level_pressure` and
    # `u_component_of_wind`, but the consolidated .zmetadata path has them.
    # See memory/wb2_truth_zarr_broken_msl_uwind.md.
    truth_parts = [xr.open_zarr(tz, consolidated=True, chunks={}) for tz in args.truth_zarrs]
    truth_ds = xr.concat(truth_parts, dim="time") if len(truth_parts) > 1 else truth_parts[0]
    # Sort by time so .sel() works deterministically
    truth_ds = truth_ds.sortby("time")

    rows: list[dict] = []
    for var in args.variables:
        for lvl in args.levels:
            for lead in args.leads:
                spread2_acc = []
                err2_acc = []
                n_members = None
                for fz in args.forecast_zarrs:
                    fz_ds = xr.open_zarr(fz, consolidated=False, chunks={})
                    # Truth slice for this init+lead
                    if "init_time" in fz_ds.dims and "init_time" in fz_ds[var].dims:
                        init_time = fz_ds["init_time"].values[0]
                    else:
                        init_time = None
                    # Pull a single (init, lead) truth field aligned to fz
                    truth_local = truth_ds
                    if init_time is not None and "time" in truth_ds.dims:
                        valid_time = np.datetime64(init_time) + np.timedelta64(lead, "h")
                        truth_local = truth_ds.sel(time=valid_time)
                    out = _compute_for_init(fz_ds, truth_local, var, lead, lvl)
                    fz_ds.close()
                    if out is None:
                        continue
                    spread2_acc.append(out["spread2"])
                    err2_acc.append(out["error2"])
                    n_members = out["n_members"]
                if not spread2_acc:
                    continue
                spread2_m = float(np.mean(spread2_acc))
                err2_m = float(np.mean(err2_acc))
                ssr = float(np.sqrt(spread2_m / err2_m)) if err2_m > 0 else float("nan")
                rows.append(
                    {
                        "model": args.model_label,
                        "variable": var,
                        "level": lvl if lvl is not None else float("nan"),
                        "lead_time_hours": lead,
                        "n_inits": len(spread2_acc),
                        "n_members": n_members,
                        "spread": float(np.sqrt(spread2_m)),
                        "error": float(np.sqrt(err2_m)),
                        "ssr": ssr,
                    }
                )

    if not rows:
        sys.stderr.write("No rows computed -- check inputs.\n")
        return 1

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
