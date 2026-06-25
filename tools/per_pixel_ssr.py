"""Per-pixel SSR diagnostic (Fortin), field-averaged per lead.

Faithful re-implementation of the per-pixel Spread-Skill Ratio that SwissClim's
probabilistic module reports in ssr_line_<var>_by_lead_ensprob.csv, so the
on-disk caches can be refreshed to the current Fortin vintage on the full
112-init grid without re-running the whole (CRPS + maps + 170 MB npz)
probabilistic eval.

Recipe verified against SwissClim_Evaluations
(src/swissclim_evaluations/metrics/probabilistic/{calc,wbx}.py):

  per pixel (lat, lon), per (variable, level, lead), reduce over init:
      FortinVar = mean_init[ (M+1)/M * var_n(member_n) ]   # var_n is ddof=1
      PlainMSE  = mean_init[ ( mean_n(member_n) - truth )^2 ]
      SSR_pixel = sqrt( FortinVar / PlainMSE )

  by_lead line (the CSV value) = cos(lat)-weighted field mean of SSR_pixel
      ( SwissClim uses scores.create_latitude_weights, which equals cos(lat)
        on this 0.25deg grid; forecast and WB2 truth share an identical grid ).

This differs from tools/spatial_mean_ssr.py, which collapses the field to a
scalar FIRST and forms spread/error of that scalar. Here spread/error are
per-pixel, SSR is formed per-pixel, and only THEN is the field averaged.

Usage:

    python tools/per_pixel_ssr.py \\
        --forecast-zarrs <path1> [<path2> ...] \\
        --truth-zarrs <wb2a.zarr> <wb2b.zarr> \\
        --variables 2m_temperature mean_sea_level_pressure geopotential ... \\
        --levels 500 850 \\
        --leads 24 72 120 240 \\
        --model-label aifs_perturbed_ic \\
        --out-csv per_pixel_ssr.csv

CSV columns: model, variable, level, lead_time_hours, n_inits, n_members, ssr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _open_zarr(path: str) -> xr.Dataset:
    try:
        return xr.open_zarr(path, consolidated=True, chunks={})
    except (KeyError, ValueError):
        return xr.open_zarr(path, consolidated=False, chunks={})


def _select_lead(da: xr.DataArray, lead_hours: int) -> xr.DataArray | None:
    if "lead_time" not in da.dims:
        return da
    lt = da["lead_time"].values
    if np.issubdtype(lt.dtype, np.timedelta64):
        hours = (lt / np.timedelta64(1, "h")).astype(int)
    else:
        hours = lt.astype(int)
    if lead_hours not in hours:
        return None
    idx = int(np.where(hours == lead_hours)[0][0])
    return da.isel(lead_time=idx)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast-zarrs", nargs="+", required=True)
    p.add_argument("--truth-zarrs", nargs="+", required=True)
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--levels", nargs="+", type=float, default=[None])
    p.add_argument("--leads", nargs="+", type=int, required=True)
    p.add_argument("--model-label", required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    truth_parts = [xr.open_zarr(tz, consolidated=True, chunks={}) for tz in args.truth_zarrs]
    truth_ds = xr.concat(truth_parts, dim="time") if len(truth_parts) > 1 else truth_parts[0]
    truth_ds = truth_ds.sortby("time")

    lat = None
    # (variable, level, lead) -> dict with per-pixel accumulators
    acc: dict[tuple, dict] = {}

    for fz in args.forecast_zarrs:
        fz_ds = _open_zarr(fz)
        n_inits_in_zarr = fz_ds.sizes["init_time"] if "init_time" in fz_ds.dims else 1
        for init_idx in range(n_inits_in_zarr):
            sub = fz_ds.isel(init_time=init_idx) if "init_time" in fz_ds.dims else fz_ds
            init_time = fz_ds["init_time"].values[init_idx] if "init_time" in fz_ds.dims else None
            for var in args.variables:
                if var not in sub.data_vars:
                    continue
                for lvl in args.levels:
                    for lead in args.leads:
                        fa = sub[var]
                        if lvl is not None and "level" in fa.dims:
                            fa = fa.sel(level=lvl)
                        fa = _select_lead(fa, lead)
                        if fa is None:
                            continue
                        if "init_time" in fa.dims:
                            fa = fa.isel(init_time=0)

                        ta = truth_ds[var]
                        if lvl is not None and "level" in ta.dims:
                            ta = ta.sel(level=lvl)
                        if init_time is not None and "time" in ta.dims:
                            valid_time = np.datetime64(init_time) + np.timedelta64(lead, "h")
                            try:
                                ta = ta.sel(time=valid_time)
                            except KeyError:
                                continue

                        # Align truth to the forecast grid: some backbones
                        # (e.g. Aurora) drop a pole row -> 720 vs 721 latitudes,
                        # so naive pixelwise differencing fails to broadcast.
                        if (
                            "latitude" in ta.dims
                            and "latitude" in fa.dims
                            and ta.sizes["latitude"] != fa.sizes["latitude"]
                        ):
                            ta = ta.sel(latitude=fa["latitude"].values, method="nearest")
                        if (
                            "longitude" in ta.dims
                            and "longitude" in fa.dims
                            and ta.sizes["longitude"] != fa.sizes["longitude"]
                        ):
                            ta = ta.sel(longitude=fa["longitude"].values, method="nearest")

                        fa_v = np.asarray(fa.values, dtype=np.float64)  # (ens, lat, lon)
                        ta_v = np.asarray(ta.values, dtype=np.float64)  # (lat, lon)
                        if fa_v.ndim != 3:
                            continue
                        M = fa_v.shape[0]
                        if M < 2:
                            continue

                        var_ens = np.nanvar(fa_v, axis=0, ddof=1)  # (lat, lon)
                        fortin = var_ens * (M + 1) / M
                        ens_mean = np.nanmean(fa_v, axis=0)
                        mse = (ens_mean - ta_v) ** 2

                        key = (var, lvl, lead)
                        if key not in acc:
                            acc[key] = {
                                "fortin_sum": np.zeros_like(fortin),
                                "mse_sum": np.zeros_like(mse),
                                "n_inits": 0,
                                "n_members": M,
                            }
                            if lat is None:
                                lat = np.asarray(fa["latitude"].values)
                        a = acc[key]
                        a["fortin_sum"] += fortin
                        a["mse_sum"] += mse
                        a["n_inits"] += 1
        fz_ds.close()

    if not acc:
        sys.stderr.write("No rows computed -- check inputs.\n")
        return 1

    weights = np.cos(np.deg2rad(lat))  # == scores.create_latitude_weights on this grid
    w2d = None

    rows: list[dict] = []
    for (var, lvl, lead), a in acc.items():
        fortin_mean = a["fortin_sum"] / a["n_inits"]
        mse_mean = a["mse_sum"] / a["n_inits"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ssr_pixel = np.sqrt(fortin_mean / mse_mean)  # (lat, lon)
        if w2d is None or w2d.shape != ssr_pixel.shape:
            w2d = np.broadcast_to(weights[:, None], ssr_pixel.shape)
        finite = np.isfinite(ssr_pixel)
        wsum = np.sum(w2d[finite])
        ssr_line = (
            float(np.sum(ssr_pixel[finite] * w2d[finite]) / wsum) if wsum > 0 else float("nan")
        )
        rows.append(
            {
                "model": args.model_label,
                "variable": var,
                "level": lvl if lvl is not None else float("nan"),
                "lead_time_hours": lead,
                "n_inits": a["n_inits"],
                "n_members": a["n_members"],
                "ssr": ssr_line,
            }
        )

    df = pd.DataFrame(rows).sort_values(["variable", "level", "lead_time_hours"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
