"""Per-init fair CRPS for every baseline, for reviewer-M1 block-bootstrap CIs.

The paper's CRPSS tables reduce over the 112 initialisations before writing, so
no per-init score survives on disk. This recomputes the numerator, one fair
(M-1) CRPS scalar per (baseline, init, variable, level, lead), using the SAME
estimator, cos(lat) weighting, and ERA5 truth as
tools/compute_climatology_crps_vs_eval.py (the CRPSS denominator). Averaging
these per-init values over inits and dividing by the cached climatology CRPS
must reproduce the headline CRPSS table (validation baked into
block_bootstrap_crpss.py); only then is the per-init sample trustworthy for a
paired block bootstrap over initialisations.

Output: /iopsstor/scratch/cscs/sadamov/per_init_crps_production.csv
  columns: baseline, init, variable, level, lead, crps, n_eff_members

Usage (CPU sbatch, ~1-2 h for all baselines):
  python tools/compute_per_init_crps.py --baselines all --leads 72 120 240 360
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
OUT_CSV = Path("/iopsstor/scratch/cscs/sadamov/per_init_crps_production.csv")

# Baseline -> forecast.zarr parent dir (per-init subdirs <YYYYMMDD_HHMM>/forecast.zarr).
BASELINES = {
    "aurora_encoder": f"{STORE}/baselines/aurora_encoder",
    "graphcast_all": f"{STORE}/baselines/graphcast_all",
    "sfno_modes10": f"{STORE}/baselines/sfno_modes10",
    "aifs_perturbed": f"{STORE}/baselines/aifs_perturbed",
    "aifsens": f"{STORE}/baselines/aifsens",
    "atlas": f"{STORE}/baselines/atlas",
    "fcn3": f"{STORE}/baselines/fcn3",
    "ifs_ens": f"{STORE}/baselines/ifs_ens",
}

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

# 112-init production grid (matches compute_climatology_crps_vs_eval.py).
INIT_DATES = [
    datetime(y, m, d, h)
    for y in (2023, 2024)
    for m in (1, 4, 7, 10)
    for d in range(2, 9)
    for h in (0, 12)
]


def cos_lat_weights(lat: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(lat))


def weighted_spatial_mean(field: np.ndarray, wlat: np.ndarray) -> float:
    W = np.broadcast_to(wlat[:, None], field.shape)
    mask = np.isfinite(field)
    den = np.sum(np.where(mask, W, 0.0))
    if den == 0:
        return float("nan")
    num = np.sum(np.where(mask, W * field, 0.0))
    return float(num / den)


def gini_spread_term(ens: np.ndarray) -> np.ndarray:
    """Fair spread 1/(2 M (M-1)) sum_{i,j}|f_i-f_j| per pixel, all-finite members."""
    M = ens.shape[0]
    s = np.sort(ens, axis=0)
    coef = (2.0 * np.arange(1, M + 1) - M - 1.0).astype(np.float64)
    acc = np.tensordot(coef, s, axes=([0], [0]))
    return acc / (M * (M - 1))


def nan_fair_crps_pixel(members: np.ndarray, o: np.ndarray) -> np.ndarray:
    """NaN-aware fair CRPS per pixel for baselines with member gaps (IFS-ENS).

    skill = nanmean_m |f_m - o|; spread = 1/(2 Meff(Meff-1)) sum_{i,j}|f_i-f_j|
    with Meff the per-pixel count of finite members. Pixels with <2 finite
    members return NaN (dropped by the skipna spatial mean).
    """
    finite = np.isfinite(members)
    meff = finite.sum(axis=0)
    skill = np.nanmean(np.abs(members - o[None]), axis=0)
    fill = np.where(finite, members, 0.0)
    diff = np.abs(fill[:, None] - fill[None, :])  # (M, M, lat, lon)
    pair_ok = finite[:, None] & finite[None, :]
    spread_sum = np.where(pair_ok, diff, 0.0).sum(axis=(0, 1))
    denom = meff * (meff - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.where(denom > 0, spread_sum / (2.0 * denom), np.nan)
    crps = skill - spread
    crps = np.where(meff >= 2, crps, np.nan)
    return crps


def load_truth_field(var: str, level, valid: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_zarr(TRUTH_SRC[valid.year], consolidated=True, decode_timedelta=True)
    da = ds[var]
    if level is not None:
        da = da.sel(level=level)
    da = da.sel(time=np.datetime64(valid, "ns"))
    return da.values, da["latitude"].values, da["longitude"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["all"])
    ap.add_argument("--leads", nargs="+", type=int, default=[72, 120, 240, 360])
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--limit-inits", type=int, default=None, help="smoke test: first N inits")
    args = ap.parse_args()

    names = list(BASELINES) if args.baselines == ["all"] else args.baselines
    inits = INIT_DATES[: args.limit_inits] if args.limit_inits else INIT_DATES
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # simple per-run truth cache keyed by (var, level, valid_iso)
    truth_cache: dict[tuple, tuple] = {}

    def get_truth(var, level, valid):
        k = (var, level, valid.isoformat())
        if k not in truth_cache:
            truth_cache[k] = load_truth_field(var, level, valid)
        return truth_cache[k]

    rows = ["baseline,init,variable,level,lead,crps,n_members"]
    for name in names:
        base = Path(BASELINES[name])
        t0 = time.time()
        n_done = 0
        for init in inits:
            tag = f"{init:%Y%m%d_%H%M}"
            zp = base / tag / "forecast.zarr"
            if not zp.is_dir():
                continue
            try:
                fc = xr.open_zarr(zp, consolidated=True, chunks={})
            except Exception as e:
                print(f"  SKIP {name} {tag}: open failed {e}", flush=True)
                continue
            flat = fc["latitude"].values
            flon = fc["longitude"].values
            wlat = cos_lat_weights(flat)
            has_nan_baseline = name == "ifs_ens"
            for lead in args.leads:
                ltd = np.timedelta64(lead, "h")
                try:
                    sub = fc.sel(lead_time=ltd)
                except Exception:
                    continue
                if "init_time" in sub.dims:
                    sub = sub.isel(init_time=0)
                valid = init + timedelta(hours=lead)
                for var, level in VARS:
                    da = sub[var]
                    if level is not None:
                        da = da.sel(level=level)
                    members = da.values  # (M, lat, lon)
                    if members.ndim != 3:
                        continue
                    o_full, tlat, tlon = get_truth(var, level, valid)
                    # align truth to forecast grid (720 vs 721 lat pole row)
                    if o_full.shape != members.shape[1:]:
                        tda = xr.DataArray(
                            o_full,
                            coords={"latitude": tlat, "longitude": tlon},
                            dims=["latitude", "longitude"],
                        ).sel(latitude=flat, longitude=flon)
                        o = tda.values
                    else:
                        o = o_full
                    if has_nan_baseline or not np.isfinite(members).all():
                        crps_pix = nan_fair_crps_pixel(
                            members.astype(np.float64), o.astype(np.float64)
                        )
                        meff = int(np.nanmedian(np.isfinite(members).sum(axis=0)))
                    else:
                        skill = np.abs(members - o[None]).mean(axis=0)
                        crps_pix = skill - gini_spread_term(members.astype(np.float64))
                        meff = members.shape[0]
                    crps = weighted_spatial_mean(crps_pix, wlat)
                    lvl = "" if level is None else level
                    rows.append(f"{name},{init.isoformat()},{var},{lvl},{lead},{crps:.8g},{meff}")
            n_done += 1
        out.write_text("\n".join(rows) + "\n")
        print(f"{name}: {n_done} inits in {time.time()-t0:.0f}s -> {out}", flush=True)

    print(f"DONE -> {out} ({len(rows)-1} rows)", flush=True)


if __name__ == "__main__":
    main()
