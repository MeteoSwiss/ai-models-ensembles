"""Spread-error relationship: is the ensemble spread state-dependent?

The paper's SSR summarises spread against error in the aggregate; it cannot say
whether a member spread that is large *for this case, at this point* is followed
by a large error. This streams every baseline's production runs and accumulates,
per (baseline, variable, level, lead), a weighted histogram over the per-pixel
ensemble spread of

    count, sum(spread), sum(squared ensemble-mean error)

on a fine log-spaced spread grid, from which any coarser (e.g. decile) binning of
the spread-error diagram follows without re-reading the forecasts. A calibrated,
genuinely state-dependent ensemble puts the binned RMSE on the 1:1 line against
the binned spread; a spread that carries no case-to-case information gives a flat
RMSE across bins.

Spread is the finite-M-adjusted per-pixel standard deviation
sqrt((M+1)/M * var_m), matching SSR^pix in the paper, so 1:1 is the calibrated
target.

Output: $AIENS_SCRATCH/spread_error_binned.csv
  baseline,variable,level,lead,bin_lo,bin_hi,w_count,w_sum_spread,w_sum_sq_err

Usage (CPU sbatch, ~2 h for all baselines):
  python tools/spread_error_binned.py --baselines all --leads 24 72 120 240
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from _env import IFS_ENS, SCRATCH, STORE, WB2_2022, WB2_2024

sys.stdout.reconfigure(line_buffering=True)

TRUTH_SRC = {2023: WB2_2022, 2024: WB2_2024}
OUT_CSV = SCRATCH / "spread_error_binned.csv"

BASELINES = {
    "aurora_encoder": f"{STORE}/baselines/aurora_encoder",
    "graphcast_all": f"{STORE}/baselines/graphcast_all",
    "sfno_modes10": f"{STORE}/baselines/sfno_modes10",
    "aifs_perturbed": f"{STORE}/baselines/aifs_perturbed",
    "aifsens": f"{STORE}/baselines/aifsens",
    "atlas": f"{STORE}/baselines/atlas",
    "fcn3": f"{STORE}/baselines/fcn3",
    # One consolidated zarr (init_time dim, 50 members), not per-init dirs.
    "ifs_ens": str(IFS_ENS),
}

VARS = [
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
    ("geopotential", 500),
    ("temperature", 850),
    ("u_component_of_wind", 850),
    ("v_component_of_wind", 850),
    ("specific_humidity", 850),
]

INIT_DATES = [
    datetime(y, m, d, h)
    for y in (2023, 2024)
    for m in (1, 4, 7, 10)
    for d in range(2, 9)
    for h in (0, 12)
]

# Stratified 10-member subsample of IFS-ENS, matching evaluate_baselines.sh.
IFS_ENS_MEMBERS = list(range(0, 50, 5))

N_BINS = 240
# Fixed decades of spread around the first field's scale; wide enough that the
# clipped end bins stay empty in practice.
DECADES_BELOW, DECADES_ABOVE = 3.0, 2.0


def cos_lat_weights(lat: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(lat))


def load_truth_field(var: str, level, valid: datetime):
    ds = xr.open_zarr(TRUTH_SRC[valid.year], consolidated=True, decode_timedelta=True)
    da = ds[var]
    if level is not None:
        da = da.sel(level=level)
    da = da.sel(time=np.datetime64(valid, "ns"))
    return da.values, da["latitude"].values, da["longitude"].values


class Accum:
    """Weighted histogram of (spread, squared error) over a log spread grid."""

    def __init__(self, s_ref: float):
        lo = np.log10(max(s_ref, 1e-12)) - DECADES_BELOW
        hi = np.log10(max(s_ref, 1e-12)) + DECADES_ABOVE
        self.edges = np.logspace(lo, hi, N_BINS + 1)
        self.w = np.zeros(N_BINS)
        self.ws = np.zeros(N_BINS)
        self.we = np.zeros(N_BINS)

    def add(self, spread: np.ndarray, err: np.ndarray, wlat: np.ndarray):
        ok = np.isfinite(spread) & np.isfinite(err)
        if not ok.any():
            return
        W = np.broadcast_to(wlat[:, None], spread.shape)
        idx = np.clip(np.digitize(spread[ok], self.edges) - 1, 0, N_BINS - 1)
        w = W[ok]
        self.w += np.bincount(idx, weights=w, minlength=N_BINS)
        self.ws += np.bincount(idx, weights=w * spread[ok], minlength=N_BINS)
        self.we += np.bincount(idx, weights=w * err[ok] ** 2, minlength=N_BINS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["all"])
    ap.add_argument("--leads", nargs="+", type=int, default=[24, 72, 120, 240])
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--limit-inits", type=int, default=None)
    args = ap.parse_args()

    names = sorted(BASELINES) if args.baselines == ["all"] else args.baselines
    inits = INIT_DATES[: args.limit_inits] if args.limit_inits else INIT_DATES
    out = Path(args.out)

    truth_cache: dict = {}

    def get_truth(var, level, valid):
        k = (var, level, valid)
        if k not in truth_cache:
            if len(truth_cache) > 64:
                truth_cache.clear()
            truth_cache[k] = load_truth_field(var, level, valid)
        return truth_cache[k]

    rows = ["baseline,variable,level,lead,bin_lo,bin_hi,w_count,w_sum_spread,w_sum_sq_err"]
    for name in names:
        base = Path(BASELINES[name])
        t0 = time.time()
        acc: dict = {}
        n_done = 0
        consolidated = None
        if name == "ifs_ens":
            consolidated = xr.open_zarr(base, consolidated=True, chunks={})

        for init in inits:
            tag = f"{init:%Y%m%d_%H%M}"
            if consolidated is not None:
                try:
                    fc = consolidated.sel(init_time=np.datetime64(init, "ns"))
                except KeyError:
                    continue
                fc = fc.isel(ensemble=IFS_ENS_MEMBERS)
            else:
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
            for lead in args.leads:
                try:
                    sub = fc.sel(lead_time=np.timedelta64(lead, "h"))
                except Exception:
                    continue
                if "init_time" in sub.dims:
                    sub = sub.isel(init_time=0)
                valid = init + timedelta(hours=lead)
                for var, level in VARS:
                    if var not in sub:
                        continue
                    da = sub[var]
                    if level is not None:
                        da = da.sel(level=level)
                    members = da.values.astype(np.float64)
                    if members.ndim != 3:
                        continue
                    M = members.shape[0]
                    if M < 2:
                        continue
                    o_full, tlat, tlon = get_truth(var, level, valid)
                    if o_full.shape != members.shape[1:]:
                        o = (
                            xr.DataArray(
                                o_full,
                                coords={"latitude": tlat, "longitude": tlon},
                                dims=["latitude", "longitude"],
                            )
                            .sel(latitude=flat, longitude=flon)
                            .values
                        )
                    else:
                        o = o_full
                    if not np.isfinite(members).any():
                        continue
                    mean = np.nanmean(members, axis=0)
                    var_pix = np.nanvar(members, axis=0, ddof=1)
                    spread = np.sqrt((M + 1.0) / M * var_pix)
                    err = mean - o
                    key = (var, level, lead)
                    if key not in acc:
                        ref = np.nanmedian(spread)
                        if not np.isfinite(ref) or ref <= 0:
                            continue
                        acc[key] = Accum(float(ref))
                    acc[key].add(spread, err, wlat)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {name}: {n_done} inits ({time.time()-t0:.0f}s)", flush=True)

        for (var, level, lead), a in sorted(acc.items(), key=lambda kv: str(kv[0])):
            lvl = "" if level is None else level
            for b in range(N_BINS):
                if a.w[b] <= 0:
                    continue
                rows.append(
                    f"{name},{var},{lvl},{lead},{a.edges[b]:.8g},{a.edges[b+1]:.8g},"
                    f"{a.w[b]:.8g},{a.ws[b]:.8g},{a.we[b]:.8g}"
                )
        out.write_text("\n".join(rows) + "\n")
        print(f"{name}: {n_done} inits in {time.time()-t0:.0f}s -> {out}", flush=True)

    print(f"DONE -> {out} ({len(rows)-1} rows)", flush=True)


if __name__ == "__main__":
    main()
