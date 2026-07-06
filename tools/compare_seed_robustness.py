"""Seed-robustness delta for the four production picks (reviewer M3/M5).

Compares CRPSS@240h (4-init ablation-grid mean) of each production-pick cell at
seed=42 (existing ablation zarrs) vs seed=43 (ablation_seed43, submitted by
scripts/submit_seed_robustness.sh). A delta within the +-0.02 ablation-grid
spread supports the "cells within +-0.02 are tied" selection statement.

Reuses the exact fair-CRPS / cos-lat helpers of compute_per_init_crps.py and the
cached ablation-grid climatology denominator.

Usage:  python tools/compare_seed_robustness.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_per_init_crps import (  # noqa: E402
    cos_lat_weights,
    weighted_spatial_mean,
    gini_spread_term,
    load_truth_field,
)

STORE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
CLIM = Path(__file__).resolve().parent / "data" / "crps_clim_eval_ablation_1990_2019.json"
LEAD = 240
INIT_TAGS = ["20230515", "20230815", "20240215", "20241115"]

# pick -> (model_id, seed42 phase dir, run_tag)
PICKS = {
    "aurora_encoder": ("aurora", "phase2b", "mag_0.025_layer_encoder"),
    "graphcast_all": ("graphcast_operational", "phase1", "mag_0.01_layer_all"),
    "sfno_modes10": ("sfno", "phase3", "mag_0.25_modes10"),
    "aifs_perturbed": ("aifs", "phase2", "mag_0.027500_layer_decoder"),
}
PAPER_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]


def init_dt(tag: str) -> datetime:
    return datetime(int(tag[:4]), int(tag[4:6]), int(tag[6:8]), 0)


def crps_field(zarr_path: Path, var: str, level, valid: datetime) -> float:
    try:
        fc = xr.open_zarr(zarr_path, consolidated=True, chunks={})
    except Exception:
        fc = xr.open_zarr(zarr_path, consolidated=False, chunks={})
    sub = fc.sel(lead_time=np.timedelta64(LEAD, "h"))
    if "init_time" in sub.dims:
        sub = sub.isel(init_time=0)
    da = sub[var]
    if level is not None:
        da = da.sel(level=level)
    members = da.values.astype(np.float64)
    flat, flon = fc["latitude"].values, fc["longitude"].values
    o_full, tlat, tlon = load_truth_field(var, level, valid)
    if o_full.shape != members.shape[1:]:
        o = (
            xr.DataArray(
                o_full, coords={"latitude": tlat, "longitude": tlon}, dims=["latitude", "longitude"]
            )
            .sel(latitude=flat, longitude=flon)
            .values
        )
    else:
        o = o_full
    skill = np.abs(members - o[None]).mean(axis=0)
    crps_pix = skill - gini_spread_term(members)
    return weighted_spatial_mean(crps_pix, cos_lat_weights(flat))


def crpss_for_root(model_id: str, run_tag: str, root: Path, phase: str | None) -> float | None:
    """4-init-mean CRPSS@240h from a set of forecast.zarr under root."""
    clim = json.loads(CLIM.read_text())
    vscores = []
    for var in PAPER_VARS:
        levels = [None] if var in ("2m_temperature", "mean_sea_level_pressure") else [500, 850]
        lvl_crpss = []
        for level in levels:
            per_init = []
            for tag in INIT_TAGS:
                if phase is not None:
                    zp = Path(f"{STORE}/ablation/{phase}/{model_id}/{tag}/{run_tag}/forecast.zarr")
                else:
                    zp = root / model_id / tag / run_tag / "forecast.zarr"
                if not zp.is_dir():
                    return None
                per_init.append(crps_field(zp, var, level, init_dt(tag) + timedelta(hours=LEAD)))
            num = float(np.nanmean(per_init))
            key = var if level is None else f"{var}_{level}"
            den = float(clim[key][str(LEAD)])
            lvl_crpss.append(1.0 - num / den)
        vscores.append(np.mean(lvl_crpss))
    return float(np.mean(vscores))


def main():
    seed43_root = Path(f"{STORE}/ablation_seed43")
    print(f"{'pick':16} {'seed42':>8} {'seed43':>8} {'delta':>8}  within +-0.02?")
    for pick, (model_id, phase, run_tag) in PICKS.items():
        s42 = crpss_for_root(model_id, run_tag, Path(f"{STORE}/ablation"), phase)
        s43 = crpss_for_root(model_id, run_tag, seed43_root, None)
        if s42 is None or s43 is None:
            print(f"{pick:16} {'(pending)' if s43 is None else s43:>8}  seed43 not ready")
            continue
        d = s43 - s42
        print(f"{pick:16} {s42:8.3f} {s43:8.3f} {d:+8.3f}  {'yes' if abs(d) <= 0.02 else 'NO'}")


if __name__ == "__main__":
    main()
