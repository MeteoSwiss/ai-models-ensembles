"""Compute per-variable persistence MAE per lead time, for the headline figure.

Persistence forecast at init time t and lead h = ERA5(t) (the initial state
broadcast across all leads). MAE(h) is the temporal structure function of
ERA5, averaged over the 112 production init dates spanning 2023 + 2024.

Truth source: the WB2 partitions used by the rest of the verification
pipeline (weatherbench2_2022_2023.zarr + weatherbench2_2024_2025.zarr,
concatenated along time). This matches evaluate_ablation.sh and
evaluate_baselines.sh so persistence is apples-to-apples with the model
verification.

Init sampling: the EXACT 112 init times that the 7-way production grid
uses, derived from the aifsens baseline directory listing
($STORE/baselines/aifsens/<YYYYMMDD_HHMM>). 8 weeks (Jan/Apr/Jul/Oct 2-8
in 2023 + 2024) x 14 inits per week (7 days x {00, 12} UTC).

Writes a JSON keyed by (var, level) -> {lead_h: persistence_MAE} suitable
for direct CRPSS conversion via sigma_clim_1990_2019.json (or, until that
is computed, the existing sigma_clim_ablation.json).

Output: /iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

OUT = Path("/iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.json")

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS = [500, 850]
LEADS_H = list(range(6, 246, 6))  # 6 h .. 240 h, matches paper-headline lead range

WB2_ZARRS = [
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
]


# Production 112-init grid: derived from the aifsens baseline directory.
# Each entry is an ISO timestamp string the WB2 zarr time index understands.
def _build_inits() -> list[str]:
    base = Path("/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifsens")
    out: list[str] = []
    for d in sorted(base.iterdir()):
        name = d.name
        if len(name) != 13 or name[8] != "_":
            continue  # skip non-init dirs (eval, spatial_mean_ssr, etc.)
        yyyy, mm, dd, hh, mn = name[0:4], name[4:6], name[6:8], name[9:11], name[11:13]
        try:
            int(yyyy + mm + dd + hh + mn)
        except ValueError:
            continue
        out.append(f"{yyyy}-{mm}-{dd}T{hh}:{mn}")
    return out


INITS = _build_inits()
print(f"Production grid: {len(INITS)} init times across {INITS[0]} .. {INITS[-1]}")


def cos_lat_mean(da: xr.DataArray) -> float:
    """Cos-latitude weighted spatial mean of the absolute deviation field."""
    w = np.cos(np.deg2rad(da["latitude"]))
    return float((da * w).sum(("latitude", "longitude")) / w.sum() / da.sizes["longitude"])


print("Opening WB2 truth zarrs (concat 2022-2023 + 2024-2025)...")
parts = [xr.open_zarr(z, consolidated=True, chunks={}) for z in WB2_ZARRS]
ds = xr.concat(parts, dim="time").sortby("time")
print(f"Opened. Variables: {len(ds.data_vars)}; init samples: {len(INITS)}; leads: {len(LEADS_H)}")

results: dict = {}

for var in VARS_2D + VARS_3D:
    is_3d = var in VARS_3D
    levels_use = LEVELS if is_3d else [None]
    for lvl in levels_use:
        key = var if lvl is None else f"{var}_{lvl}"
        print(f"  Processing {key}...")

        if var not in ds.data_vars:
            print(f"    SKIP: {var} not in WB2 dataset")
            results[key] = None
            continue
        da_var = ds[var].sel(level=lvl, method="nearest") if is_3d else ds[var]

        per_lead_maes: dict[int, list[float]] = {h: [] for h in LEADS_H}
        for init_str in INITS:
            init_t = np.datetime64(init_str)
            try:
                a = da_var.sel(time=init_t).load()
            except KeyError:
                print(f"    SKIP init {init_str}: not in dataset")
                continue
            for h in LEADS_H:
                valid_t = init_t + np.timedelta64(h, "h")
                try:
                    b = da_var.sel(time=valid_t).load()
                except KeyError:
                    continue
                per_lead_maes[h].append(cos_lat_mean(abs(a - b)))

        results[key] = {h: float(sum(v) / len(v)) if v else None for h, v in per_lead_maes.items()}

OUT.write_text(json.dumps(results, indent=2))
print(f"\n-> {OUT}")
print("\nPersistence MAE at lead 240 h (sanity check):")
for k, lead_maes in results.items():
    v = lead_maes.get(240) if isinstance(lead_maes, dict) else None
    if v is None:
        print(f"  {k:32s} --")
    else:
        print(f"  {k:32s} {v:.4f}")
