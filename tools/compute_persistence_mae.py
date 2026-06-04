"""Compute per-variable persistence MAE per lead time, for the headline figure.

Persistence forecast at init time t and lead h = ERA5(t) (the initial state
broadcast across all leads). MAE(h) is the temporal structure function of
ERA5, averaged over a sample of 2024 init dates.

Writes a JSON keyed by (var, level) -> {lead_h: persistence_MAE} suitable for
direct CRPSS conversion via the existing sigma_clim_*.json files.

Output: /iopsstor/scratch/cscs/sadamov/persistence_mae_2024.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import xarray as xr

OUT = Path("/iopsstor/scratch/cscs/sadamov/persistence_mae_2024.json")

VARS_2D = ["2m_temperature"]  # MSL excluded per the ifs_ens MSL bug
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS = [500, 850]
LEADS_H = list(range(6, 366, 6))  # 6 h ... 360 h

# Sample 20 init dates from the 2024 weekly production grid (Wed 00 UTC).
# Using the local zarr that covers 2022-2025 (same source SwissClim eval uses).
random.seed(42)
ALL_2024_WEDS = [
    f"2024-{m:02d}-{d:02d}"
    for m, d in [
        (1, 3),
        (1, 10),
        (1, 17),
        (1, 24),
        (1, 31),
        (2, 7),
        (2, 14),
        (2, 21),
        (2, 28),
        (3, 6),
        (3, 13),
        (3, 20),
        (3, 27),
        (4, 3),
        (4, 10),
        (4, 17),
        (4, 24),
        (5, 1),
        (5, 8),
        (5, 15),
        (5, 22),
        (5, 29),
        (6, 5),
        (6, 12),
        (6, 19),
        (6, 26),
        (7, 3),
        (7, 10),
        (7, 17),
        (7, 24),
        (7, 31),
        (8, 7),
        (8, 14),
        (8, 21),
        (8, 28),
        (9, 4),
        (9, 11),
        (9, 18),
        (9, 25),
        (10, 2),
        (10, 9),
        (10, 16),
        (10, 23),
        (10, 30),
        (11, 6),
        (11, 13),
        (11, 20),
        (11, 27),
        (12, 4),
        (12, 11),
        (12, 18),
    ]
]
INITS = sorted(random.sample(ALL_2024_WEDS, 20))
INITS = [f"{d}T00" for d in INITS]


def cos_lat_mean(da: xr.DataArray) -> float:
    """Cos-latitude weighted spatial mean of the absolute deviation field."""
    w = np.cos(np.deg2rad(da["latitude"]))
    return float((da * w).sum(("latitude", "longitude")) / w.sum() / da.sizes["longitude"])


# Local ERA5 zarr (2022-2025, same source SwissClim eval uses).
ZARR = "/capstor/store/cscs/swissai/a122/ERA5-2022-2025.zarr"
print(f"Opening {ZARR}...")
ds = xr.open_zarr(ZARR, chunks="auto")
print(f"Opened. Variables: {len(ds.data_vars)}; init samples: {len(INITS)}; leads: {len(LEADS_H)}")

results: dict = {}

for var in VARS_2D + VARS_3D:
    is_3d = var in VARS_3D
    levels_use = LEVELS if is_3d else [None]
    for lvl in levels_use:
        key = var if lvl is None else f"{var}_{lvl}"
        print(f"  Processing {key}...")

        if is_3d:
            da_var = ds[var].sel(level=lvl, method="nearest")
        else:
            # WB2 has 2m_temperature as a top-level variable
            if var not in ds:
                print(f"    SKIP: {var} not in dataset")
                results[key] = None
                continue
            da_var = ds[var]

        # For each init, pull the (init, init+lead) pair and compute |a - b| spatial mean.
        per_lead_maes = {h: [] for h in LEADS_H}
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
                mae_field = abs(a - b)
                per_lead_maes[h].append(cos_lat_mean(mae_field))

        results[key] = {h: float(sum(v) / len(v)) if v else None for h, v in per_lead_maes.items()}

OUT.write_text(json.dumps(results, indent=2))
print(f"\n-> {OUT}")
print("\nPersistence MAE at lead 240 h (rough sanity check):")
for k, lead_maes in results.items():
    v = lead_maes.get(240) if isinstance(lead_maes, dict) else None
    print(f"  {k:32s} {v:.4f}" if v else f"  {k:32s} --")
