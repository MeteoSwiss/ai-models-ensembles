"""Rename the IFS-ENS perturbed-IC zarr variables from MARS shortNames to
SwissClim long names, in place (metadata-only directory renames).

The download (`download_ic_perturbed.py`) writes vars as MARS shortNames
(`z, t, u, v, q, w, 2t, 10u, 10v, 100u, 100v, msl, sp, tcwv`). The Phase 5
inference path reads the IC via `XarrayDataSource.from_swissclim`, which expects
SwissClim long names to map them to the earth2studio lexicon (`2t` would NOT
become `t2m`). Renaming to long names also makes the store consistent with the
rest of the SwissClim pipeline.

This is a LocalStore (zarr v3) directory rename plus a rebuild of the root
consolidated metadata; no array data is copied. Idempotent: a variable already
at its long name is skipped. Run AFTER `fill_interp_levels.py` (which keys off
the MARS shortNames `t,u,v,q,z,w`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import xarray as xr
import zarr

RENAME = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "w": "vertical_velocity",
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "100u": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "sp": "surface_pressure",
    "tcwv": "total_column_water_vapour",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--zarr",
        default="/capstor/store/cscs/swissai/a122/IFS/ifs_analysis_perturbed_ic.zarr",
    )
    args = p.parse_args()
    root = args.zarr

    existing = set(zarr.open_group(root, mode="r").array_keys())

    renamed = []
    for old, new in RENAME.items():
        old_dir, new_dir = os.path.join(root, old), os.path.join(root, new)
        if new in existing:
            continue  # already renamed
        if old not in existing:
            print(f"WARNING: neither '{old}' nor '{new}' present, skipping")
            continue
        os.rename(old_dir, new_dir)
        renamed.append(f"{old} -> {new}")

    if renamed:
        print("renamed:", ", ".join(renamed))
        # Drop the stale consolidated metadata, then rebuild from a fresh scan.
        rj = os.path.join(root, "zarr.json")
        meta = json.load(open(rj))
        meta.pop("consolidated_metadata", None)
        json.dump(meta, open(rj, "w"))
        g = zarr.open_group(root, mode="r+", use_consolidated=False)
        zarr.consolidate_metadata(g.store)
    else:
        print("nothing to rename (already SwissClim names)")

    ds = xr.open_zarr(root, consolidated=True)
    missing = [v for v in RENAME.values() if v not in ds.data_vars]
    if missing:
        print(f"ERROR: expected SwissClim vars missing after rename: {missing}")
        return 1
    print("final data_vars:", sorted(ds.data_vars))
    return 0


if __name__ == "__main__":
    sys.exit(main())
