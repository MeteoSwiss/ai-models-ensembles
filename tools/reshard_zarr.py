#!/usr/bin/env python3
"""Reshard existing zarr v3 stores to use sharding codec.

Reads each forecast.zarr, rewrites with sharding (many inner chunks packed
into few shard files) and float32 dtype.  The original store is replaced
atomically (write to .tmp, rename).

Usage:
    python tools/reshard_zarr.py /path/to/ablation/phase1
    python tools/reshard_zarr.py /path/to/201801010000
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from zarr.storage import LocalStore

INNER_CHUNKS: dict[str, int] = {
    "ensemble": 1,
    "level": 1,
    "init_time": 1,
    "lead_time": 6,
    "latitude": 90,
    "longitude": 360,
}


def reshard(zarr_path: Path) -> None:
    ds = xr.open_zarr(str(zarr_path), consolidated=True)

    tmp_path = zarr_path.parent / f"{zarr_path.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    store = LocalStore(str(tmp_path))
    root = zarr.open_group(store, mode="w", zarr_format=3)

    for vname in ds.data_vars:
        da = ds[vname]
        shape = da.shape
        inner = tuple(min(s, INNER_CHUNKS.get(d, s)) for d, s in zip(da.dims, shape))
        shards = tuple(math.ceil(s / c) * c for s, c in zip(shape, inner))
        dtype = "float32" if da.dtype == np.float64 else str(da.dtype)

        arr = root.create_array(
            vname,
            shape=shape,
            chunks=inner,
            shards=shards,
            dtype=dtype,
            dimension_names=da.dims,
            attributes=dict(da.attrs),
        )
        arr[:] = da.values.astype(dtype)

    for cname in ds.coords:
        if cname not in ds.data_vars:
            ca = ds.coords[cname]
            root.create_array(
                cname,
                data=ca.values,
                dtype=str(ca.dtype),
                dimension_names=ca.dims if ca.dims else (cname,),
                attributes=dict(ca.attrs),
            )

    zarr.consolidate_metadata(store)
    ds.close()

    # Atomic swap
    old_path = zarr_path.parent / f"{zarr_path.name}.old"
    zarr_path.rename(old_path)
    tmp_path.rename(zarr_path)
    shutil.rmtree(old_path)


def main() -> None:
    root = Path(sys.argv[1])
    zarr_stores = sorted(root.rglob("forecast.zarr"))
    print(f"Found {len(zarr_stores)} forecast.zarr stores under {root}")

    for i, zp in enumerate(zarr_stores):
        # Skip already-sharded stores (check for sharding codec in metadata)
        meta_file = zp / "zarr.json"
        if meta_file.exists():
            import json

            json.loads(meta_file.read_text())  # validate it's valid json
            # Already zarr v3 group; check first data var for sharding
            first_var = next(
                (d for d in zp.iterdir() if d.is_dir() and d.name != ".zmetadata"), None
            )
            if first_var:
                var_meta = first_var / "zarr.json"
                if var_meta.exists():
                    vm = json.loads(var_meta.read_text())
                    codecs = vm.get("codecs", [])
                    if any(c.get("name") == "sharding_indexed" for c in codecs):
                        print(f"[{i + 1}/{len(zarr_stores)}] SKIP (already sharded): {zp}")
                        continue

        print(f"[{i + 1}/{len(zarr_stores)}] Resharding: {zp}", flush=True)
        reshard(zp)
        print(f"[{i + 1}/{len(zarr_stores)}] Done: {zp}", flush=True)


if __name__ == "__main__":
    main()
