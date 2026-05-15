#!/usr/bin/env python3
"""Reshard existing zarr v3 stores to use sharding codec.

Reads each forecast.zarr, rewrites with sharding (many inner chunks packed
into few shard files) and float32 dtype.  The original store is replaced
atomically (write to .tmp, rename).

Usage:
    python tools/reshard_zarr.py /path/to/ablation/phase1
    python tools/reshard_zarr.py /path/to/baselines
    python tools/reshard_zarr.py /path/to/ablation --workers 4
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from zarr.storage import LocalStore

INNER_CHUNKS: dict[str, int] = {
    "ensemble": -1,
    "level": 1,
    "init_time": 1,
    "lead_time": 1,
    "latitude": -1,
    "longitude": -1,
}


def is_already_sharded(zarr_path: Path) -> bool:
    """Check if zarr is already sharded with the correct inner chunk sizes."""
    meta_file = zarr_path / "zarr.json"
    if not meta_file.exists():
        return False

    # Find first data variable (skip coordinate-only arrays)
    first_var = next(
        (d for d in zarr_path.iterdir() if d.is_dir() and d.name != ".zmetadata"), None
    )
    if not first_var:
        return False
    var_meta = first_var / "zarr.json"
    if not var_meta.exists():
        return False
    vm = json.loads(var_meta.read_text())
    codecs = vm.get("codecs", [])
    sharding = next((c for c in codecs if c.get("name") == "sharding_indexed"), None)
    if not sharding:
        return False

    # Verify inner chunk sizes match current policy
    inner_shape = sharding["configuration"]["chunk_shape"]
    dim_names = vm.get("dimension_names", [])
    var_shape = vm.get("shape", [])
    if len(dim_names) != len(inner_shape):
        return False
    for dim, ichunk, full in zip(dim_names, inner_shape, var_shape):
        desired = INNER_CHUNKS.get(dim, -1)
        expected = full if desired == -1 else min(full, desired)
        if ichunk != expected:
            return False
    return True


def reshard(zarr_path: Path) -> None:
    ds = xr.open_zarr(str(zarr_path), consolidated=False)

    tmp_path = zarr_path.parent / f"{zarr_path.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    store = LocalStore(str(tmp_path))
    root = zarr.open_group(store, mode="w", zarr_format=3)

    for vname in ds.data_vars:
        da = ds[vname]
        shape = da.shape
        inner = tuple(
            s if INNER_CHUNKS.get(d, -1) == -1 else min(s, INNER_CHUNKS[d])
            for d, s in zip(da.dims, shape)
        )
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
            values = ca.values
            arr = root.create_array(
                cname,
                shape=values.shape,
                dtype=str(ca.dtype),
                dimension_names=ca.dims if ca.dims else (cname,),
                attributes=dict(ca.attrs),
            )
            arr[:] = values

    zarr.consolidate_metadata(store)
    ds.close()

    # Atomic swap
    old_path = zarr_path.parent / f"{zarr_path.name}.old"
    zarr_path.rename(old_path)
    tmp_path.rename(zarr_path)
    shutil.rmtree(old_path)


def _reshard_one(args: tuple[int, int, Path]) -> str:
    idx, total, zp = args
    tag = f"[{idx + 1}/{total}]"
    if is_already_sharded(zp):
        return f"{tag} SKIP (already sharded): {zp}"
    reshard(zp)
    return f"{tag} Done: {zp}"


def main() -> None:
    root = Path(sys.argv[1])
    workers = 1
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    zarr_stores = sorted(root.rglob("forecast.zarr"))
    total = len(zarr_stores)
    print(f"Found {total} forecast.zarr stores under {root}")

    if workers <= 1:
        for i, zp in enumerate(zarr_stores):
            if is_already_sharded(zp):
                print(f"[{i + 1}/{total}] SKIP (already sharded): {zp}")
                continue
            print(f"[{i + 1}/{total}] Resharding: {zp}", flush=True)
            reshard(zp)
            print(f"[{i + 1}/{total}] Done: {zp}", flush=True)
    else:
        print(f"Using {workers} parallel workers")
        tasks = [(i, total, zp) for i, zp in enumerate(zarr_stores)]
        failed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_reshard_one, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    print(fut.result(), flush=True)
                except Exception as e:
                    failed += 1
                    print(f"[{t[0] + 1}/{t[1]}] FAILED: {t[2]}: {e}", flush=True)
        print(f"Done. {failed}/{total} failed.", flush=True)


if __name__ == "__main__":
    main()
