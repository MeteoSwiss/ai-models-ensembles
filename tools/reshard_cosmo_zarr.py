#!/usr/bin/env python3
"""Reshard the COSMO regional training zarr (v2) to zarr v3 with sharding codec.

Reads /capstor/store/cscs/swissai/a122/COSMO/cosmo_ml_data.zarr (zarr v2,
~19 TB across 27 vars) and writes a sibling cosmo_ml_data_v3.zarr with the
same inner-chunk principle as tools/reshard_zarr.py: one slice along the
leading sample axis (time) per inner chunk, full y/x per chunk.

The source is NOT modified or deleted. Verify the output, then move/remove
the source yourself.

Streaming write: 3D vars (~2.3 TB) are written one time-slab at a time so
peak RAM stays bounded by --slab-time.

Usage:
    python tools/reshard_cosmo_zarr.py
    python tools/reshard_cosmo_zarr.py --src <path> --dst <path>
    python tools/reshard_cosmo_zarr.py --slab-time 256 --only T,QV
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from zarr.storage import LocalStore

DEFAULT_SRC = "/capstor/store/cscs/swissai/a122/COSMO/cosmo_ml_data.zarr"
DEFAULT_DST = "/capstor/store/cscs/swissai/a122/COSMO/cosmo_ml_data_sharded.zarr"

INNER_CHUNKS: dict[str, int] = {
    "time": 1,
    "z": 1,
    "y": -1,
    "x": -1,
}

SKIP_GROUPS = {"shps"}


def _inner_for(dims: tuple[str, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        s if INNER_CHUNKS.get(d, -1) == -1 else min(s, INNER_CHUNKS[d]) for d, s in zip(dims, shape)
    )


def _shards_for(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    inner: tuple[int, ...],
    shard_time: int,
) -> tuple[int, ...]:
    """Pack many inner chunks into one shard.

    For axes named in `SHARD_BLOCKS`, the shard length is that block, capped
    by the full axis length and rounded up to a multiple of the inner chunk.
    For other axes (lat/lon-like), the shard covers the full axis. Result:
    one shard per (time block, z block) tile.
    """
    SHARD_BLOCKS = {"time": shard_time}
    out = []
    for d, s, c in zip(dims, shape, inner):
        if d in SHARD_BLOCKS:
            block = min(s, SHARD_BLOCKS[d])
            out.append(math.ceil(block / c) * c)
        else:
            out.append(math.ceil(s / c) * c)
    return tuple(out)


def _write_streamed(arr, da: xr.DataArray, slab_time: int) -> None:
    """Write da into arr in time-slabs to bound peak RAM."""
    dtype = arr.dtype
    if "time" not in da.dims:
        arr[:] = da.values.astype(dtype, copy=False)
        return
    t_axis = da.dims.index("time")
    n_t = da.shape[t_axis]
    for start in range(0, n_t, slab_time):
        end = min(start + slab_time, n_t)
        sl = [slice(None)] * da.ndim
        sl[t_axis] = slice(start, end)
        chunk = da.isel({da.dims[t_axis]: slice(start, end)}).values
        arr[tuple(sl)] = chunk.astype(dtype, copy=False)
        print(
            f"      time[{start:>6d}:{end:>6d}] / {n_t} written",
            flush=True,
        )


def reshard(
    src: Path,
    dst: Path,
    slab_time: int,
    shard_time: int,
    only: set[str] | None,
) -> None:
    if dst.exists():
        print(f"Destination already exists: {dst}", file=sys.stderr)
        print("Refusing to overwrite. Remove it first or pass --dst.", file=sys.stderr)
        sys.exit(2)

    tmp = dst.parent / f"{dst.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    ds = xr.open_zarr(str(src), consolidated=False)
    store = LocalStore(str(tmp))
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.attrs.update(dict(ds.attrs))

    # Coordinates first (cast lat/lon float64 -> float32 for parity with existing tool).
    for cname in ds.coords:
        if cname in SKIP_GROUPS:
            continue
        ca = ds.coords[cname]
        values = ca.values
        dtype = "float32" if values.dtype == np.float64 else str(values.dtype)
        a = root.create_array(
            cname,
            shape=values.shape,
            dtype=dtype,
            dimension_names=ca.dims if ca.dims else (cname,),
            attributes=dict(ca.attrs),
        )
        a[:] = values.astype(dtype, copy=False)
        print(f"coord {cname}: shape={values.shape} dtype={dtype}", flush=True)

    for vname in ds.data_vars:
        if vname in SKIP_GROUPS:
            continue
        if only is not None and vname not in only:
            continue
        da = ds[vname]
        shape = tuple(da.shape)
        inner = _inner_for(da.dims, shape)
        shards = _shards_for(da.dims, shape, inner, shard_time)
        dtype = "float32" if da.dtype == np.float64 else str(da.dtype)

        size_gb = math.prod(shape) * np.dtype(dtype).itemsize / 1024**3
        print(
            f"var {vname}: dims={da.dims} shape={shape} dtype={dtype} "
            f"inner={inner} shard={shards} size={size_gb:.2f} GB",
            flush=True,
        )

        attrs = dict(da.attrs)
        coords_attr = da.encoding.get("coordinates")
        if coords_attr:
            attrs["coordinates"] = coords_attr

        arr = root.create_array(
            vname,
            shape=shape,
            chunks=inner,
            shards=shards,
            dtype=dtype,
            dimension_names=da.dims,
            attributes=attrs,
        )
        _write_streamed(arr, da, slab_time)

    zarr.consolidate_metadata(store)
    ds.close()
    tmp.rename(dst)
    print(f"\nDone. Output: {dst}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--dst", default=DEFAULT_DST)
    p.add_argument(
        "--slab-time",
        type=int,
        default=128,
        help="Time-slab size for streaming writes. 128 -> 3D 3 GB slab, 2D 0.1 GB.",
    )
    p.add_argument(
        "--shard-time",
        type=int,
        default=100,
        help=(
            "Time block per shard (in inner-chunks of time). Default 100 -> "
            "4D shard ~5.4 GB (100 x 60 z x 390 y x 582 x x 4 B), 3D shard "
            "~22 MB (100 x 390 x 582 x 4 B). Use --shard-time 1000 for "
            "fewer, larger shards (~54 GB / ~220 MB)."
        ),
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of data_vars to reshard (debug aid).",
    )
    args = p.parse_args()
    only = set(args.only.split(",")) if args.only else None
    reshard(Path(args.src), Path(args.dst), args.slab_time, args.shard_time, only)


if __name__ == "__main__":
    main()
