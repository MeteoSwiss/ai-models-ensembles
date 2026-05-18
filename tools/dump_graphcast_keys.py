#!/usr/bin/env python3
"""Dump GraphCast NPZ float tensor keys in the EXACT order used by
`_perturb_npz` (i.e. `list(f.files)` -- the stored order), plus the
sorted-alphabetical order for comparison.

Writes:
    <out>/graphcast_keys_stored.tsv   (index<TAB>name<TAB>shape<TAB>dtype)
    <out>/graphcast_keys_sorted.tsv   (same, sorted by name)
    <out>/graphcast_keys_summary.txt
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    from ai_models_ensembles.e2s_perturbation import cache_default_checkpoints

    print("Caching graphcast_operational checkpoints...", flush=True)
    cached = cache_default_checkpoints("graphcast_operational")

    npz_path = None
    for rel, local in cached.items():
        if local.lower().endswith(".npz"):
            npz_path = Path(local)
            print(f"NPZ checkpoint: {rel} -> {local}", flush=True)
            break
    if npz_path is None:
        raise RuntimeError(f"No .npz in cached files: {list(cached)}")

    with np.load(npz_path, allow_pickle=False) as f:
        stored_keys = list(f.files)
        entries_stored = [(k, f[k].shape, str(f[k].dtype)) for k in stored_keys]

    # Float-only, mimicking _perturb_array's filter
    stored_float = [(k, s, d) for (k, s, d) in entries_stored if "float" in d.lower()]

    print(f"Total keys in NPZ: {len(entries_stored)}", flush=True)
    print(f"Float tensors: {len(stored_float)}", flush=True)
    print(f"Non-float keys: {len(entries_stored) - len(stored_float)}", flush=True)

    # _perturb_npz uses ALL keys (not filtered to float). It only skips non-float
    # inside `_perturb_array` which returns the array unchanged for non-float.
    # So `_select_indices(len(keys), layer)` is over ALL keys, not just floats.
    # This is critical: the index range refers to position in the full list.
    print(f"\n_perturb_npz total indexable keys (all dtypes): {len(entries_stored)}", flush=True)

    # Write stored-order TSV (full list including non-float)
    with open(out_dir / "graphcast_keys_stored.tsv", "w") as fh:
        fh.write("idx\tname\tshape\tdtype\n")
        for i, (k, s, d) in enumerate(entries_stored):
            fh.write(f"{i}\t{k}\t{tuple(s)}\t{d}\n")

    # Write sorted-order TSV (full list including non-float)
    entries_sorted_all = sorted(entries_stored, key=lambda x: x[0])
    with open(out_dir / "graphcast_keys_sorted.tsv", "w") as fh:
        fh.write("idx\tname\tshape\tdtype\n")
        for i, (k, s, d) in enumerate(entries_sorted_all):
            fh.write(f"{i}\t{k}\t{tuple(s)}\t{d}\n")

    # Summary by top-level prefix
    def top_prefix(name: str) -> str:
        # First dotted segment, or first slash segment
        parts = name.replace("/", ".").split(".")
        return parts[0] if parts else name

    counter_all = Counter(top_prefix(k) for k, _, _ in entries_stored)
    counter_float = Counter(top_prefix(k) for k, _, _ in stored_float)

    with open(out_dir / "graphcast_keys_summary.txt", "w") as fh:
        fh.write(f"NPZ path: {npz_path}\n")
        fh.write(f"Total keys: {len(entries_stored)}\n")
        fh.write(f"Float keys: {len(stored_float)}\n")
        fh.write(f"Non-float keys: {len(entries_stored) - len(stored_float)}\n\n")

        fh.write("Top-level prefix counts (all dtypes):\n")
        for p, n in sorted(counter_all.items()):
            fh.write(f"  {p:40s} {n}\n")
        fh.write("\nTop-level prefix counts (float only):\n")
        for p, n in sorted(counter_float.items()):
            fh.write(f"  {p:40s} {n}\n")

        # Find stored-order index ranges per prefix
        fh.write("\nSTORED ORDER -- contiguous index ranges per top-level prefix:\n")
        cur_prefix = None
        start = 0
        for i, (k, _, _) in enumerate(entries_stored):
            p = top_prefix(k)
            if p != cur_prefix:
                if cur_prefix is not None:
                    fh.write(f"  [{start:4d}, {i:4d})  {cur_prefix}\n")
                cur_prefix = p
                start = i
        fh.write(f"  [{start:4d}, {len(entries_stored):4d})  {cur_prefix}\n")

        # Same for sorted order
        fh.write("\nSORTED ORDER -- contiguous index ranges per top-level prefix:\n")
        cur_prefix = None
        start = 0
        for i, (k, _, _) in enumerate(entries_sorted_all):
            p = top_prefix(k)
            if p != cur_prefix:
                if cur_prefix is not None:
                    fh.write(f"  [{start:4d}, {i:4d})  {cur_prefix}\n")
                cur_prefix = p
                start = i
        fh.write(f"  [{start:4d}, {len(entries_sorted_all):4d})  {cur_prefix}\n")

        # Non-float keys (the unaccounted-for ones)
        fh.write("\nNon-float keys (stored order):\n")
        for i, (k, s, d) in enumerate(entries_stored):
            if "float" not in d.lower():
                fh.write(f"  stored_idx={i}  name={k}  shape={tuple(s)}  dtype={d}\n")

    print(f"\nWrote: {out_dir / 'graphcast_keys_stored.tsv'}", flush=True)
    print(f"Wrote: {out_dir / 'graphcast_keys_sorted.tsv'}", flush=True)
    print(f"Wrote: {out_dir / 'graphcast_keys_summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
