#!/usr/bin/env python3
"""Inspect checkpoint tensors for earth2studio models.

Prints the number, names, shapes, and dtypes of weight tensors in each
model's checkpoint -- exactly the tensors that --layer indexing addresses.

Usage (inside a model container):
    python tools/inspect_weights.py aurora
    python tools/inspect_weights.py graphcast_operational
    python tools/inspect_weights.py sfno
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def inspect_npz(path: Path) -> list[tuple[str, tuple, str]]:
    with np.load(path, allow_pickle=False) as f:
        return [(k, f[k].shape, str(f[k].dtype)) for k in f.files]


def inspect_torch(path: Path) -> list[tuple[str, tuple, str]]:
    import torch

    from ai_models_ensembles.e2s_perturbation import _extract_state_dict

    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)
    entries = []
    for k in sorted(state.keys()):
        t = state[k]
        if hasattr(t, "shape"):
            entries.append((k, tuple(t.shape), str(t.dtype)))
        else:
            entries.append((k, (), str(type(t))))
    return entries


def inspect_safetensors(path: Path) -> list[tuple[str, tuple, str]]:
    from safetensors.numpy import load_file

    tensors = load_file(str(path))
    return [(k, tensors[k].shape, str(tensors[k].dtype)) for k in sorted(tensors)]


_INSPECTORS = {
    ".npz": inspect_npz,
    ".pt": inspect_torch,
    ".pth": inspect_torch,
    ".ckpt": inspect_torch,
    ".tar": inspect_torch,
    ".safetensors": inspect_safetensors,
}


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "aurora"

    from ai_models_ensembles.e2s_perturbation import cache_default_checkpoints

    print(f"Caching checkpoints for {model_name}...")
    cached = cache_default_checkpoints(model_name)

    total_tensors = 0
    total_float = 0

    for rel, local_path in cached.items():
        p = Path(local_path)
        suffix = p.suffix.lower()
        if suffix not in _INSPECTORS:
            print(f"\n--- {rel} (non-checkpoint, skipped) ---")
            continue

        print(f"\n{'='*80}")
        print(f"File: {rel}")
        print(f"Local: {local_path}")
        print(f"Format: {suffix}")
        print(f"{'='*80}")

        entries = _INSPECTORS[suffix](p)
        n_float = sum(1 for _, _, dt in entries if "float" in dt.lower())

        print(f"Total keys: {len(entries)}")
        print(f"Float tensors (perturbable): {n_float}")
        print()

        # Group by layer name prefix (first 2 dot-segments)
        groups: dict[str, list] = {}
        for name, shape, dtype in entries:
            parts = name.split(".")
            prefix = ".".join(parts[:2]) if len(parts) > 2 else name
            groups.setdefault(prefix, []).append((name, shape, dtype))

        # Print summary by group
        print(f"{'Group':<50} {'Count':>6}  {'Float':>6}  Example shape")
        print("-" * 100)
        for prefix in sorted(groups):
            items = groups[prefix]
            n_grp_float = sum(1 for _, _, dt in items if "float" in dt.lower())
            example = items[0]
            print(f"{prefix:<50} {len(items):>6}  {n_grp_float:>6}  {example[1]}")

        total_tensors += len(entries)
        total_float += n_float

    print(f"\n{'='*80}")
    print(f"SUMMARY for {model_name}:")
    print(f"  Total tensor keys:          {total_tensors}")
    print(f"  Float tensors (perturbable): {total_float}")
    print(f"  --layer range:              0 to {total_float - 1}")
    print(f"  --layer 'early' (0.0:0.33): indices 0..{int(total_float * 0.33) - 1}")
    print(
        f"  --layer 'middle' (0.33:0.67): indices {int(total_float * 0.33)}..{int(total_float * 0.67) - 1}"
    )
    print(f"  --layer 'late' (0.67:1.0):  indices {int(total_float * 0.67)}..{total_float - 1}")


if __name__ == "__main__":
    main()
