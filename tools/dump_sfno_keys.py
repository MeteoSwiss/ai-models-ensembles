#!/usr/bin/env python3
"""Minimal SFNO key dump: load the cached .tar directly with torch, no earth2studio.

The earth2studio + makani import chain spawns many subprocesses and OOMs
in a 64-200G slurm job. Since the .tar is already cached, we just torch.load
it and dump keys.
"""

from __future__ import annotations

import sys

import torch


def _extract_state_dict(obj):
    """Mirror ai_models_ensembles.e2s_perturbation._extract_state_dict."""
    if not isinstance(obj, dict):
        sd = getattr(obj, "state_dict", None)
        if callable(sd):
            return sd()
        raise RuntimeError(f"Unrecognised checkpoint structure: {type(obj)}")
    for nested in ("model_state", "state_dict", "model_state_dict", "model"):
        if nested in obj and isinstance(obj[nested], dict):
            cand = obj[nested]
            if any(isinstance(v, torch.Tensor) for v in list(cand.values())[:5]):
                return cand
    return obj


def main() -> None:
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
            "/earth2studio_cache/sfno/best_ckpt_mp0.tar"
        )
    )
    print(f"# loading {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)

    float_keys = sorted(k for k, v in state.items() if hasattr(v, "numpy"))
    print(f"# total float tensors: {len(float_keys)}")
    print()
    print("INDEX\tNAME\tSHAPE\tDTYPE")
    for i, k in enumerate(float_keys):
        t = state[k]
        print(f"{i}\t{k}\t{tuple(t.shape)}\t{t.dtype}")

    # Prefix tally at depth 3 and 4
    print()
    print("# === prefix summary (3 dot-segments) ===")
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(float_keys):
        parts = k.split(".")
        prefix = ".".join(parts[:3]) if len(parts) >= 3 else k
        groups.setdefault(prefix, []).append(i)
    for prefix in sorted(groups):
        idxs = groups[prefix]
        print(f"# {prefix:<60} count={len(idxs):>4}  range=[{idxs[0]},{idxs[-1]}]")

    print()
    print("# === prefix summary (4 dot-segments) ===")
    groups4: dict[str, list[int]] = {}
    for i, k in enumerate(float_keys):
        parts = k.split(".")
        prefix = ".".join(parts[:4]) if len(parts) >= 4 else k
        groups4.setdefault(prefix, []).append(i)
    for prefix in sorted(groups4):
        idxs = groups4[prefix]
        print(f"# {prefix:<70} count={len(idxs):>4}  range=[{idxs[0]},{idxs[-1]}]")


if __name__ == "__main__":
    main()
