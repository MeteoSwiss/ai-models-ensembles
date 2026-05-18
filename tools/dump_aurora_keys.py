"""Dump full sorted list of float-tensor keys for the Aurora checkpoint.

Mirrors how `_perturb_torch` enumerates tensors: state = _extract_state_dict(obj),
then `sorted(k for k, v in state.items() if hasattr(v, 'numpy'))`. Prints one
line per float tensor: index<TAB>name<TAB>shape<TAB>dtype, plus per-top-level
counts and the alphabetical order of top-level prefixes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ai_models_ensembles.e2s_perturbation import (
    _extract_state_dict,
    cache_default_checkpoints,
)


def main() -> None:
    cached = cache_default_checkpoints("aurora")
    # Expect a single .ckpt
    rel, local = next(iter(cached.items()))
    print(f"# file: {rel}")
    print(f"# local: {local}")
    p = Path(local)

    obj = torch.load(p, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)

    # Match _perturb_torch's filter exactly: keys whose values look like tensors.
    all_keys_sorted = sorted(state.keys())
    float_keys = sorted(k for k, v in state.items() if hasattr(v, "numpy"))

    print(f"# total state keys: {len(all_keys_sorted)}")
    print(f"# float (perturbable) keys: {len(float_keys)}")

    # Also confirm non-float keys, if any (for completeness)
    nonfloat = [k for k in all_keys_sorted if not hasattr(state[k], "numpy")]
    print(f"# non-tensor keys: {len(nonfloat)}")
    for k in nonfloat:
        print(f"# nonfloat: {k!r} -> {type(state[k]).__name__}")

    # Dump every float tensor's full record.
    print("# idx\tname\tshape\tdtype")
    for i, k in enumerate(float_keys):
        t = state[k]
        print(f"{i}\t{k}\t{tuple(t.shape)}\t{t.dtype}")

    # Top-level prefix tally (first dot segment) in alphabetical order of names.
    from collections import Counter, OrderedDict

    first_seen: OrderedDict[str, int] = OrderedDict()
    last_seen: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for i, k in enumerate(float_keys):
        top = k.split(".", 1)[0]
        counts[top] += 1
        if top not in first_seen:
            first_seen[top] = i
        last_seen[top] = i

    print("# top-level prefix tally (in first-occurrence / alphabetical order):")
    for top, first in first_seen.items():
        print(f"#   {top}: count={counts[top]} first_idx={first} last_idx={last_seen[top]}")


if __name__ == "__main__":
    main()
