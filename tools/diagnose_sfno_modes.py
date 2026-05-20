"""Diagnose SFNO spectral conv weight layout.

Loads the default SFNO model inside the SFNO container and inspects:
- spectral conv layer attributes (operator_type, modes_lat, modes_lon)
- forward/inverse SHT lmax / mmax
- weight tensor shape post-load (checks earth2studio's unsqueeze(0))
- whether the 240 axis is l (dhconv) or something else

Run via tools/submit_diagnose_sfno_modes.sh in the SFNO container.
"""

from __future__ import annotations

import torch

from ai_models_ensembles.e2s_models import load_model


def main() -> None:
    print("=" * 70)
    print("Loading SFNO model via load_model('sfno') ...")
    model, _ = load_model("sfno")
    print(f"Top-level model type: {type(model).__name__}")
    print()

    # Walk named_modules to find a SpectralConv-like layer (has modes_lat attr)
    print("Searching named_modules for a layer with `modes_lat`...")
    spec = None
    spec_name = None
    for name, mod in model.named_modules():
        if hasattr(mod, "modes_lat") and hasattr(mod, "weight"):
            spec = mod
            spec_name = name
            print(f"  found: {name}  ({type(mod).__name__})")
            break

    if spec is None:
        print("  no module with modes_lat found. Dumping top-level tree:")
        for name, mod in list(model.named_modules())[:30]:
            print(f"    {name}: {type(mod).__name__}")
        return
    print(f"Spectral conv class: {type(spec).__name__} at attribute path '{spec_name}'")
    print()

    # Inspect key attributes
    keys_of_interest = [
        "operator_type",
        "modes_lat",
        "modes_lon",
        "modes_lat_local",
        "modes_lon_local",
        "num_groups",
        "in_channels",
        "out_channels",
        "separable",
    ]
    print("SpectralConv attributes:")
    for k in keys_of_interest:
        v = getattr(spec, k, "MISSING")
        print(f"  {k}: {v}")
    print()

    # Weight shape
    if hasattr(spec, "weight"):
        w = spec.weight
        print("weight tensor:")
        print(f"  shape: {tuple(w.shape)}")
        print(f"  dtype: {w.dtype}")
        print(f"  ndim: {w.ndim}")
    else:
        print("WARNING: spec has no .weight")
    print()

    # SHT transforms
    for tf_attr in ("forward_transform", "inverse_transform"):
        if hasattr(spec, tf_attr):
            tf = getattr(spec, tf_attr)
            print(f"{tf_attr}: {type(tf).__name__}")
            for k in ("lmax", "mmax", "nlat", "nlon"):
                v = getattr(tf, k, "MISSING")
                print(f"  {tf_attr}.{k}: {v}")
            print()

    # Hard interpretation check: is 240 = modes_lat?
    modes_lat = getattr(spec, "modes_lat", None)
    modes_lon = getattr(spec, "modes_lon", None)
    last_axis = w.shape[-1] if hasattr(spec, "weight") else None
    print("=" * 70)
    print("INTERPRETATION:")
    if modes_lat == last_axis:
        print(f"  CONFIRMED: weight last axis ({last_axis}) == modes_lat ({modes_lat})")
        print(f"  -> Axis is total wavenumber l (degree), running 0..{last_axis - 1}")
        print("  -> Phase 3 L_cut=10 perturbs l in {0, 1, ..., 9} (wavelengths >= 4000 km)")
    elif modes_lat and modes_lon and last_axis == modes_lat * modes_lon:
        print(f"  Last axis ({last_axis}) == modes_lat * modes_lon ({modes_lat} * {modes_lon})")
        print("  -> Flattened (l, m) layout; need to know ordering to map L_cut.")
    else:
        print(f"  UNEXPECTED: modes_lat={modes_lat}, modes_lon={modes_lon}, last_axis={last_axis}")
    print()

    # Sample weight values at low vs high l to sanity-check
    if hasattr(spec, "weight") and w.ndim >= 3:
        # weight is (num_groups, in_ch_per_g, out_ch_per_g, modes_lat) after earth2studio unsqueeze
        # or similar; take a slice along the last axis
        slice_low = (
            w[..., :3].abs().mean().item()
            if torch.is_complex(w)
            else w[..., :3].abs().mean().item()
        )
        slice_high = (
            w[..., -3:].abs().mean().item()
            if torch.is_complex(w)
            else w[..., -3:].abs().mean().item()
        )
        slice_mid = w[..., 100:103].abs().mean().item() if w.shape[-1] >= 103 else None
        print("Weight magnitude diagnostics (mean of |w|):")
        print(f"  first 3 entries (low-l):   {slice_low:.6e}")
        if slice_mid is not None:
            print(f"  middle 3 entries (l~100):  {slice_mid:.6e}")
        print(f"  last 3 entries (high-l):   {slice_high:.6e}")
        print()
        print("  (For a SHT-conv weight, low-l entries typically have larger")
        print("   magnitudes than high-l: low modes carry more variance in")
        print("   physical fields. This is a soft sanity check that the")
        print("   ordering is low-l first.)")


if __name__ == "__main__":
    main()
