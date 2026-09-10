"""Zonal energy spectra of every Phase-1 ablation run at one lead time.

The routine SwissClim eval runs on a 24 h lead stride, so the first forecast
step is absent from the ``energy_spectra`` NPZ bundles. This script recomputes
the spectra straight from ``forecast.zarr`` at an arbitrary lead time, using the
same processing as the eval module (mean of per-member spectra, cos-phi latitude
weighting, zero wavenumber dropped) so the numbers are comparable to the
published spectrograms.

Usage (compute node):
    python tools/compute_psd_lead.py --lead 6 --out tools/data/psd_phase1_lead006h.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
from _env import STORE, WB2_2022, WB2_2024  # noqa: E402

from swissclim_evaluations.plots.energy_spectra import calculate_energy_spectra  # noqa: E402

MODELS = ["aurora", "graphcast_operational", "sfno", "aifs"]
INITS = ["20240215", "20230515", "20230815", "20241115"]
VARIABLES = [
    ("geopotential", 500),
    ("temperature", 850),
    ("specific_humidity", 850),
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
    ("10m_u_component_of_wind", None),
]


def var_key(name: str, level: int | None) -> str:
    return name if level is None else f"{name}@{level}hPa"


_TRUTH: dict[str, xr.Dataset] = {}


def truth_for(year: int) -> xr.Dataset:
    """WeatherBench2 ERA5 store covering ``year`` (the two stores split at 2024)."""
    path = WB2_2022 if year < 2024 else WB2_2024
    if path not in _TRUTH:
        _TRUTH[path] = xr.open_zarr(path)
    return _TRUTH[path]


def spectrum(da: xr.DataArray) -> xr.DataArray:
    """Mean of per-member spectra, matching _compute_spectra_pair(reduce_ensemble=True)."""
    return calculate_energy_spectra(
        da, average_dims=["ensemble"] if "ensemble" in da.dims else None
    )


def run_dirs(model: str) -> list[str]:
    root = STORE / "ablation" / "phase1" / model / INITS[0]
    runs = sorted(p for p in os.listdir(root) if p.startswith("mag_"))
    # unperturbed first, then ascending sigma
    return sorted(runs, key=lambda r: float(r.split("_")[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=6, help="lead time in hours")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lead = np.timedelta64(args.lead, "h")
    out: dict[str, np.ndarray] = {}

    for model in args.models:
        runs = run_dirs(model)
        print(f"[{model}] runs: {runs}", flush=True)
        for run in runs:
            per_init: dict[str, list[np.ndarray]] = {}
            tgt_per_init: dict[str, list[np.ndarray]] = {}
            for init in INITS:
                path = STORE / "ablation" / args.phase / model / init / run / "forecast.zarr"
                if not path.exists():
                    print(f"  MISSING {path}", flush=True)
                    continue
                fc = xr.open_zarr(path).sel(lead_time=lead).squeeze("init_time", drop=True)
                valid = np.datetime64(f"{init[:4]}-{init[4:6]}-{init[6:]}T00") + lead
                tr = (
                    truth_for(int(init[:4]))
                    .sel(time=valid)
                    .sel(latitude=fc.latitude, longitude=fc.longitude, method="nearest")
                )
                for name, level in VARIABLES:
                    key = var_key(name, level)
                    da_p = fc[name]
                    da_t = tr[name]
                    if level is not None:
                        da_p = da_p.sel(level=level, drop=True)
                        da_t = da_t.sel(level=level, drop=True)
                    sp = spectrum(da_p.load())
                    st = spectrum(da_t.load())
                    per_init.setdefault(key, []).append(sp.values)
                    tgt_per_init.setdefault(key, []).append(st.values)
                    if "wavenumber" not in out:
                        out["wavenumber"] = sp["wavenumber"].values
                print(f"  {run} {init} done", flush=True)
            for key, arrs in per_init.items():
                out[f"{model}|{run}|{key}"] = np.mean(np.stack(arrs), axis=0)
            for key, arrs in tgt_per_init.items():
                out[f"{model}|__truth__|{key}"] = np.mean(np.stack(arrs), axis=0)

    out["lead_hours"] = np.array([args.lead])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Wrote {args.out} ({len(out)} arrays)")


if __name__ == "__main__":
    main()
