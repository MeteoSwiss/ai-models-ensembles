"""Band-integrated LSD against lead time for an ablation run, from forecast.zarr.

Fills gaps in the SwissClim intercomparison CSVs: the AIFS Phase-1 unperturbed
run (mag_0_layer_all) has forecast data but was never evaluated, so its row is
missing from lsd_metrics_banded_lead_time_combined.csv.

Reproduces the eval's reduction order exactly - mean of per-member spectra, then
mean over inits, then LSD per band - and defaults to the eval's 24 h lead stride.
Use --validate on an already-evaluated run to check the numbers against the CSV.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
from _env import STORE  # noqa: E402

from swissclim_evaluations.plots.energy_spectra import (  # noqa: E402
    _compute_banded_lsd_da,
    calculate_energy_spectra,
)

INITS = ["20240215", "20230515", "20230815", "20241115"]
MODELS = ["aurora", "graphcast_operational", "sfno", "aifs"]


def parse_spec(spec: str) -> tuple[str, int | None]:
    """ "geopotential@500" -> ("geopotential", 500); "2m_temperature" -> (name, None)."""
    if "@" in spec:
        name, level = spec.split("@", 1)
        return name, int(level)
    return spec, None


def truth_for(year: int) -> xr.Dataset:
    from _env import WB2_2022, WB2_2024

    return xr.open_zarr(WB2_2022 if year < 2024 else WB2_2024)


def banded_lsd(model: str, run: str, variable: str, level: int | None, phase: str, stride: int):
    spec_p, spec_t = [], []
    for init in INITS:
        path = STORE / "ablation" / phase / model / init / run / "forecast.zarr"
        fc = xr.open_zarr(path).squeeze("init_time", drop=True)
        fc = fc.isel(lead_time=slice(None, None, stride // 6))
        valid = np.datetime64(f"{init[:4]}-{init[4:6]}-{init[6:]}T00") + fc["lead_time"].values
        tr = (
            truth_for(int(init[:4]))
            .sel(time=valid)
            .sel(latitude=fc.latitude, longitude=fc.longitude, method="nearest")
            .rename({"time": "lead_time"})
            .assign_coords(lead_time=fc["lead_time"])
        )
        da_p, da_t = fc[variable], tr[variable]
        if level is not None:
            da_p = da_p.sel(level=level, drop=True)
            da_t = da_t.sel(level=level, drop=True)
        avg = ["ensemble"] if "ensemble" in da_p.dims else None
        spec_p.append(calculate_energy_spectra(da_p.load(), average_dims=avg))
        spec_t.append(calculate_energy_spectra(da_t.load(), average_dims=None))
        print(f"  {run} {init} done", flush=True)

    sp = xr.concat(spec_p, dim="init").mean(dim="init")
    st = xr.concat(spec_t, dim="init").mean(dim="init")
    banded_lsd.spectra = (st, sp)
    lsd = _compute_banded_lsd_da(st, sp)
    hours = (sp["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    label = variable if level is None else f"{variable}@{level}hPa"
    return pd.DataFrame(
        [
            {
                "model": run,
                "lead_time_hours": float(h),
                "variable": label,
                "band": str(b),
                "LSD": float(lsd.sel(band=b).values[i]),
            }
            for b in lsd["band"].values
            for i, h in enumerate(hours)
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--all", action="store_true", help="sweep every model and Phase-1 run")
    ap.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="--all only: variable specs, e.g. geopotential@500 2m_temperature",
    )
    ap.add_argument("--variable", default="geopotential")
    ap.add_argument("--level", type=int, default=500)
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--stride", type=int, default=24, help="lead stride in hours")
    ap.add_argument("--validate", action="store_true", help="compare against the eval CSV")
    ap.add_argument("--dump-spectra", default=None, help="npz path for the init-averaged spectra")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.all:
        out = args.out or "tools/data/lsd_bands_phase1_pooled.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        specs = (
            [parse_spec(v) for v in args.variables]
            if args.variables
            else [(args.variable, args.level)]
        )
        frames = []
        for name, level in specs:
            for model in MODELS:
                root = STORE / "ablation" / args.phase / model / INITS[0]
                runs = sorted(
                    (p for p in os.listdir(root) if p.startswith("mag_")),
                    key=lambda r: float(r.split("_")[1]),
                )
                for run in runs:
                    print(f"[{name}@{level}] [{model}] {run}", flush=True)
                    d = banded_lsd(model, run, name, level, args.phase, args.stride)
                    d.insert(0, "backbone", model)
                    frames.append(d)
                    # Write after every run so a wall-clock kill keeps what finished.
                    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"Wrote {out} ({sum(len(f) for f in frames)} rows)")
        return

    df = banded_lsd(args.model, args.run, args.variable, args.level, args.phase, args.stride)

    if args.dump_spectra:
        st, sp = banded_lsd.spectra
        np.savez_compressed(
            args.dump_spectra,
            wavenumber=st["wavenumber"].values,
            lead_hours=(st["lead_time"].values / np.timedelta64(1, "h")).astype(int),
            energy_target=st.values,
            energy_prediction=sp.values,
        )
        print(f"Wrote {args.dump_spectra}")

    if args.validate:
        ref = pd.read_csv(
            STORE
            / "ablation"
            / args.phase
            / args.model
            / "intercomparison"
            / "energy_spectra"
            / "lsd_metrics_banded_lead_time_combined.csv"
        )
        ref = ref[(ref["model"] == args.run) & (ref["variable"] == df["variable"].iloc[0])]
        m = df.merge(
            ref, on=["model", "lead_time_hours", "variable", "band"], suffixes=("", "_ref")
        )
        err = (m["LSD"] - m["LSD_ref"]).abs()
        print(
            f"validate: n={len(m)} max_abs_err={err.max():.3e} max_rel={(err / m['LSD_ref'].abs()).max():.3e}"
        )
        return

    out = args.out or f"tools/data/lsd_bands_{args.model}_{args.run}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
