"""Compute per-variable persistence MAE per lead time, for the headline figure.

Persistence forecast at init time t and lead h = ERA5(t) (the initial state
broadcast across all leads). MAE(h) is the temporal structure function of
ERA5, averaged over the 112 production init dates spanning 2023 + 2024.

Truth source: the WB2 partitions used by the rest of the verification
pipeline (weatherbench2_2022_2023.zarr + weatherbench2_2024_2025.zarr,
concatenated along time). This matches evaluate_ablation.sh and
evaluate_baselines.sh so persistence is apples-to-apples with the model
verification.

Init sampling: the EXACT 112 init times that the 7-way production grid
uses, derived from the aifsens baseline directory listing
($STORE/baselines/aifsens/<YYYYMMDD_HHMM>). 8 weeks (Jan/Apr/Jul/Oct 2-8
in 2023 + 2024) x 14 inits per week (7 days x {00, 12} UTC).

Writes a JSON keyed by (var, level) -> {lead_h: persistence_MAE} suitable
for direct CRPSS conversion via sigma_clim_1990_2019.json (or, until that
is computed, the existing sigma_clim_ablation.json).

Implementation: parallelised across the 12 (variable, level) keys via
ProcessPoolExecutor, with per-key bulk loading. Each worker opens WB2
fresh, then for every lead pulls all 112 init slabs and all 112 valid
slabs in two `sel(time=times)` calls instead of per-timestep load()s.
Each (var, level) key now takes ~5-10 min wall on Clariden's WB2 storage,
vs the ~75 min/key seen with the original per-timestep loop. Per-key
JSONL checkpointing means kills lose at most the in-flight key.

Output: /iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.json
Checkpoint: /iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

sys.stdout.reconfigure(line_buffering=True)

OUT = Path("/iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.json")
CHECKPOINT = Path("/iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.jsonl")

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEVELS = [500, 850]
LEADS_H = list(range(6, 246, 6))  # 6 h .. 240 h, matches paper-headline lead range

WB2_ZARRS = [
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
]


def _build_inits() -> list[str]:
    base = Path("/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifsens")
    out: list[str] = []
    for d in sorted(base.iterdir()):
        name = d.name
        if len(name) != 13 or name[8] != "_":
            continue
        yyyy, mm, dd, hh, mn = name[0:4], name[4:6], name[6:8], name[9:11], name[11:13]
        try:
            int(yyyy + mm + dd + hh + mn)
        except ValueError:
            continue
        out.append(f"{yyyy}-{mm}-{dd}T{hh}:{mn}")
    return out


INITS = _build_inits()
INIT_TIMES_NP = np.array([np.datetime64(s) for s in INITS])


def _open_wb2() -> xr.Dataset:
    parts = [xr.open_zarr(z, consolidated=True, chunks={"time": 200}) for z in WB2_ZARRS]
    return xr.concat(parts, dim="time").sortby("time")


def _compute_key(args: tuple[str, int | None]) -> tuple[str, dict[int, float] | None, float]:
    """Worker: compute the per-lead persistence MAE for one (var, level) key.

    Returns (key, per_lead_maes_dict_or_None, elapsed_seconds).
    """
    var, level = args
    key = var if level is None else f"{var}_{level}"
    t_start = time.time()

    # Pin numpy threads so 8 workers don't fight for the same cores.
    for env in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env, "1")

    ds = _open_wb2()
    if var not in ds.data_vars:
        return key, None, time.time() - t_start
    da = ds[var]
    if level is not None and "level" in da.dims:
        da = da.sel(level=level)

    # cos-lat weights, computed once per worker
    lat = da["latitude"].values
    w = np.cos(np.deg2rad(lat))
    w_sum = w.sum()
    n_lon = da.sizes["longitude"]

    # Bulk-load all 112 init slabs in one zarr call.
    try:
        init_slab = da.sel(time=INIT_TIMES_NP).load().values.astype(np.float32)
    except KeyError as e:
        print(f"  [{key}] cannot resolve some init times in WB2: {e}", flush=True)
        return key, None, time.time() - t_start

    per_lead_maes: dict[int, float] = {}
    for h in LEADS_H:
        valid_times = INIT_TIMES_NP + np.timedelta64(h, "h")
        try:
            valid_slab = da.sel(time=valid_times).load().values.astype(np.float32)
        except KeyError:
            # Some valid times may run off the end of WB2 (rare for h<=240)
            continue
        diff = np.abs(init_slab - valid_slab)  # (n_inits, lat, lon)
        # cos-lat-weighted spatial mean per init, then mean over inits
        per_init_mae = (diff * w[:, None]).sum(axis=(1, 2)) / (w_sum * n_lon)
        per_lead_maes[h] = float(per_init_mae.mean())

    return key, per_lead_maes, time.time() - t_start


def _all_keys() -> list[tuple[str, int | None]]:
    keys: list[tuple[str, int | None]] = []
    for v in VARS_2D:
        keys.append((v, None))
    for v in VARS_3D:
        for lvl in LEVELS:
            keys.append((v, lvl))
    return keys


def _load_checkpoint() -> dict[str, dict[int, float] | None]:
    done: dict[str, dict[int, float] | None] = {}
    if not CHECKPOINT.exists():
        return done
    for raw in CHECKPOINT.read_text().splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        if rec["maes"] is None:
            done[rec["key"]] = None
        else:
            done[rec["key"]] = {int(k): float(v) for k, v in rec["maes"].items()}
    return done


def main() -> None:
    print(f"Production grid: {len(INITS)} init times {INITS[0]} .. {INITS[-1]}", flush=True)
    keys_all = _all_keys()
    print(f"Total keys: {len(keys_all)} (12 expected: 2 2D + 5x2 3D)", flush=True)

    done = _load_checkpoint()
    if done:
        print(f"Resuming with {len(done)} keys already in checkpoint", flush=True)

    todo = [k for k in keys_all if (k[0] if k[1] is None else f"{k[0]}_{k[1]}") not in done]
    if not todo:
        print("Nothing to do; checkpoint already complete.", flush=True)
        OUT.write_text(json.dumps({k: v for k, v in done.items()}, indent=2))
        return
    print(f"Keys remaining: {len(todo)}", flush=True)

    results: dict[str, dict[int, float] | None] = dict(done)
    n_workers = int(os.environ.get("PERS_WORKERS", "8"))
    print(f"Launching {n_workers} workers...", flush=True)

    with CHECKPOINT.open("a") as ckpt, ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_compute_key, k): k for k in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            key, maes, dt = fut.result()
            results[key] = maes
            ckpt.write(json.dumps({"key": key, "maes": maes, "elapsed_s": dt}) + "\n")
            ckpt.flush()
            mae240 = maes.get(240) if maes else None
            tag = f"{mae240:.4f}" if mae240 is not None else "NA"
            print(f"  [{i:2d}/{len(todo)}] {key:32s} mae(240h)={tag} ({dt:.1f}s)", flush=True)
            # Snapshot the JSON after every key
            OUT.write_text(json.dumps(results, indent=2))

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n-> {OUT}", flush=True)
    print("\nPersistence MAE at lead 240 h (sanity check):", flush=True)
    for k, lead_maes in results.items():
        v = lead_maes.get(240) if isinstance(lead_maes, dict) else None
        print(f"  {k:32s} {v:.4f}" if v is not None else f"  {k:32s} --", flush=True)


if __name__ == "__main__":
    main()
