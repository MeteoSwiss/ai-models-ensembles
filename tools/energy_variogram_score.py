"""Energy Score + Variogram Score at a single lead for paper Tab. 3 supplement.

Two strictly proper multivariate scoring rules to complement the per-variable
CRPSS / SSR / SSIM / LSD / FSS / W$_1$ bundle already in Tab. 3.

ENERGY SCORE -- multivariate over (var, level) at each pixel, then cos(lat)-
weighted spatial mean. Tests joint inter-variable calibration at the same
location (e.g., is the (u, v) wind vector distribution jointly calibrated,
not just u marginal + v marginal). Lower is better.

    ES(F, y) = E_F ||X - y|| - (1/2) E_F ||X - X'||
    -> per-pixel ES with norm over (var, level), then spatial mean

VARIOGRAM SCORE -- per variable across pixel pairs, p=0.5 (Scheuerer & Hamill
2015). Tests spatial correlation structure. Lower is better.

    VS_p(F, y) = sum_{(i,j) in S} [|y_i - y_j|^p - (1/M) sum_m |X_m,i - X_m,j|^p]^2

Both scores are computed at a single lead time (default 240 h) on the 112-init
production-baseline grid. Output: CSV with one row per baseline x score.

Usage:
    python tools/energy_variogram_score.py \\
        --forecast-zarrs <path1> [<path2> ...] \\
        --truth-zarrs <truth.zarr> [<truth2.zarr>] \\
        --variables 2m_temperature mean_sea_level_pressure geopotential \\
                    temperature u_component_of_wind v_component_of_wind \\
                    specific_humidity \\
        --levels 500 850 \\
        --lead 240 \\
        --model-label <baseline_name> \\
        --variogram-pairs 5000 \\
        --out-csv <path>

Output columns: model, score, lead_hours, n_inits, n_members, value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _stack_var_level(
    fcst: xr.Dataset,
    truth: xr.Dataset,
    variables: list[str],
    levels: list[float],
    lead_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Stack the forecast and truth into multivariate arrays for one init.

    Returns
    -------
    ens : np.ndarray, shape (M, V, lat, lon)
        Ensemble forecast. V = number of (variable, level) entries that
        intersect what the dataset has.
    truth_arr : np.ndarray, shape (V, lat, lon)
        ERA5 truth.
    lat : np.ndarray, shape (lat,)
    """
    lead_idx = None
    if "lead_time" in fcst.dims:
        lt = fcst["lead_time"].values
        if np.issubdtype(lt.dtype, np.timedelta64):
            hours = (lt / np.timedelta64(1, "h")).astype(int)
        else:
            hours = lt.astype(int)
        if lead_hours not in hours:
            return None
        lead_idx = int(np.where(hours == lead_hours)[0][0])

    init_np = None
    if "init_time" in fcst.dims:
        init_np = fcst["init_time"].values[0]
    valid_time = init_np + np.timedelta64(lead_hours, "h") if init_np is not None else None

    ens_layers: list[np.ndarray] = []
    truth_layers: list[np.ndarray] = []
    for var in variables:
        if var not in fcst.data_vars or var not in truth.data_vars:
            continue
        fa = fcst[var]
        ta = truth[var]
        if "lead_time" in fa.dims:
            fa = fa.isel(lead_time=lead_idx)
        if "init_time" in fa.dims:
            fa = fa.isel(init_time=0)
        if valid_time is not None and "time" in ta.dims:
            try:
                ta = ta.sel(time=valid_time)
            except KeyError:
                return None

        if "level" in fa.dims:
            for lvl in levels:
                if float(lvl) in [float(x) for x in fa["level"].values.tolist()]:
                    fa_l = fa.sel(level=lvl)
                    ta_l = ta.sel(level=lvl).sel(latitude=fa_l["latitude"], method="nearest")
                    ens_layers.append(fa_l.values.astype(np.float32, copy=False))
                    truth_layers.append(ta_l.values.astype(np.float32, copy=False))
        else:
            ta = ta.sel(latitude=fa["latitude"], method="nearest")
            ens_layers.append(fa.values.astype(np.float32, copy=False))
            truth_layers.append(ta.values.astype(np.float32, copy=False))

    if not ens_layers:
        return None

    ens = np.stack(ens_layers, axis=1)  # (M, V, lat, lon)
    truth_arr = np.stack(truth_layers, axis=0)  # (V, lat, lon)

    # NORMALISE per (var, level) by truth std so the multivariate norm gives
    # equal weight to vars with different units. Per-layer std across the
    # truth field (single timestep) is a cheap proxy for climatological std.
    # Without this, geopotential ~10^4 m^2/s^2 swamps temperature ~ K.
    for k in range(truth_arr.shape[0]):
        s = float(truth_arr[k].std())
        if s > 0:
            ens[:, k, :, :] /= s
            truth_arr[k, :, :] /= s

    lat = fcst["latitude"].values
    return ens, truth_arr, lat


def energy_score(ens: np.ndarray, truth: np.ndarray, lat: np.ndarray) -> float:
    """Multivariate-over-(var,level) ES per pixel, cos(lat) spatial mean.

    Parameters
    ----------
    ens : (M, V, H, W) - normalised per-layer
    truth : (V, H, W) - normalised per-layer
    lat : (H,)
    """
    M = ens.shape[0]
    # term1 = E_F ||X - y||  (norm over (var, level) axis)
    diff = ens - truth[None]  # (M, V, H, W)
    norms = np.linalg.norm(diff, axis=1)  # (M, H, W)
    term1 = norms.mean(axis=0)  # (H, W)

    # term2 = (1/2) E ||X - X'|| -- unbiased ensemble variance estimator:
    # use sum_{i<j} / (M*(M-1)/2) / 2 = sum / (M*(M-1))
    # Vectorise over (i, j) pairs.
    pair_sum = np.zeros((ens.shape[2], ens.shape[3]), dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            pair_sum += np.linalg.norm(ens[i] - ens[j], axis=0)
    n_pairs = M * (M - 1) // 2
    term2 = 0.5 * (pair_sum / n_pairs)

    es_pixel = term1 - term2  # (H, W)
    w = np.cos(np.deg2rad(lat)).astype(np.float64)  # (H,)
    num = (es_pixel * w[:, None]).sum()
    den = (np.ones_like(es_pixel) * w[:, None]).sum()
    return float(num / den)


def variogram_score(
    ens: np.ndarray,
    truth: np.ndarray,
    n_pairs: int = 5000,
    p: float = 0.5,
    rng: np.random.Generator | None = None,
) -> float:
    """VS per layer over random pixel pairs, averaged across layers.

    Parameters
    ----------
    ens : (M, V, H, W)
    truth : (V, H, W)
    n_pairs : int
        Random pixel pairs per variable.
    p : float
        Variogram exponent (0.5 per Scheuerer & Hamill).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    M, V, H, W = ens.shape
    N = H * W
    i_idx = rng.integers(0, N, n_pairs)
    j_idx = rng.integers(0, N, n_pairs)

    vs_per_var = []
    for k in range(V):
        flat_t = truth[k].reshape(N)
        flat_e = ens[:, k].reshape(M, N)
        obs = np.abs(flat_t[i_idx] - flat_t[j_idx]) ** p
        ens_mean = (np.abs(flat_e[:, i_idx] - flat_e[:, j_idx]) ** p).mean(axis=0)
        vs_per_var.append(float(((obs - ens_mean) ** 2).sum() / n_pairs))
    return float(np.mean(vs_per_var))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast-zarrs", nargs="+", required=True)
    p.add_argument("--truth-zarrs", nargs="+", required=True)
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--levels", nargs="+", type=float, default=[500.0, 850.0])
    p.add_argument("--lead", type=int, default=240)
    p.add_argument("--model-label", required=True)
    p.add_argument("--variogram-pairs", type=int, default=5000)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    print("Opening truth zarrs (consolidated=True)...", flush=True)
    truth_parts = [xr.open_zarr(tz, consolidated=True, chunks={}) for tz in args.truth_zarrs]
    truth_ds = xr.concat(truth_parts, dim="time") if len(truth_parts) > 1 else truth_parts[0]
    truth_ds = truth_ds.sortby("time")

    es_per_init: list[float] = []
    vs_per_init: list[float] = []
    n_members = None

    rng = np.random.default_rng(42)
    # Expand any consolidated-layout zarr (one zarr with init_time as a dim,
    # many inits) into per-init xarray views so the existing per-init scorer
    # can consume them uniformly. The fcst_ds_iter yields (label, single-init
    # dataset) tuples for both layouts.
    def fcst_ds_iter():
        for fz in args.forecast_zarrs:
            try:
                ds = xr.open_zarr(fz, consolidated=True, chunks={})
            except Exception:
                ds = xr.open_zarr(fz, consolidated=False, chunks={})
            if "init_time" in ds.sizes and ds.sizes["init_time"] > 1:
                n = ds.sizes["init_time"]
                for j in range(n):
                    yield f"{Path(fz).name}#init_{j}", ds.isel(init_time=[j])
            else:
                yield Path(fz).parent.name, ds

    k_total = 0
    for k, (label, fcst_ds) in enumerate(fcst_ds_iter(), start=1):
        k_total = k
        stacked = _stack_var_level(fcst_ds, truth_ds, args.variables, args.levels, args.lead)
        if stacked is None:
            if k % 25 == 0:
                print(f"  [{k}] SKIP {label} (no usable layers)", flush=True)
            continue
        ens, truth_arr, lat = stacked
        n_members = ens.shape[0]
        es_val = energy_score(ens, truth_arr, lat)
        vs_val = variogram_score(ens, truth_arr, n_pairs=args.variogram_pairs, rng=rng)
        es_per_init.append(es_val)
        vs_per_init.append(vs_val)
        if k % 10 == 0:
            print(
                f"  [{k}] ES_mean={np.mean(es_per_init):.4g} "
                f"VS_mean={np.mean(vs_per_init):.4g}",
                flush=True,
            )

    n_inits = len(es_per_init)
    if n_inits == 0:
        print("No inits scored.", file=sys.stderr)
        return 1

    rows = [
        {
            "model": args.model_label,
            "score": "energy_score_mvar",
            "lead_hours": args.lead,
            "n_inits": n_inits,
            "n_members": n_members,
            "value": float(np.mean(es_per_init)),
        },
        {
            "model": args.model_label,
            "score": "variogram_score_p05",
            "lead_hours": args.lead,
            "n_inits": n_inits,
            "n_members": n_members,
            "value": float(np.mean(vs_per_init)),
        },
    ]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
