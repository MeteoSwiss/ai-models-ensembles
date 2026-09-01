"""Attribute ensemble spread to weight perturbation versus initial conditions.

The paper runs weight-only and weight+IC but no IC-only control, so the share of
the short-lead dispersion that the weight perturbation actually contributes is
not separable (reviewer item 1). With the `*_ic_only` arms in place this scores
all three arms of each backbone on the SAME initialisations and reports, per
lead, the cos(lat)-weighted ensemble spread and the fair CRPS.

Only initialisations present in all three arms are used, so every comparison is
paired.

Output CSV: baseline,arm,variable,level,lead,spread,crps,n_inits

Usage:
  python tools/ic_weight_decomposition.py --backbones all --leads 6 24 72 120 240
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

sys.stdout.reconfigure(line_buffering=True)

STORE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
TRUTH_SRC = {
    2023: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    2024: "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
}
OUT_CSV = Path("/iopsstor/scratch/cscs/sadamov/ic_weight_decomposition.csv")

# backbone -> {arm: baseline dir name}
TRIPLETS = {
    "aurora": {
        "weight": "aurora_encoder",
        "ic": "aurora_ic_only",
        "weight+ic": "aurora_encoder_ic",
    },
    "graphcast": {
        "weight": "graphcast_all",
        "ic": "graphcast_ic_only",
        "weight+ic": "graphcast_all_ic",
    },
    "sfno": {
        "weight": "sfno_modes10",
        "ic": "sfno_ic_only",
        "weight+ic": "sfno_modes10_ic",
    },
    "aifs": {
        "weight": "aifs_perturbed",
        "ic": "aifs_ic_only",
        "weight+ic": "aifs_perturbed_ic",
    },
}

# The paper's seven variables, 3D ones at both levels: the same twelve
# (variable, level) channels the multivariate scores use. An earlier five-channel
# subset dropped v and q entirely, and q is the worst-calibrated variable in the
# intercomparison, so a decomposition run on the subset is not comparable with
# anything else in the paper.
VARS = [
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
    ("geopotential", 500),
    ("geopotential", 850),
    ("temperature", 500),
    ("temperature", 850),
    ("u_component_of_wind", 500),
    ("u_component_of_wind", 850),
    ("v_component_of_wind", 500),
    ("v_component_of_wind", 850),
    ("specific_humidity", 500),
    ("specific_humidity", 850),
]


_TRUTH_DS: dict[int, xr.Dataset] = {}


def _truth(year: int) -> xr.Dataset:
    """Opened truth stores are reused: this scorer would otherwise reopen the
    multi-TB WB2 zarr once per (arm, variable, init)."""
    if year not in _TRUTH_DS:
        _TRUTH_DS[year] = xr.open_zarr(TRUTH_SRC[year], consolidated=True, decode_timedelta=True)
    return _TRUTH_DS[year]


def fair_crps_pixel(members: np.ndarray, o: np.ndarray) -> np.ndarray:
    M = members.shape[0]
    skill = np.abs(members - o[None]).mean(axis=0)
    s = np.sort(members, axis=0)
    coef = 2.0 * np.arange(1, M + 1) - M - 1.0
    spread = np.tensordot(coef, s, axes=([0], [0])) / (M * (M - 1))
    return skill - spread


def wmean(field: np.ndarray, wlat: np.ndarray) -> float:
    W = np.broadcast_to(wlat[:, None], field.shape)
    m = np.isfinite(field)
    den = np.sum(np.where(m, W, 0.0))
    return float(np.sum(np.where(m, W * field, 0.0)) / den) if den else float("nan")


def _load_members(path: Path, var: str, level, lead: int):
    """(members, latitude, longitude) for one run/variable/lead, or None if unusable.

    The store is closed before returning. Leaving it open leaks its buffers, and
    with 112 inits x 3 arms x 5 variables per lead that is enough to get the
    process OOM-killed partway through (2026-08-28: SIGKILL at the fifth
    variable, reproducibly).
    """
    ds = None
    try:
        ds = xr.open_zarr(path, consolidated=True)
        da = ds[var].sel(lead_time=np.timedelta64(lead, "h"))
        if level is not None:
            da = da.sel(level=level)
        if "init_time" in da.dims:
            da = da.isel(init_time=0)
        v = da.values.astype(np.float64)
        if v.ndim != 3 or not np.isfinite(v).all():
            return None
        return v, ds["latitude"].values, ds["longitude"].values
    except Exception:
        return None
    finally:
        if ds is not None:
            ds.close()


def is_scorable(path: Path, var: str, level, lead: int) -> bool:
    """Whether this store can actually be scored, not merely whether it exists.

    A write interrupted by a filesystem stall leaves the directory and
    forecast.zarr in place but unreadable (2026-08-28: aifs_ic_only/20230704_0000
    kept 7 of 9 variables and no consolidated metadata). A presence-only check
    passes such a store and the init then drops silently at scoring time, so the
    reported n_inits sits below the number the --require-inits guard verified.
    """
    return _load_members(path, var, level, lead) is not None


def score_one(path: Path, var: str, level, lead: int, init: datetime):
    """(mean-of-std spread, mean variance, crps), or None if unusable.

    Two spread aggregations are returned because they answer different
    questions. The first is the cos-lat mean of the per-pixel spread, which is
    the quantity the paper's SSR uses. The second is the cos-lat mean of the
    per-pixel *variance*: variances are what add when two perturbation sources
    are independent, so testing additivity needs the variance aggregation
    rather than a mean of standard deviations.
    """
    loaded = _load_members(path, var, level, lead)
    if loaded is None:
        return None
    v, lat, lon = loaded
    valid = init + timedelta(hours=lead)
    t = _truth(valid.year)[var]
    if level is not None:
        t = t.sel(level=level)
    o = t.sel(time=np.datetime64(valid, "ns")).sel(latitude=lat, longitude=lon).values
    wlat = np.cos(np.deg2rad(lat))
    M = v.shape[0]
    variance = (M + 1.0) / M * v.var(axis=0, ddof=1)
    return (
        wmean(np.sqrt(variance), wlat),
        wmean(variance, wlat),
        wmean(fair_crps_pixel(v, o), wlat),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["all"])
    ap.add_argument("--leads", nargs="+", type=int, default=[6, 24, 72, 120, 240])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument(
        "--require-inits",
        type=int,
        default=112,
        help="fail unless every backbone has this many paired inits (0 disables)",
    )
    args = ap.parse_args()

    names = sorted(TRIPLETS) if args.backbones == ["all"] else args.backbones
    rows = ["baseline,arm,variable,level,lead,spread,spread_rms,crps,n_inits"]
    probe_var, probe_level = VARS[0]
    probe_lead = min(args.leads)

    for bb in names:
        arms = TRIPLETS[bb]
        dirs = {a: Path(f"{STORE}/baselines/{d}") for a, d in arms.items()}
        present, scorable = {}, {}
        for a, d in dirs.items():
            present[a] = {p.parent.name for p in d.glob("*/forecast.zarr")}
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                ok = ex.map(
                    lambda tag, _d=d: (
                        tag,
                        is_scorable(_d / tag / "forecast.zarr", probe_var, probe_level, probe_lead),
                    ),
                    sorted(present[a]),
                )
                scorable[a] = {tag for tag, good in ok if good}
            broken = sorted(present[a] - scorable[a])
            if broken:
                print(
                    f"{bb}/{a}: {len(broken)} store(s) present but unreadable: "
                    f"{' '.join(broken)}",
                    flush=True,
                )
        common = sorted(set.intersection(*scorable.values())) if scorable else []
        print(
            f"{bb}: {len(common)} initialisations scorable in all three arms "
            f"(present: { {a: len(s) for a, s in present.items()} })",
            flush=True,
        )
        if args.require_inits and len(common) < args.require_inits:
            short = {a: args.require_inits - len(s) for a, s in scorable.items()}
            raise SystemExit(
                f"{bb}: only {len(common)}/{args.require_inits} paired initialisations "
                f"(short by arm: {short}). Fill the gaps or pass --require-inits 0."
            )
        if not common:
            continue

        for var, level in VARS:
            for lead in args.leads:
                t0 = time.time()
                for arm, d in dirs.items():

                    def one(tag, _d=d, _var=var, _lv=level, _lead=lead):
                        init = datetime.strptime(tag, "%Y%m%d_%H%M")
                        return score_one(_d / tag / "forecast.zarr", _var, _lv, _lead, init)

                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        res = [r for r in ex.map(one, common) if r is not None]
                    if not res:
                        continue
                    if len(res) != len(common):
                        raise SystemExit(
                            f"{bb}/{arm} {var} lead {lead}h: scored {len(res)} of "
                            f"{len(common)} pre-validated inits. A store became "
                            f"unreadable mid-run; rerun once the filesystem is healthy."
                        )
                    sp = float(np.mean([r[0] for r in res]))
                    # variances average, then take the root: this is the aggregation
                    # that makes sigma_both^2 = sigma_wt^2 + sigma_ic^2 testable.
                    sp_rms = float(np.sqrt(np.mean([r[1] for r in res])))
                    cr = float(np.mean([r[2] for r in res]))
                    lvl = "" if level is None else level
                    rows.append(
                        f"{bb},{arm},{var},{lvl},{lead},{sp:.8g},{sp_rms:.8g},{cr:.8g},{len(res)}"
                    )
                print(
                    f"  {bb} {var}{'' if level is None else '_' + str(level)} "
                    f"lead {lead}h done ({time.time()-t0:.0f}s)",
                    flush=True,
                )
                Path(args.out).write_text("\n".join(rows) + "\n")

    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"DONE -> {args.out} ({len(rows)-1} rows)", flush=True)


if __name__ == "__main__":
    main()
