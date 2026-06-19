"""Model-soups view of the deterministic gain: how a single perturbed member
and the ensemble mean compare to the unperturbed control (mag_0), in RMSE.

For each production pick we area-weight RMSE (cos-lat) against WeatherBench-2
ERA5 at lead 240 h, averaged over the 4 ablation inits, and report the per-7-variable
mean relative change vs the control's single-forecast RMSE:
  - per-member  : mean over the 10 perturbed members of (RMSE_member - RMSE_ctrl)/RMSE_ctrl
  - ensemble-mean: (RMSE_ensmean - RMSE_ctrl)/RMSE_ctrl

The ensemble-mean RMSE here reproduces the pipeline's deterministic
temporal_metrics_combined.csv to 3 decimals (validation), so the per-member
numbers from the same code path are trustworthy.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import xarray as xr

B = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation"
WB2 = [
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
]
INITS = ["20230515", "20230815", "20241115", "20240215"]
LEAD_IDX = 40  # 240 h (6-h steps); overridden by --lead at runtime
VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
PICKS = {
    "Aurora encoder": ("aurora", "phase2b", "mag_0.025_layer_encoder"),
    "GraphCast all": ("graphcast_operational", "phase1", "mag_0.01_layer_all"),
    "SFNO modes10": ("sfno", "phase3", "mag_0.25_modes10"),
}
CONTROL = ("phase1", "mag_0_layer_all")

truths = [xr.open_zarr(p) for p in WB2]


def truth_at(vt):
    for t in truths:
        if t.time.values[0] <= np.datetime64(vt) <= t.time.values[-1]:
            return t
    raise ValueError(f"no truth covers {vt}")


def rmse_member_and_mean(model, phase, cell, vname):
    ems, mems = [], []
    for it in INITS:
        ds = xr.open_zarr(f"{B}/{phase}/{model}/{it}/{cell}/forecast.zarr")
        f = ds[vname].isel(lead_time=LEAD_IDX)
        vt = np.datetime64(ds.init_time.values[0]) + ds.lead_time.values[LEAD_IDX]
        tr = truth_at(vt)[vname].sel(time=vt)
        if vname in VARS_3D:
            f = f.sel(level=[500, 850])
            tr = tr.sel(level=[500, 850])
        tr = tr.sel(latitude=f.latitude, longitude=f.longitude)
        w = np.cos(np.deg2rad(f.latitude))
        w = w / w.mean()
        ems.append(
            float(
                np.sqrt(
                    (((f.mean("ensemble") - tr) ** 2) * w).mean(("latitude", "longitude")).mean()
                )
            )
        )
        mems.append(
            np.mean(
                [
                    float(
                        np.sqrt(
                            (((f.isel(ensemble=m) - tr) ** 2) * w)
                            .mean(("latitude", "longitude"))
                            .mean()
                        )
                    )
                    for m in range(f.sizes["ensemble"])
                ]
            )
        )
    return np.mean(ems), np.mean(mems)


def main():
    global LEAD_IDX
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=240, help="lead time in hours (6-h steps)")
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()
    if args.lead % 6 != 0:
        raise ValueError(f"--lead must be a multiple of 6 h, got {args.lead}")
    LEAD_IDX = args.lead // 6

    print(
        f"{'Production pick':16s} {'dRMSE member':>14s} {'dRMSE ens-mean':>16s}  "
        f"(7-var mean, {args.lead} h)"
    )
    print("-" * 70)
    rows = []
    for pretty, (model, phase, cell) in PICKS.items():
        rel_mem, rel_em = [], []
        for vname in VARS_2D + VARS_3D:
            c_em, _ = rmse_member_and_mean(model, CONTROL[0], CONTROL[1], vname)
            p_em, p_mem = rmse_member_and_mean(model, phase, cell, vname)
            rel_mem.append((p_mem - c_em) / c_em)
            rel_em.append((p_em - c_em) / c_em)
        d_mem = float(np.mean(rel_mem) * 100)
        d_em = float(np.mean(rel_em) * 100)
        print(f"{pretty:16s} {d_mem:>+13.1f}% {d_em:>+15.1f}%")
        rows.append(
            {
                "pick": pretty,
                "model": model,
                "lead_h": args.lead,
                "drmse_member_pct": round(d_mem, 3),
                "drmse_ensmean_pct": round(d_em, 3),
            }
        )

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["pick", "model", "lead_h", "drmse_member_pct", "drmse_ensmean_pct"],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
