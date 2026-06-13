#!/usr/bin/env python
"""Milton ensemble member / mean / spread spatial panel (10m wind speed + direction).

Layout mirrors a per-member/mean/std map comparison (cf. Oskarsson et al. Fig 25):
columns = ERA5 analysis + N ensemble systems, rows = member 1, member 2, ensemble
mean, ensemble std. Filled field is 10m wind speed (m/s); white arrows show 10m wind
direction on the member and mean panels. The std panel has its own sequential scale.

Default config: Hurricane Milton, init 2024-10-04 00 UTC, +132 h (valid 2024-10-09
12 UTC, landfall window), comparing AIFS-perturbed (post-hoc weight perturbation),
IFS-ENS (operational), AIFS-ENS (trained probabilistic) vs ERA5 analysis.

Note: IFS-ENS 10m winds on the WB2 grid are entirely NaN for some leads over the
Milton domain (upstream WB2 property). +120 h is one such gap; +132 h is clean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

STORE = Path("/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles")
ERA5 = "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr"
IFS_ENS = "/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr"
FIGS = Path("/users/sadamov/pyprojects/ai-models-ensembles/figures")

# Milton box (matches milton_F1)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 255, 290, 13, 32
IFS_SUBSAMPLE = list(range(0, 50, 5))  # 10 members, matches eval subsample

U, V = "10m_u_component_of_wind", "10m_v_component_of_wind"
CMAP_SPD, CMAP_STD = "YlOrRd", "viridis"  # wind speed -> swissclim YlOrRd; spread -> viridis

MODEL_TITLES = {
    "aifs_perturbed": "AIFS-perturbed",
    "aifs_perturbed_ic": "AIFS-perturbed",
    "ifs_ens": "IFS-ENS",
    "aifsens": "AIFS-ENS",
}
ROW_LABELS = ["member 1", "member 2", "ens. mean", "ens. std"]


def _box(da):
    return da.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))


def _speed(u, v):
    return np.sqrt(u**2 + v**2)


def load_truth(valid):
    ds = xr.open_zarr(ERA5, decode_timedelta=True).sel(time=valid)
    u, v = _box(ds[U]), _box(ds[V])
    return dict(
        lon=u.longitude.values,
        lat=u.latitude.values,
        u=u.values,
        v=v.values,
        spd=_speed(u.values, v.values),
    )


def load_model(model, init_tag, init_dt, lead_h):
    td = np.timedelta64(lead_h, "h")
    if model == "ifs_ens":
        ds = xr.open_zarr(IFS_ENS, decode_timedelta=True)
        ds = ds.sel(init_time=init_dt, lead_time=td).isel(ensemble=IFS_SUBSAMPLE)
    else:
        ds = xr.open_zarr(
            STORE / "baselines" / model / init_tag / "forecast.zarr", decode_timedelta=True
        )
        ds = ds.sel(lead_time=td).isel(init_time=0)
    u, v = _box(ds[U]), _box(ds[V])
    u, v = u.values, v.values  # (member, lat, lon)
    spd = _speed(u, v)
    return dict(
        lon=_box(ds[U]).longitude.values,
        lat=_box(ds[U]).latitude.values,
        u=u,
        v=v,
        spd=spd,
        u_mean=u.mean(0),
        v_mean=v.mean(0),
        spd_mean=spd.mean(0),
        spd_std=spd.std(0, ddof=1),
    )


def add_dir(ax, lon, lat, u, v, tr, step):
    mag = np.maximum(np.sqrt(u**2 + v**2), 1e-6)
    un, vn = u / mag, v / mag  # unit vectors: direction only, uniform length
    ax.quiver(
        lon[::step],
        lat[::step],
        un[::step, ::step],
        vn[::step, ::step],
        transform=tr,
        scale=28,
        width=0.004,
        color="white",
        alpha=0.85,
        zorder=4,
    )


def style(ax, tr):
    ax.set_extent([LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX], crs=tr)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.25")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", edgecolor="0.4")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="20241004_0000", help="init tag YYYYMMDD_HHMM")
    p.add_argument("--lead-h", type=int, default=132)
    p.add_argument("--models", nargs="+", default=list(MODEL_TITLES))
    p.add_argument("--step", type=int, default=8, help="quiver subsample stride")
    p.add_argument("--out", default="milton_F6_member_spread_maps")
    args = p.parse_args()

    init_tag = args.init
    init_dt = np.datetime64(
        f"{init_tag[:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
    )
    valid = init_dt + np.timedelta64(args.lead_h, "h")
    print(f"init {init_dt}  +{args.lead_h}h  valid {valid}")

    truth = load_truth(valid)
    M = [load_model(m, init_tag, init_dt, args.lead_h) for m in args.models]

    spd_pool = (
        [truth["spd"]]
        + [m["spd"][0] for m in M]
        + [m["spd"][1] for m in M]
        + [m["spd_mean"] for m in M]
    )
    vmax_spd = float(np.nanpercentile(np.concatenate([f.ravel() for f in spd_pool]), 99.5))
    vmax_std = float(np.nanpercentile(np.concatenate([m["spd_std"].ravel() for m in M]), 99.5))
    print(f"vmax_spd={vmax_spd:.1f}  vmax_std={vmax_std:.1f}")

    proj = ccrs.PlateCarree(central_longitude=270)
    tr = ccrs.PlateCarree()
    ncols = len(M)
    fig, axes = plt.subplots(
        5,
        ncols,
        figsize=(3.1 * ncols, 11.0),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    # Row 0: ERA5 truth band (col 0 only)
    for c in range(ncols):
        ax = axes[0, c]
        if c == 0:
            style(ax, tr)
            ax.pcolormesh(
                truth["lon"],
                truth["lat"],
                truth["spd"],
                cmap=CMAP_SPD,
                vmin=0,
                vmax=vmax_spd,
                transform=tr,
                shading="auto",
            )
            add_dir(ax, truth["lon"], truth["lat"], truth["u"], truth["v"], tr, args.step)
            ax.set_title("ERA5 analysis", fontsize=11, fontweight="bold")
        else:
            ax.axis("off")

    im_spd = im_std = None
    for c, m in enumerate(M):
        for r, kind in enumerate(["m1", "m2", "mean", "std"]):
            ax = axes[r + 1, c]
            style(ax, tr)
            if kind == "std":
                im_std = ax.pcolormesh(
                    m["lon"],
                    m["lat"],
                    m["spd_std"],
                    cmap=CMAP_STD,
                    vmin=0,
                    vmax=vmax_std,
                    transform=tr,
                    shading="auto",
                )
            else:
                fld = {"m1": m["spd"][0], "m2": m["spd"][1], "mean": m["spd_mean"]}[kind]
                uu = {"m1": m["u"][0], "m2": m["u"][1], "mean": m["u_mean"]}[kind]
                vv = {"m1": m["v"][0], "m2": m["v"][1], "mean": m["v_mean"]}[kind]
                im_spd = ax.pcolormesh(
                    m["lon"],
                    m["lat"],
                    fld,
                    cmap=CMAP_SPD,
                    vmin=0,
                    vmax=vmax_spd,
                    transform=tr,
                    shading="auto",
                )
                add_dir(ax, m["lon"], m["lat"], uu, vv, tr, args.step)
            if r == 0:
                ax.set_title(
                    MODEL_TITLES.get(args.models[c], args.models[c]), fontsize=11, fontweight="bold"
                )
            if c == 0:
                ax.text(
                    -0.10,
                    0.5,
                    ROW_LABELS[r],
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )

    cb1 = fig.colorbar(
        im_spd,
        ax=axes[0:4, :].ravel().tolist(),
        orientation="horizontal",
        fraction=0.018,
        pad=0.01,
        aspect=50,
    )
    cb1.set_label("10 m wind speed (m s$^{-1}$)", fontsize=10)
    cb2 = fig.colorbar(
        im_std,
        ax=axes[4, :].ravel().tolist(),
        orientation="horizontal",
        fraction=0.07,
        pad=0.04,
        aspect=50,
    )
    cb2.set_label("ensemble std of 10 m wind speed (m s$^{-1}$)", fontsize=10)

    vt = str(valid.astype("datetime64[h]")).replace("T", " ")
    fig.suptitle(
        f"Hurricane Milton 10 m wind: ensemble members, mean, spread "
        f"(init {init_dt.astype('datetime64[h]')} UTC, +{args.lead_h} h, valid {vt} UTC)",
        fontsize=12,
    )

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIGS / f"{args.out}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
