"""Sanity plot: AIFS-CRPS member 0 MSL field around Milton peak with IBTrACS truth + tracker overlay.

Verifies that the tracker is locating what's actually in the forecast (not a wrong feature)
and shows whether AIFS-CRPS member 0 genuinely under-predicts RI or whether tracker missed it.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE = Path("/iopsstor/scratch/cscs/sadamov/milton_case_study")
FC = xr.open_zarr(
    "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifsens/20241004_0000/forecast.zarr",
    consolidated=True,
).isel(ensemble=0, init_time=0)

# Resolve valid_time
init_t = pd.Timestamp(FC["init_time"].values)
lead_h = (FC["lead_time"].values / np.timedelta64(1, "h")).astype(int)
valid = init_t + pd.to_timedelta(lead_h, unit="h")

# IBTrACS truth
ibt = xr.open_dataset(BASE / "milton_2024_ibtracs.nc")
# Convert IBTrACS lon to 0-360 to match forecast
ibt_lon = ibt["lon"].values % 360
ibt_lat = ibt["lat"].values
ibt_time = ibt["time"].values
ibt_pres = ibt["usa_pres"].values  # hPa

# Parse our tracker output for the Milton-region track
import re

stitch_text = (BASE / "test_stitch_aifsens_20241004_m0.txt").read_text()
blocks = re.split(r"(?=^start\t)", stitch_text, flags=re.M)
my_track_pts = []
for blk in blocks:
    if not blk.startswith("start"):
        continue
    pts = []
    for line in blk.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            lon = float(parts[3])
            lat = float(parts[4])
            psl = float(parts[5])
            yr, mo, dy, hr = int(parts[8]), int(parts[9]), int(parts[10]), int(parts[11])
            pts.append((lat, lon, psl, yr, mo, dy, hr))
        except Exception:
            continue
    # pick the Milton-region track (covers 20-30 N, 258-280 E during Oct 4-10 with at least 5 pts)
    in_milton = [
        p
        for p in pts
        if 18 <= p[0] <= 30
        and 258 <= p[1] <= 282
        and 4 <= p[5] <= 10
        and p[3] == 2024
        and p[4] == 10
    ]
    if len(in_milton) >= 5:
        my_track_pts = in_milton
        break

print(f"Tracker Milton-region track: {len(my_track_pts)} points")

# 4-panel grid showing MSL field at Oct 6 00, Oct 7 00, Oct 8 00, Oct 9 00 UTC
plot_times = [
    np.datetime64("2024-10-06T00:00:00"),
    np.datetime64("2024-10-07T00:00:00"),
    np.datetime64("2024-10-08T00:00:00"),
    np.datetime64("2024-10-09T00:00:00"),
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=270)},
)

# Domain: Gulf of Mexico + Caribbean + SE US
lon_min, lon_max, lat_min, lat_max = 255, 290, 12, 33

for ax, ptime in zip(axes.ravel(), plot_times):
    # Pick forecast time closest to ptime
    fidx = int(np.argmin(np.abs(valid - pd.Timestamp(ptime))))
    pt = pd.Timestamp(valid[fidx])
    msl = (
        FC["mean_sea_level_pressure"]
        .isel(lead_time=fidx)
        .sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )
        .values
        / 100.0
    )  # Pa -> hPa
    lat = FC["latitude"].sel(latitude=slice(lat_max, lat_min)).values
    lon = FC["longitude"].sel(longitude=slice(lon_min, lon_max)).values
    LO, LA = np.meshgrid(lon, lat)
    cf = ax.contourf(
        LO,
        LA,
        msl,
        levels=np.arange(900, 1025, 4),
        cmap="viridis_r",
        transform=ccrs.PlateCarree(),
        extend="both",
    )
    cs = ax.contour(
        LO,
        LA,
        msl,
        levels=np.arange(900, 1025, 4),
        colors="white",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%d")

    # IBTrACS truth position at this time
    ibt_idx = int(
        np.argmin(np.abs((pd.to_datetime(ibt_time) - pd.Timestamp(ptime)).total_seconds()))
    )
    ibt_dt = pd.Timestamp(ibt_time[ibt_idx])
    if abs((ibt_dt - pd.Timestamp(ptime)).total_seconds()) < 3600 * 4:
        ax.plot(
            ibt_lon[ibt_idx],
            ibt_lat[ibt_idx],
            marker="*",
            color="red",
            markersize=20,
            markeredgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=10,
            label=f"IBTrACS truth (MSL={ibt_pres[ibt_idx]:.0f} hPa)",
        )

    # Tracker output: nearest point
    track_pts_at_time = [
        p
        for p in my_track_pts
        if abs(
            (
                pd.Timestamp(f"{p[3]:04d}-{p[4]:02d}-{p[5]:02d}T{p[6]:02d}:00")
                - pd.Timestamp(ptime)
            ).total_seconds()
        )
        < 3600 * 4
    ]
    if track_pts_at_time:
        tp = track_pts_at_time[0]
        ax.plot(
            tp[1],
            tp[0],
            marker="^",
            color="white",
            markersize=12,
            markeredgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=10,
            label=f"tracker (MSL={tp[2]/100:.0f} hPa)",
        )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
    ax.set_extent([lon_min - 360, lon_max - 360, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_title(
        f"{pd.Timestamp(pt).strftime('%Y-%m-%d %H UTC')}  (lead {lead_h[fidx]} h)", fontsize=10
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(5)
    gl.ylocator = mticker.MultipleLocator(5)
    gl.xlabel_style = gl.ylabel_style = {"size": 7}

    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

cbar = fig.colorbar(
    cf, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04, pad=0.06, shrink=0.7
)
cbar.set_label("MSL pressure (hPa)")

fig.suptitle(
    "Hurricane Milton, AIFS-CRPS init 2024-10-04 00 UTC, member 0\n"
    "Forecast MSL (viridis), IBTrACS best-track truth (red star), TempestExtremes tracker (white triangle)",
    fontsize=11,
    y=0.98,
)
out = "/users/sadamov/pyprojects/ai-models-ensembles/figures/milton_aifsens_m0_sanity.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"-> {out}")
