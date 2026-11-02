"""ERA5 control: run the same tracker on WB2 truth for Milton's 8-day window
and verify it recovers a track matching IBTrACS within ~50 km position and
~10 hPa intensity. Also produces a 4-panel plot analogous to the AIFS-CRPS
sanity plot so we can visually compare what the tracker SHOULD see (truth)
vs what it found in the model.
"""

from __future__ import annotations
import subprocess
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from _env import FIGURES, SCRATCH, WB2_2024  # noqa: E402

BASE = SCRATCH / "milton_case_study"
NC_OUT = BASE / "era5_milton_window.nc"
DETECT_OUT = BASE / "era5_detect.txt"
STITCH_OUT = BASE / "era5_stitch.txt"
OUT_PNG = FIGURES / "milton_era5_control.png"

T0, T1 = "2024-10-04T00:00", "2024-10-12T00:00"
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 240, 300, 5, 40


def build_input():
    if NC_OUT.exists():
        print(f"reusing existing {NC_OUT}")
        return
    print("opening WB2 truth zarr...")
    truth = xr.open_zarr(
        WB2_2024,
        consolidated=True,
        chunks={"time": 8, "level": 1},
    )
    sub = truth.sel(
        time=slice(T0, T1), latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX)
    )
    # Resample to 6 hourly if needed
    sub = sub.sel(time=sub["time"].dt.hour.isin([0, 6, 12, 18]))
    print(
        f"selected {sub.sizes['time']} timesteps, lat={sub.sizes['latitude']}, lon={sub.sizes['longitude']}"
    )

    out = xr.Dataset(
        coords={
            "time": sub["time"],
            "latitude": sub["latitude"],
            "longitude": sub["longitude"],
        }
    )
    out["psl"] = sub["mean_sea_level_pressure"].astype("float32")
    z500 = sub["geopotential"].sel(level=500).astype("float32")
    z850 = sub["geopotential"].sel(level=850).astype("float32")
    out["z_thick_500m850"] = z500 - z850
    u10 = sub["10m_u_component_of_wind"].astype("float32")
    v10 = sub["10m_v_component_of_wind"].astype("float32")
    out["v10"] = np.sqrt(u10**2 + v10**2)
    u850 = sub["u_component_of_wind"].sel(level=850).astype("float64")
    v850 = sub["v_component_of_wind"].sel(level=850).astype("float64")
    lat = sub["latitude"].values
    lon = sub["longitude"].values
    R = 6.371e6
    dlat = np.deg2rad(np.gradient(lat))
    dlon = np.deg2rad(np.gradient(lon))
    cos_lat = np.cos(np.deg2rad(lat))[:, None]
    dv_dx = np.gradient(v850.values, axis=2) / (R * cos_lat * dlon[None, None, :])
    du_dy = np.gradient(u850.values, axis=1) / (R * dlat[None, :, None])
    out["vort850"] = (("time", "latitude", "longitude"), (dv_dx - du_dy).astype("float32"))
    out["time"].encoding = {"units": "hours since 2024-10-04 00:00:00", "calendar": "standard"}
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    print("writing NetCDF...")
    out.to_netcdf(NC_OUT, encoding=encoding)
    print(f"wrote {NC_OUT.stat().st_size/1e6:.1f} MB")


def run_te():
    if STITCH_OUT.exists():
        print(f"reusing {STITCH_OUT}")
        return
    cmd_d = [
        "DetectNodes",
        "--in_data",
        str(NC_OUT),
        "--out",
        str(DETECT_OUT),
        "--searchbymin",
        "psl",
        "--closedcontourcmd",
        "psl,200.0,5.5,0",
        "--closedcontourcmd",
        "z_thick_500m850,-58.8,6.5,1.0",
        "--outputcmd",
        "psl,min,0;v10,max,2;vort850,max,2",
        "--mergedist",
        "6.0",
        "--latname",
        "latitude",
        "--lonname",
        "longitude",
        "--regional",
    ]
    r = subprocess.run(cmd_d, capture_output=True, text=True)
    print("DetectNodes stdout tail:")
    print("\n".join(r.stdout.splitlines()[-15:]))
    if r.returncode != 0 or "EXCEPTION" in r.stdout:
        print("DetectNodes failed; stderr:", r.stderr[-2000:])
        sys.exit(1)
    cmd_s = [
        "StitchNodes",
        "--in",
        str(DETECT_OUT),
        "--out",
        str(STITCH_OUT),
        "--in_fmt",
        "lon,lat,psl,v10,vort850",
        "--range",
        "5.0",
        "--mintime",
        "36h",
        "--maxgap",
        "6h",
        "--threshold",
        "v10,>=,17.0,3",
    ]
    r = subprocess.run(cmd_s, capture_output=True, text=True)
    if r.returncode != 0:
        print("StitchNodes stderr:", r.stderr[-2000:])
        sys.exit(1)
    print("stitched")


def parse_milton_track():
    text = STITCH_OUT.read_text()
    blocks = re.split(r"(?=^start\t)", text, flags=re.M)
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
                pts.append(
                    {
                        "time": pd.Timestamp(f"{yr:04d}-{mo:02d}-{dy:02d}T{hr:02d}:00"),
                        "lat": lat,
                        "lon": lon,
                        "psl_hpa": psl / 100,
                    }
                )
            except Exception:
                continue
        in_milton = [
            p
            for p in pts
            if 17 <= p["lat"] <= 30
            and 258 <= p["lon"] <= 282
            and pd.Timestamp("2024-10-04") <= p["time"] <= pd.Timestamp("2024-10-11")
        ]
        if len(in_milton) >= 5:
            return pd.DataFrame(in_milton)
    return pd.DataFrame()


def compare_to_ibtracs(track):
    ibt = xr.open_dataset(BASE / "milton_2024_ibtracs.nc")
    ibt_df = pd.DataFrame(
        {
            "time": pd.to_datetime(ibt["time"].values),
            "lat": ibt["lat"].values,
            "lon": ibt["lon"].values % 360,
            "usa_pres_hpa": ibt["usa_pres"].values,
        }
    )
    # match by nearest time within +-3h
    rows = []
    for _, t in track.iterrows():
        idx = (ibt_df["time"] - t["time"]).abs().idxmin()
        ib = ibt_df.iloc[idx]
        if abs((ib["time"] - t["time"]).total_seconds()) > 3 * 3600:
            continue
        # great-circle distance
        from math import radians, sin, cos, asin, sqrt

        lat1, lon1, lat2, lon2 = map(radians, [t["lat"], t["lon"], ib["lat"], ib["lon"]])
        d = (
            2
            * asin(
                sqrt(
                    sin((lat2 - lat1) / 2) ** 2
                    + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
                )
            )
            * 6371
        )
        rows.append(
            {
                "time": t["time"],
                "track_psl": t["psl_hpa"],
                "ibt_psl": ib["usa_pres_hpa"],
                "psl_err": t["psl_hpa"] - ib["usa_pres_hpa"],
                "track_lat": t["lat"],
                "track_lon": t["lon"],
                "ibt_lat": ib["lat"],
                "ibt_lon": ib["lon"],
                "pos_err_km": d,
            }
        )
    return pd.DataFrame(rows)


def make_plot(track, comp):
    ibt = xr.open_dataset(BASE / "milton_2024_ibtracs.nc")
    ibt_time = pd.to_datetime(ibt["time"].values)
    ibt_lat = ibt["lat"].values
    ibt_lon = ibt["lon"].values % 360
    ibt_pres = ibt["usa_pres"].values
    truth = xr.open_zarr(
        WB2_2024,
        consolidated=True,
        chunks={"time": 8},
    )
    plot_times = [
        np.datetime64(t)
        for t in [
            "2024-10-06T00:00:00",
            "2024-10-07T00:00:00",
            "2024-10-08T00:00:00",
            "2024-10-09T00:00:00",
        ]
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=270)}
    )
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 255, 290, 12, 33
    for ax, ptime in zip(axes.ravel(), plot_times):
        sub = (
            truth["mean_sea_level_pressure"]
            .sel(time=ptime, method="nearest")
            .sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
            .values
            / 100
        )
        lat = truth["latitude"].sel(latitude=slice(LAT_MAX, LAT_MIN)).values
        lon = truth["longitude"].sel(longitude=slice(LON_MIN, LON_MAX)).values
        LO, LA = np.meshgrid(lon, lat)
        cf = ax.contourf(
            LO,
            LA,
            sub,
            levels=np.arange(900, 1025, 4),
            cmap="viridis_r",
            transform=ccrs.PlateCarree(),
            extend="both",
        )
        cs = ax.contour(
            LO,
            LA,
            sub,
            levels=np.arange(900, 1025, 4),
            colors="white",
            linewidths=0.3,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs, inline=True, fontsize=6, fmt="%d")
        idx = int(np.argmin(np.abs((ibt_time - pd.Timestamp(ptime)).total_seconds())))
        if abs((ibt_time[idx] - pd.Timestamp(ptime)).total_seconds()) < 3600 * 4:
            ax.plot(
                ibt_lon[idx],
                ibt_lat[idx],
                marker="*",
                color="red",
                markersize=20,
                markeredgecolor="black",
                transform=ccrs.PlateCarree(),
                zorder=10,
                label=f"IBTrACS truth (MSL={ibt_pres[idx]:.0f} hPa)",
            )
        match = track[(track["time"] - pd.Timestamp(ptime)).abs() < pd.Timedelta(hours=4)]
        if not match.empty:
            r = match.iloc[0]
            ax.plot(
                r["lon"],
                r["lat"],
                marker="^",
                color="white",
                markersize=12,
                markeredgecolor="black",
                transform=ccrs.PlateCarree(),
                zorder=10,
                label=f"tracker (MSL={r['psl_hpa']:.0f} hPa)",
            )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
        ax.set_extent([LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.set_title(pd.Timestamp(ptime).strftime("%Y-%m-%d %H UTC"), fontsize=10)
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
    if not comp.empty:
        mean_pos_err = comp["pos_err_km"].mean()
        mean_psl_err = comp["psl_err"].mean()
        fig.suptitle(
            "ERA5 control: Hurricane Milton, WB2 truth field + IBTrACS truth track + tracker output\n"
            f"Tracker vs IBTrACS: mean position error {mean_pos_err:.0f} km, "
            f"mean MSL bias {mean_psl_err:+.1f} hPa (n={len(comp)} matched)",
            fontsize=11,
            y=0.98,
        )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"-> {OUT_PNG}")


if __name__ == "__main__":
    build_input()
    run_te()
    track = parse_milton_track()
    print(f"\nMilton-region track from tracker: {len(track)} points")
    if len(track):
        print(track.to_string(index=False))
    comp = compare_to_ibtracs(track)
    print(f"\nMatched against IBTrACS at {len(comp)} times:")
    if len(comp):
        print(
            comp[["time", "track_psl", "ibt_psl", "psl_err", "pos_err_km"]].to_string(index=False)
        )
    make_plot(track, comp)
