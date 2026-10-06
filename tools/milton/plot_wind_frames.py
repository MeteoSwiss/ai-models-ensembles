"""Milton 10 m wind animation frames for the 2024-10-06 00 UTC initialisation (talk slide).

One PNG per lead (0-120 h, 6-hourly). Columns: ERA5, SPW AIFS weight-only, SPW AIFS
weight+IC, AIFS-ENS, IFS-ENS. Rows: member 1 (ensemble index 0; for IFS-ENS the first
of the stratified 10 members used everywhere), ensemble mean and ensemble spread (std
across members, ddof=1) of 10 m wind speed; ERA5 fills rows 1-2 and leaves row 3 blank.
Rows 1-2 carry MSL contours every 4 hPa. Every panel shows the NHC best track (IBTrACS
USA-agency records, IBTrACS's own 3-hourly interpolation dropped) and the best-track
position linearly interpolated to the valid time.

Colour maps, domain and the mean/std definitions follow the paper's member/spread maps
(tools/plot_milton_member_spread_maps.py). Colour limits are fixed for all frames and
the axes are placed at absolute positions (no tight bbox), so the frames are
pixel-aligned and carry no per-frame text; the slide adds lead and valid time.

Stage 1 extracts the Milton box once per source into $STORE/analysis/milton_wind_frames
(small netCDFs + the best-track CSV); stage 2 plots from those. IFS-ENS wind speed is
hypot(u, v), falling back to the zarr's own 10m_wind_speed (identical where both exist)
where u or v is a WB2 whole-field NaN; leads with neither are drawn as n/a.

Output: figures/talk/milton_wind_frames/milton_wind_f{lead:03d}.png, frames.json and
milton_wind_f096.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from _env import BASELINES, FIGURES, IFS_ENS, STORE, WB2_2024  # noqa: E402

INIT = pd.Timestamp("2024-10-06T00:00")
INIT_TAG = "20241006_0000"
LEADS_H = np.arange(0, 121, 6)
PDF_LEAD_H = 96
IFS_STRATIFIED = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

CACHE = STORE / "analysis" / "milton_wind_frames"
OUT_DIR = FIGURES / "talk" / "milton_wind_frames"

U, V = "10m_u_component_of_wind", "10m_v_component_of_wind"
WS, MSL = "10m_wind_speed", "mean_sea_level_pressure"

# Milton box and colour maps of the paper figure (milton_F1 / F8).
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 255, 290, 13, 32
CMAP_SPD, CMAP_STD = "YlOrRd", "viridis"
SPD_MAX, STD_MAX = 50, 15
MSL_LEVELS = np.arange(880, 1061, 4)

COLUMNS = [
    ("era5", "ERA5"),
    ("aifs_perturbed", "SPW AIFS (weight only)"),
    ("aifs_perturbed_ic", "SPW AIFS (weight+IC)"),
    ("aifsens", "AIFS-ENS"),
    ("ifs_ens", "IFS-ENS"),
]
ROWS = ["Member 1", "Ensemble mean", "Ensemble spread"]

PROJ = ccrs.PlateCarree(central_longitude=270)
TR = ccrs.PlateCarree()

# Absolute layout in inches; panel aspect matches the PlateCarree box exactly.
FIG_W, FIG_H, DPI = 16.0, 9.5, 150
LEFT, RIGHT, GAP = 0.5, 0.15, 0.08
PANEL_W = (FIG_W - LEFT - RIGHT - 4 * GAP) / 5
PANEL_H = PANEL_W * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN)
GRID_H = 3 * PANEL_H + 2 * GAP
TITLE_H, CB_PAD, CB_H, CB_LABEL_H = 0.35, 0.3, 0.14, 0.5
BLOCK_H = TITLE_H + GRID_H + CB_PAD + CB_H + CB_LABEL_H
GRID_TOP = FIG_H - (FIG_H - BLOCK_H) / 2 - TITLE_H
CB_Y = GRID_TOP - GRID_H - CB_PAD - CB_H


def _box(ds):
    return ds.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))


def _lead_index(ds):
    lead_h = (ds["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    ds = ds.assign_coords(lead_time=lead_h).sel(lead_time=LEADS_H)
    return ds.rename(lead_time="lead_h")


def extract(source: str) -> xr.Dataset:
    path = CACHE / f"{source}_{INIT_TAG}.nc"
    if path.exists():
        return xr.open_dataset(path).load()
    print(f"[extract] {source}")
    if source == "era5":
        valid = INIT + pd.to_timedelta(LEADS_H, unit="h")
        ds = _box(xr.open_zarr(WB2_2024)[[U, V, MSL]].sel(time=valid.values)).load()
        ds = ds.assign_coords(time=LEADS_H).rename(time="lead_h")
    elif source == "ifs_ens":
        ds = xr.open_zarr(IFS_ENS, consolidated=False)[[U, V, WS, MSL]]
        ds = ds.sel(init_time=INIT.to_datetime64()).isel(ensemble=IFS_STRATIFIED)
        ds = _box(_lead_index(ds)).load()
    else:
        ds = xr.open_zarr(BASELINES / source / INIT_TAG / "forecast.zarr")[[U, V, MSL]]
        ds = _box(_lead_index(ds.isel(init_time=0))).load()
    spd = np.hypot(ds[U], ds[V])
    if source == "ifs_ens":
        spd = spd.fillna(ds[WS])
    out = xr.Dataset({"spd": spd, "msl": ds[MSL] / 100}).astype("float32")
    for v in out.data_vars:
        gone = out[v].isnull().all(["latitude", "longitude"])
        if bool((out[v].isnull().any(["latitude", "longitude"]) != gone).any()):
            raise RuntimeError(f"{source} {v} has partially-NaN fields")
        if "ensemble" in gone.dims and bool((gone.any("ensemble") != gone.all("ensemble")).any()):
            raise RuntimeError(f"{source} {v} NaN leads differ across members")
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)
    return out


def best_track() -> pd.DataFrame:
    path = CACHE / "milton_nhc_best_track.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["time"])
    import huracanpy

    ibt = huracanpy.load(source="ibtracs", ibtracs_subset="last3years")
    ibt = ibt.where((ibt["name"] == "MILTON") & (ibt["season"] == 2024), drop=True)
    df = pd.DataFrame(
        {
            "time": ibt["time"].values,
            "lat": ibt["usa_lat"].values,
            "lon": ibt["usa_lon"].values,
            "agency": ibt["usa_agency"].values,
        }
    )
    df = df[df["agency"] == "hurdat_atl"].drop(columns="agency").sort_values("time")
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def track_position(bt: pd.DataFrame, t: pd.Timestamp):
    tt = bt["time"].values.astype("datetime64[s]").astype(float)
    x = np.datetime64(t, "s").astype(float)
    if not tt[0] <= x <= tt[-1]:
        return None
    return np.interp(x, tt, bt["lon"].values), np.interp(x, tt, bt["lat"].values)


def _rect(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


def _panel(fig, r, c):
    x = LEFT + c * (PANEL_W + GAP)
    y = GRID_TOP - (r + 1) * PANEL_H - r * GAP
    return fig.add_axes(_rect(x, y, PANEL_W, PANEL_H), projection=PROJ)


def _base(ax, bt, pos):
    ax.set_extent([LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX], crs=TR)
    ax.coastlines(resolution="50m", linewidth=0.4, color="0.45", zorder=3)
    ax.spines["geo"].set_linewidth(0.5)
    ax.plot(bt["lon"], bt["lat"], color="black", linewidth=0.8, transform=TR, zorder=4)
    if pos is not None:
        ax.plot(
            *pos,
            marker="*",
            markersize=11,
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=0.5,
            transform=TR,
            zorder=5,
        )


def draw_frame(data, bt, lead_h, stems):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    pos = track_position(bt, INIT + pd.Timedelta(hours=int(lead_h)))
    for c, (src, title) in enumerate(COLUMNS):
        x_mid = LEFT + c * (PANEL_W + GAP) + PANEL_W / 2
        fig.text(
            x_mid / FIG_W,
            (GRID_TOP + 0.08) / FIG_H,
            title,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
        ds = data[src].sel(lead_h=lead_h)
        for r in range(3):
            ax = _panel(fig, r, c)
            if src == "era5" and r == 2:
                ax.axis("off")
                continue
            _base(ax, bt, pos)
            if bool(ds["spd"].isnull().all()):
                ax.text(
                    0.03,
                    0.95,
                    "n/a",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="0.5",
                )
                continue
            if src == "era5":
                spd, msl = ds["spd"], ds["msl"]
            elif r == 0:
                spd, msl = ds["spd"].isel(ensemble=0), ds["msl"].isel(ensemble=0)
            elif r == 1:
                spd, msl = ds["spd"].mean("ensemble"), ds["msl"].mean("ensemble")
            else:
                spd, msl = ds["spd"].std("ensemble", ddof=1), None
            ax.pcolormesh(
                ds["longitude"],
                ds["latitude"],
                spd,
                cmap=CMAP_STD if r == 2 else CMAP_SPD,
                vmin=0,
                vmax=STD_MAX if r == 2 else SPD_MAX,
                transform=TR,
                shading="auto",
                rasterized=True,
                zorder=1,
            )
            if msl is not None and not bool(msl.isnull().all()):
                ax.contour(
                    ds["longitude"],
                    ds["latitude"],
                    msl,
                    levels=MSL_LEVELS,
                    colors="0.2",
                    linewidths=0.4,
                    transform=TR,
                    zorder=2,
                )
    for r, label in enumerate(ROWS):
        y_mid = GRID_TOP - r * (PANEL_H + GAP) - PANEL_H / 2
        fig.text(
            (LEFT - 0.22) / FIG_W,
            y_mid / FIG_H,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
        )
    cb_w = 2 * PANEL_W + GAP
    for x0, cmap, vmax, ticks, label in (
        (LEFT, CMAP_SPD, SPD_MAX, np.arange(0, SPD_MAX + 1, 10), "10 m wind speed (m s$^{-1}$)"),
        (
            LEFT + 3 * (PANEL_W + GAP),
            CMAP_STD,
            STD_MAX,
            np.arange(0, STD_MAX + 1, 3),
            "Ensemble spread (m s$^{-1}$)",
        ),
    ):
        cax = fig.add_axes(_rect(x0, CB_Y, cb_w, CB_H))
        cb = fig.colorbar(
            ScalarMappable(Normalize(0, vmax), cmap),
            cax=cax,
            orientation="horizontal",
            extend="max",
            ticks=ticks,
        )
        cb.ax.tick_params(labelsize=11)
        cb.set_label(label, fontsize=12)
    for stem in stems:
        fig.savefig(stem, dpi=DPI if stem.suffix == ".png" else 300, facecolor="white")
    plt.close(fig)


def main():
    data = {src: extract(src) for src, _ in COLUMNS}
    bt = best_track()
    for src, title in COLUMNS:
        gone = data[src]["spd"].isnull().all(["latitude", "longitude"])
        if "ensemble" in gone.dims:
            gone = gone.all("ensemble")
            print(f"{title:24s} row-1 member: ensemble={data[src]['ensemble'].values[0]}")
        print(f"{title:24s} missing leads (h): {LEADS_H[gone.values].tolist()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for lead in LEADS_H:
        name = f"milton_wind_f{lead:03d}"
        stems = [OUT_DIR / f"{name}.png"]
        if lead == PDF_LEAD_H:
            stems.append(OUT_DIR / f"{name}.pdf")
        draw_frame(data, bt, lead, stems)
        valid = INIT + pd.Timedelta(hours=int(lead))
        frames.append(
            {
                "file": f"{name}.png",
                "lead_h": int(lead),
                "valid_utc": valid.strftime("%Y-%m-%d %H UTC"),
            }
        )
        print(f"-> {stems[0]}  star={track_position(bt, valid)}")
    (OUT_DIR / "frames.json").write_text(json.dumps(frames, indent=2) + "\n")


if __name__ == "__main__":
    main()
