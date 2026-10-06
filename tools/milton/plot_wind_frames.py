"""Milton 10 m wind animation for the 2024-10-06 00 UTC initialisation (talk slide).

One frame per lead (0-120 h, 6-hourly). Columns: ERA5, SPW AIFS weight-only, SPW AIFS
weight+IC, AIFS-ENS, IFS-ENS. Rows: member 1 (ensemble index 0; for IFS-ENS the first
of the stratified 10 members used everywhere), ensemble mean and ensemble spread (std
across members, ddof=1) of 10 m wind speed; ERA5 fills rows 1-2, its row-3 cell carries
the lead time and init date. Rows 1-2 carry MSL contours every 4 hPa. Every panel shows
the NHC best track (IBTrACS USA-agency records, IBTrACS's own 3-hourly interpolation
dropped) and the best-track position linearly interpolated to the valid time.

Domain and the mean/std definitions follow the paper's member/spread maps
(tools/plot_milton_member_spread_maps.py). Colour limits are fixed for all frames and
sit just above the data (wind 99.9th pct 19-24 m/s, spread 99.9th pct 5-6 m/s); the
axes are placed at absolute positions (no tight bbox), so the frames are pixel-aligned.

Outputs, all in figures/talk/milton_wind_frames/:
  milton_wind_f{lead:03d}.png + frames.json, milton_wind_f096.pdf  static frames
  milton_wind.gif  the stepped frames; each frame's palette holds every colour-map colour
                   it uses (<= 200) plus its most frequent other colours, no dithering,
                   so the shading keeps its exact colours
  milton_wind_particles.mp4 (+ _poster.png)  the frames with wind particles advected
                   by the member-1 / ensemble-mean 10 m wind on rows 1-2 (nullschool-
                   style fading trails, opacity scaled with speed, track and star kept
                   on top). Each lead holds for 1.2 s and particles move with the true
                   wind over that 6 h (DT = 6 h / STEPS_PER_LEAD), so the flow and the
                   forecast clock share one time-lapse. H.264 yuv420p, BT.709 tagged.
  milton_wind_particles_1920.mp4  same video at 1920 px, bitrate-capped below 15 MB
                   (artifact / Claude Design asset limit)

Stage 1 extracts the Milton box once per source into $STORE/analysis/milton_wind_frames
(small netCDFs + the best-track CSV); stage 2 plots from those. IFS-ENS wind speed is
hypot(u, v), falling back to the zarr's own 10m_wind_speed (identical where both exist)
where u or v is a WB2 whole-field NaN. Leads still missing hold the last available lead
(u and v only as a pair, so particles never mix two leads) instead of flashing blank;
held panels say so in their corner ("held from +12 h").
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import map_coordinates

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

# Milton box of the paper figure (milton_F1 / F8).
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 255, 290, 13, 32
# 100 levels each (0.3 / 0.06 m/s steps): both maps then fit one 256-colour GIF palette.
CMAP_SPD = matplotlib.colormaps["turbo"].resampled(100)
CMAP_STD = matplotlib.colormaps["viridis"].resampled(100)
SPD_MAX, STD_MAX = 30, 6
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

GIF_MS, GIF_LAST_MS = 600, 2000

FPS, STEPS_PER_LEAD, HOLD_STEPS, SPINUP_STEPS = 30, 36, 60, 60
DT = 6 * 3600 / STEPS_PER_LEAD  # model seconds per video frame
M_PER_DEG = 111_195.0
N_PARTICLES, LIFE, FADE_IN = 1000, (40, 120), 10
FADE, TRAIL_ALPHA = 0.92, 0.8
DEPOSIT, SPD_REF = 0.45, 10.0  # trail opacity scales with speed up to SPD_REF m/s
CRF = 18
WEB_W, WEB_CRF, WEB_MAXRATE, WEB_BUFSIZE = 1920, 20, "4M", "8M"  # < 15 MB for 27 s


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
    out = xr.Dataset({"spd": spd, "u": ds[U], "v": ds[V], "msl": ds[MSL] / 100})
    out = out.astype("float32")
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
    ax.coastlines(resolution="50m", linewidth=0.4, color="0.75", zorder=3)
    ax.spines["geo"].set_linewidth(0.5)
    _track(ax, bt, pos)


def _track(ax, bt, pos):
    ax.plot(
        bt["lon"],
        bt["lat"],
        color="black",
        linewidth=0.8,
        transform=TR,
        zorder=4,
        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
    )
    if pos is not None:
        ax.plot(
            *pos,
            marker="*",
            markersize=11,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=0.8,
            transform=TR,
            zorder=5,
        )


def hold_missing(ds):
    """Fill whole-field NaN leads with the last available lead (u and v only as a pair).

    Returns the filled dataset and {lead_h: source lead_h} of the held wind-speed leads.
    """
    space = ["latitude", "longitude"]
    gaps = (
        (["spd"], ds["spd"].isnull().all(space)),
        (["msl"], ds["msl"].isnull().all(space)),
        (["u", "v"], (ds["u"].isnull() | ds["v"].isnull()).all(space)),
    )
    ds, held = ds.copy(deep=True), {}
    for names, gone in gaps:
        if "ensemble" in gone.dims:
            gone = gone.all("ensemble")
        last = None
        for lead, g in zip(LEADS_H.tolist(), gone.values):
            if not g:
                last = lead
            elif last is not None:
                for n in names:
                    ds[n].loc[{"lead_h": lead}] = ds[n].sel(lead_h=last).values
                if names == ["spd"]:
                    held[lead] = last
    return ds, held


def draw_frame(data, held, bt, lead_h):
    """Render one lead; returns (fig, {(row, col): axes})."""
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    pos = track_position(bt, INIT + pd.Timedelta(hours=int(lead_h)))
    axes = {}
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
            ax = axes[r, c] = _panel(fig, r, c)
            if src == "era5" and r == 2:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.56,
                    f"+{lead_h} h",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=24,
                    fontweight="bold",
                )
                ax.text(
                    0.5,
                    0.44,
                    f"Init {INIT:%Y-%m-%d %H} UTC",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=13,
                )
                continue
            _base(ax, bt, pos)
            if int(lead_h) in held[src]:
                ax.text(
                    0.03,
                    0.95,
                    f"held from +{held[src][int(lead_h)]} h",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="white",
                    zorder=6,
                )
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
                    colors="0.1",
                    linewidths=0.45,
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
        (LEFT, CMAP_SPD, SPD_MAX, np.arange(0, SPD_MAX + 1, 5), "10 m wind speed (m s$^{-1}$)"),
        (
            LEFT + 3 * (PANEL_W + GAP),
            CMAP_STD,
            STD_MAX,
            np.arange(0, STD_MAX + 1, 1),
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
    return fig, axes


def draw_overlay(bt, lead_h):
    """Track + star of rows 1-2 on a transparent canvas (uint8 RGBA), drawn above the particles."""
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="none")
    pos = track_position(bt, INIT + pd.Timedelta(hours=int(lead_h)))
    for r in range(2):
        for c in range(len(COLUMNS)):
            ax = _panel(fig, r, c)
            ax.set_extent([LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX], crs=TR)
            ax.set_facecolor("none")
            ax.spines["geo"].set_visible(False)
            _track(ax, bt, pos)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def _key(rgb):
    return (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]


def _quantize(rgb, keep):
    """P image with a 256-colour palette: the colour-map colours `keep` (keys) present in
    the frame first, then the most frequent others; remaining pixels go to the nearest."""
    uniq, inv, counts = np.unique(_key(rgb).ravel(), return_inverse=True, return_counts=True)
    cols = np.stack([uniq >> 16, (uniq >> 8) & 255, uniq & 255], -1).astype(np.float32)
    order = np.argsort(counts)[::-1]
    first = np.isin(uniq[order], keep)
    pal = cols[np.concatenate([order[first], order[~first]])[:256]]
    d = (cols**2).sum(1)[:, None] - 2 * cols @ pal.T + (pal**2).sum(1)[None]
    img = Image.fromarray(d.argmin(1).astype(np.uint8)[inv].reshape(rgb.shape[:2]))
    img.putpalette(pal.astype(np.uint8).ravel().tolist())
    return img


def write_gif(frames, path):
    lut = np.concatenate([CMAP_SPD(np.arange(CMAP_SPD.N)), CMAP_STD(np.arange(CMAP_STD.N))])
    keep = np.unique(_key((lut[:, :3] * 255 + 0.5).astype(np.uint8)))
    gif = [_quantize(f, keep) for f in frames]
    path.unlink(missing_ok=True)  # flaky Lustre clients have kept stale tails on overwrite
    durations = [GIF_MS] * (len(gif) - 1) + [GIF_LAST_MS]
    gif[0].save(
        path,
        save_all=True,
        append_images=gif[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=1,
    )


def _panel_wind(ds, r):
    if "ensemble" not in ds.dims:
        u, v = ds["u"], ds["v"]
    elif r == 0:
        u, v = ds["u"].isel(ensemble=0), ds["v"].isel(ensemble=0)
    else:
        u, v = ds["u"].mean("ensemble"), ds["v"].mean("ensemble")
    if bool(u.isnull().all() | v.isnull().all()):
        return None
    return u.values, v.values


def _splat(buf, x, y, wgt):
    h, w = buf.shape
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for dx, dy, wt in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        xx, yy = x0 + dx, y0 + dy
        m = (xx >= 1) & (xx < w - 1) & (yy >= 1) & (yy < h - 1)
        buf += np.bincount(yy[m] * w + xx[m], (wt * wgt)[m], minlength=h * w).reshape(h, w)


class Particles:
    """Particles in grid-index space of one panel, with a fading trail buffer in pixels."""

    def __init__(self, rng, lat, n_lon, box):
        self.rng, self.lat, self.ny, self.nx = rng, lat, lat.size, n_lon
        self.x0, self.y0, w, h = box
        self.buf = np.zeros((h, w), np.float32)
        self.sx, self.sy = w / (self.nx - 1), h / (self.ny - 1)
        self.i = rng.uniform(0, self.ny - 1, N_PARTICLES)
        self.j = rng.uniform(0, self.nx - 1, N_PARTICLES)
        self.life = rng.integers(*LIFE, N_PARTICLES)
        self.age = rng.integers(0, LIFE[0], N_PARTICLES)

    def step(self, wind):
        self.buf *= FADE
        if wind is None:
            self.buf[:] = 0
            return
        ij = np.stack([self.i, self.j])
        u = map_coordinates(wind[0], ij, order=1, mode="nearest")
        v = map_coordinates(wind[1], ij, order=1, mode="nearest")
        coslat = np.cos(np.deg2rad(np.interp(self.i, np.arange(self.ny), self.lat)))
        dlon = u * DT / (M_PER_DEG * coslat)
        dlat = v * DT / M_PER_DEG
        res = abs(self.lat[1] - self.lat[0])
        di, dj = -dlat / res, dlon / res
        i1, j1 = self.i + di, self.j + dj
        self.age += 1
        ok = (i1 >= 0) & (i1 <= self.ny - 1) & (j1 >= 0) & (j1 <= self.nx - 1)
        ok &= self.age < self.life
        env = np.clip(np.minimum(self.age, self.life - self.age) / FADE_IN, 0, 1)
        wgt = (DEPOSIT * np.clip(np.hypot(u, v) / SPD_REF, 0.15, 1) * env)[ok]
        for t in (0.25, 0.5, 0.75, 1.0):
            x, y = (self.j + t * dj)[ok] * self.sx, (self.i + t * di)[ok] * self.sy
            _splat(self.buf, x, y, wgt)
        n = int((~ok).sum())
        i1[~ok] = self.rng.uniform(0, self.ny - 1, n)
        j1[~ok] = self.rng.uniform(0, self.nx - 1, n)
        self.age[~ok] = 0
        self.life[~ok] = self.rng.integers(*LIFE, n)
        self.i, self.j = i1, j1

    def composite(self, frame):
        h, w = self.buf.shape
        a = np.minimum(self.buf, 1)[..., None] * TRAIL_ALPHA
        sl = frame[self.y0 : self.y0 + h, self.x0 : self.x0 + w]
        sl += a * (1 - sl)


def write_particles(data, bases, overlays, boxes, path, web_path, poster):
    rng = np.random.default_rng(0)
    lat = data["era5"]["latitude"].values
    n_lon = data["era5"]["longitude"].size
    parts = {rc: Particles(rng, lat, n_lon, box) for rc, box in boxes.items()}
    h, w = bases[0].shape[:2]
    yuv = "out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int"
    enc = [
        "-c:v", "libx264", "-preset", "slow", "-tune", "animation",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-color_range", "tv", "-movflags", "+faststart",
    ]  # fmt: skip
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
        "-filter_complex",
        f"[0:v]pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,split[a][b];"
        f"[a]scale={yuv},format=yuv420p[full];[b]scale={WEB_W}:-2:{yuv},format=yuv420p[web]",
        "-map", "[full]", *enc, "-crf", str(CRF), str(path),
        "-map", "[web]", *enc, "-crf", str(WEB_CRF), "-maxrate", WEB_MAXRATE,
        "-bufsize", WEB_BUFSIZE, str(web_path),
    ]  # fmt: skip
    for p in (path, web_path):
        p.unlink(missing_ok=True)  # an in-place overwrite once kept the old file's tail
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    winds = [
        {(r, c): _panel_wind(data[COLUMNS[c][0]].sel(lead_h=lead), r) for r, c in boxes}
        for lead in LEADS_H
    ]
    schedule = [(0, None)] * SPINUP_STEPS
    schedule += [(k, s) for k in range(len(LEADS_H)) for s in range(STEPS_PER_LEAD)]
    schedule += [(len(LEADS_H) - 1, STEPS_PER_LEAD + s) for s in range(HOLD_STEPS)]
    n_frames = 0
    for k, s in schedule:
        for rc, p in parts.items():
            p.step(winds[k][rc])
        if s is None:
            continue
        frame = bases[k].astype(np.float32) / 255
        for p in parts.values():
            p.composite(frame)
        ov = overlays[k].astype(np.float32) / 255
        frame += (ov[..., :3] - frame) * ov[..., 3:]
        out = (frame * 255 + 0.5).astype(np.uint8)
        proc.stdin.write(out.tobytes())
        if LEADS_H[k] == PDF_LEAD_H and s == STEPS_PER_LEAD // 2:
            Image.fromarray(out).save(poster)
            print(f"poster = video frame {n_frames}")
        n_frames += 1
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")
    print(f"-> {path}, {web_path}  {n_frames} frames, {n_frames / FPS:.1f} s")


def main():
    data = {src: extract(src) for src, _ in COLUMNS}
    bt = best_track()
    for src, title in COLUMNS:
        gone = data[src]["spd"].isnull().all(["latitude", "longitude"])
        no_uv = (data[src]["u"].isnull() | data[src]["v"].isnull()).all(["latitude", "longitude"])
        if "ensemble" in gone.dims:
            gone, no_uv = gone.all("ensemble"), no_uv.all("ensemble")
            print(f"{title:24s} row-1 member: ensemble={data[src]['ensemble'].values[0]}")
        print(
            f"{title:24s} missing leads (h): {LEADS_H[gone.values].tolist()}"
            f"  no particles (h): {LEADS_H[no_uv.values].tolist()}"
        )
    held = {}
    for src, title in COLUMNS:
        data[src], held[src] = hold_missing(data[src])
        if held[src]:
            print(f"{title:24s} held leads (lead: from): {held[src]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, bases, overlays, boxes = [], [], [], None
    for lead in LEADS_H:
        overlays.append(draw_overlay(bt, lead))
        name = f"milton_wind_f{lead:03d}"
        fig, axes = draw_frame(data, held, bt, lead)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        Image.fromarray(rgb).save(OUT_DIR / f"{name}.png")
        if lead == PDF_LEAD_H:
            fig.savefig(OUT_DIR / f"{name}.pdf", dpi=300, facecolor="white")
        if boxes is None:
            boxes = {}
            for (r, c), ax in axes.items():
                if r < 2:
                    bb = ax.get_window_extent()
                    boxes[r, c] = (
                        round(bb.x0),
                        round(rgb.shape[0] - bb.y1),
                        round(bb.width),
                        round(bb.height),
                    )
        plt.close(fig)
        bases.append(rgb)
        valid = INIT + pd.Timedelta(hours=int(lead))
        frames.append(
            {
                "file": f"{name}.png",
                "lead_h": int(lead),
                "valid_utc": valid.strftime("%Y-%m-%d %H UTC"),
            }
        )
        print(f"-> {OUT_DIR / name}.png  star={track_position(bt, valid)}")
    (OUT_DIR / "frames.json").write_text(json.dumps(frames, indent=2) + "\n")
    write_gif(bases, OUT_DIR / "milton_wind.gif")
    print(f"-> {OUT_DIR / 'milton_wind.gif'}")
    write_particles(
        data,
        bases,
        overlays,
        boxes,
        OUT_DIR / "milton_wind_particles.mp4",
        OUT_DIR / f"milton_wind_particles_{WEB_W}.mp4",
        OUT_DIR / "milton_wind_particles_poster.png",
    )


if __name__ == "__main__":
    main()
