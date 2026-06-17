"""Generate the case-study figures from the aggregated master + verification CSVs.

Figures:
  F1. Track spaghetti (3 init rows x 7 baseline cols, 10 members each + IBTrACS truth)
  F2. Intensity (min MSL) vs lead time per member, faceted by baseline
  F3. Cascading detection: ensemble-mean MSL field at Oct 7 18 UTC peak,
      from each of 14 inits, faceted per baseline (one figure per baseline)
  F4. Storm-relative composites at peak: MSL, 10m wind speed, 850 q, 500 Z;
      one panel per baseline (Cartesian visualisation).
  F5. Track + intensity error vs lead time, per baseline
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/
from model_colors import color_for, style_for

BASE = Path("/iopsstor/scratch/cscs/sadamov/milton_case_study")
FIGS = Path("/users/sadamov/pyprojects/ai-models-ensembles/figures")
FIGS.mkdir(parents=True, exist_ok=True)

BASELINES = [
    "aurora_encoder_ic",
    "graphcast_all_ic",
    "sfno_modes10_ic",
    "aifs_perturbed_ic",
    "aifsens",
    "fcn3",
    "atlas",
    "ifs_ens",
]
BASELINE_COLORS = {b: color_for(b) for b in BASELINES}


def load():
    master = pd.read_csv(BASE / "milton_master_tracks.csv", parse_dates=["time"])
    verif = pd.read_csv(BASE / "milton_verification.csv", parse_dates=["time"])
    return master, verif


def f1_track_spaghetti(master):
    """3 init rows x 3 baseline cols. Selected inits: 02 00 / 04 00 / 06 00.

    Reviewer asked to show only IFS-ENS, AIFS and AIFS-ENS, in that order.
    """
    f1_baselines = ["ifs_ens", "aifs_perturbed_ic", "aifsens"]
    init_picks = ["20241002_0000", "20241004_0000", "20241006_0000"]
    nrows, ncols = len(init_picks), len(f1_baselines)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.2, nrows * 3.0),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=270)},
    )
    # IBTrACS truth track
    ibt = xr.open_dataset(BASE / "milton_2024_ibtracs.nc")
    ibt_lat = ibt["lat"].values
    ibt_lon = ibt["lon"].values % 360
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 255, 290, 13, 32
    for r, init_tag in enumerate(init_picks):
        for c, b in enumerate(f1_baselines):
            ax = axes[r, c] if nrows > 1 else axes[c]
            sub = master[(master["baseline"] == b) & (master["init_tag"] == init_tag)]
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
            ax.set_extent([LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            # Truth (black so it never collides with the Atlas baseline colour)
            ax.plot(
                ibt_lon,
                ibt_lat,
                color="black",
                linewidth=1.5,
                transform=ccrs.PlateCarree(),
                label="IBTrACS" if (r == 0 and c == 0) else None,
                zorder=5,
            )
            # Members
            for m in sorted(sub["member"].unique()):
                tk = sub[sub["member"] == m].sort_values("time")
                ax.plot(
                    tk["lon"],
                    tk["lat"],
                    color=BASELINE_COLORS.get(b, "black"),
                    linewidth=0.5,
                    alpha=0.6,
                    transform=ccrs.PlateCarree(),
                )
            n_det = sub["member"].nunique()
            ax.set_title(
                f"{b.replace('_', ' ')}  ({n_det}/10)"
                if r == 0
                else f"{init_tag[6:8]} {init_tag[9:11]}Z  {b.replace('_', ' ')}  ({n_det}/10)",
                fontsize=8,
            )
            if c == 0:
                ax.text(
                    -0.18,
                    0.5,
                    f"init {init_tag[6:8]}/10 {init_tag[9:11]} UTC",
                    rotation=90,
                    transform=ax.transAxes,
                    va="center",
                    fontsize=9,
                )
    fig.suptitle(
        "Milton track spaghetti (10 members) per baseline x init  (black: IBTrACS truth)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS / f"milton_F1_track_spaghetti.{ext}"
        fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}")


def f2_intensity_vs_lead(master):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    for ax, b in zip(axes.flat, BASELINES):
        sub = master[master["baseline"] == b]
        for (init_tag, m), grp in sub.groupby(["init_tag", "member"]):
            grp = grp.sort_values("time")
            init_t = pd.Timestamp(
                f"{init_tag[0:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
            )
            lead = [(t - init_t).total_seconds() / 3600 for t in grp["time"]]
            ax.plot(lead, grp["psl_hpa"], color="steelblue", alpha=0.18, linewidth=0.6)
        ax.set_title(b.replace("_", " "), fontsize=12)
        ax.set_xlabel("lead time (h)", fontsize=11)
        ax.set_ylabel("min MSL (hPa)", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.axhline(
            895,
            color="red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label="IBTrACS peak (895 hPa)",
        )
        ax.axhline(
            976,
            color="purple",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label="0.25deg ERA5 control min (976 hPa)",
        )
        ax.set_ylim(890, 1015)
        ax.grid(True, alpha=0.3)
        if b == BASELINES[0]:
            ax.legend(loc="lower right", fontsize=10)
    fig.suptitle(
        "Hurricane Milton MSL minimum vs lead time, per baseline (each line = one member from one init)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS / f"milton_F2_intensity_vs_lead.{ext}"
        fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}")


def f5_track_intensity_err_vs_lead(verif):
    """Track error + MSL error vs lead time, per baseline."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for b in BASELINES:
        sub = verif[verif["baseline"] == b].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("lead_h")
        agg = sub.groupby(pd.cut(sub["lead_h"], bins=np.arange(0, 252, 24))).agg(
            pos_err_ibt_km=("pos_err_ibt_km", "mean"),
            psl_err_ibt=("psl_err_ibt", "mean"),
            psl_err_era5=("psl_err_era5", "mean"),
            n=("psl_err_ibt", "count"),
        )
        x = [i.mid for i in agg.index]
        axes[0].plot(
            x,
            agg["pos_err_ibt_km"],
            color=BASELINE_COLORS.get(b, "black"),
            linestyle=style_for(b),
            marker="o",
            linewidth=1.5,
            label=b.replace("_", " "),
        )
        axes[1].plot(
            x,
            agg["psl_err_era5"],
            color=BASELINE_COLORS.get(b, "black"),
            linestyle=style_for(b),
            marker="o",
            linewidth=1.5,
            label=b.replace("_", " "),
        )
    axes[0].set_ylabel("Track error vs IBTrACS (km)")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("MSL bias vs ERA5 (hPa)")
    axes[1].set_xlabel("Lead time (h)")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        "Milton verification: track position error and MSL intensity bias vs lead time", fontsize=11
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS / f"milton_F5_verification_vs_lead.{ext}"
        fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}")


STORE = Path("/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles")
ALL_INITS = [
    "20241002_0000",
    "20241002_1200",
    "20241003_0000",
    "20241003_1200",
    "20241004_0000",
    "20241004_1200",
    "20241005_0000",
    "20241005_1200",
    "20241006_0000",
    "20241006_1200",
    "20241007_0000",
    "20241007_1200",
    "20241008_0000",
    "20241008_1200",
]
# Hurricane Milton peak intensity per IBTrACS: 2024-10-08 06 UTC was Cat 5
# central pressure low. We use Oct 9 12 UTC ("near landfall") as a strong-impact
# valid time that ALL inits can verify against within the 240 h horizon.
VALID_TIME = np.datetime64("2024-10-09T12:00")


def _msl_ensmean_at(baseline: str, init_tag: str, valid_t: np.datetime64) -> xr.DataArray | None:
    """Return the ensemble-mean MSL field at valid_t, or None if the init does
    not reach valid_t within its forecast horizon."""
    if baseline == "ifs_ens":
        # Consolidated multi-init zarr. Open with consolidated=False because
        # the in-progress reshard's .reshard_state.json breaks the
        # consolidated-metadata layout.
        try:
            init_t = np.datetime64(
                f"{init_tag[0:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
            )
            ds_all = xr.open_zarr(
                "/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr",
                consolidated=False,
                chunks={},
            )
            ds = ds_all.sel(init_time=init_t)
        except Exception as e:
            print(f"  WARN ifs_ens {init_tag}: {type(e).__name__} {str(e)[:80]}")
            return None
    else:
        zp = STORE / "baselines" / baseline / init_tag / "forecast.zarr"
        if not zp.exists():
            return None
        try:
            ds = xr.open_zarr(zp, consolidated=True, chunks={})
        except Exception as e:
            print(f"  WARN {baseline} {init_tag}: {type(e).__name__} {str(e)[:80]}")
            return None
    try:
        if "init_time" in ds.dims:
            ds = ds.isel(init_time=0)
        if "mean_sea_level_pressure" not in ds.data_vars:
            return None
        if "lead_time" in ds.dims:
            lt = ds["lead_time"].values
            if np.issubdtype(lt.dtype, np.timedelta64):
                init_t = (
                    ds["init_time"].values
                    if "init_time" in ds.coords
                    else np.datetime64(
                        f"{init_tag[0:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
                    )
                )
                valid_arr = init_t + lt
            else:
                valid_arr = lt
            if valid_t not in valid_arr:
                return None
            ds = ds.isel(lead_time=int(np.where(valid_arr == valid_t)[0][0]))
        msl = ds["mean_sea_level_pressure"]
        if "ensemble" in msl.dims:
            msl = msl.mean("ensemble")
        return msl.load()
    except Exception as e:
        print(f"  WARN {baseline} {init_tag}: {type(e).__name__} {str(e)[:80]}")
        return None


def f3_cascading_detection(baseline: str = "aifsens"):
    """14-panel grid of ensemble-mean MSL at VALID_TIME, one per init, lead time
    decreasing top-left to bottom-right. Shows how the forecast converges as
    init approaches the event. Default baseline: AIFS-CRPS."""
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 260, 290, 18, 38
    ibt = xr.open_dataset(BASE / "milton_2024_ibtracs.nc")
    ibt_t = ibt["time"].values
    if VALID_TIME in ibt_t:
        idx = int(np.where(ibt_t == VALID_TIME)[0][0])
        truth_lat = float(ibt["lat"].isel(record=idx).values)
        truth_lon = float(ibt["lon"].isel(record=idx).values) % 360
    else:
        truth_lat = truth_lon = None

    fig, axes = plt.subplots(
        2, 7, figsize=(20, 6.5), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=270)}
    )
    for ax, init_tag in zip(axes.flat, ALL_INITS):
        msl = _msl_ensmean_at(baseline, init_tag, VALID_TIME)
        ax.set_rasterization_zorder(0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", zorder=3)
        ax.set_extent([LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        if msl is None:
            ax.text(
                0.5,
                0.5,
                "no forecast\nreaches\nvalid time",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="gray",
            )
        else:
            cf = ax.contourf(
                msl["longitude"],
                msl["latitude"],
                msl / 100,
                levels=np.arange(990, 1025, 2),
                cmap="RdYlBu_r",
                extend="both",
                transform=ccrs.PlateCarree(),
                zorder=-1,
            )
            ax.contour(
                msl["longitude"],
                msl["latitude"],
                msl / 100,
                levels=np.arange(990, 1025, 4),
                colors="black",
                linewidths=0.3,
                transform=ccrs.PlateCarree(),
                zorder=-1,
            )
        if truth_lat is not None:
            ax.plot(
                truth_lon,
                truth_lat,
                marker="*",
                color="red",
                markersize=10,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )
        init_t = pd.Timestamp(
            f"{init_tag[0:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
        )
        lead = int((pd.Timestamp(VALID_TIME) - init_t).total_seconds() / 3600)
        ax.set_title(f"init {init_tag[6:8]}/10 {init_tag[9:11]}Z, lead {lead}h", fontsize=8)

    fig.suptitle(
        f"{baseline.replace('_', ' ')} ensemble-mean MSL at 2024-10-09 12 UTC (Milton landfall window) -- "
        f"how 14 init times converge on the event (red star: IBTrACS truth position)",
        fontsize=11,
        y=0.995,
    )
    cbar_ax = fig.add_axes([0.25, -0.03, 0.5, 0.02])
    fig.colorbar(cf, cax=cbar_ax, orientation="horizontal", label="MSL (hPa)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS / f"milton_F3_cascading_{baseline}.{ext}"
        fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}")


def _storm_relative_thickness(thick, center_field, rel_lat, rel_lon, box, center_on="min"):
    """Recentre a 500-850 thickness field on the center_field extremum.

    thick / center_field are 2D (lat, lon) DataArrays on the same grid.
    center_on="min" locates the MSL minimum (the cyclone centre); "max" locates
    the thickness maximum (warm-core peak), used as a fallback when MSL is
    missing at this lead (e.g. the IFS-ENS consolidated zarr has NaN MSL slices).
    Returns the storm-relative thickness anomaly (field minus a fitted background
    plane) on the +-10 deg rel grid, or None if the search box is empty / all-NaN.
    """
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = box
    c_box = center_field.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
    if c_box.size == 0 or not np.isfinite(c_box.values).any():
        return None
    flat_idx = int(np.nanargmax(c_box.values) if center_on == "max" else np.nanargmin(c_box.values))
    ny, nx = c_box.shape
    iy, ix = flat_idx // nx, flat_idx % nx
    ctr_lat = float(c_box["latitude"].values[iy])
    ctr_lon = float(c_box["longitude"].values[ix])
    a = thick.interp(
        latitude=ctr_lat + rel_lat, longitude=ctr_lon + rel_lon, method="linear"
    ).values
    # Remove the background meridional thickness gradient: fit a plane to the
    # environment (radius > 5 deg, little storm influence) and subtract it, so
    # the residual isolates the warm-core anomaly above its local surroundings.
    RLON, RLAT = np.meshgrid(rel_lon, rel_lat)
    r = np.sqrt(RLON**2 + RLAT**2)
    env = (r > 5.0) & np.isfinite(a)
    A = np.column_stack([np.ones(env.sum()), RLON[env], RLAT[env]])
    coef = np.linalg.lstsq(A, a[env], rcond=None)[0]
    plane = coef[0] + coef[1] * RLON + coef[2] * RLAT
    return a - plane


def f4_storm_relative_composite(
    baselines: list[str] | None = None, init_tag: str = "20241004_0000"
):
    """Storm-relative 500-850 hPa thickness anomaly composite, with ERA5 control.

    The 500-850 hPa thickness is the warm-core signature of a TC. For each
    member of each baseline at a given init, recentre the thickness field on the
    member's predicted MSL minimum (within a search box), bilinearly interpolate
    onto a fixed +-10 deg storm-relative grid, take the anomaly relative to the
    box mean, then mean across members. The ERA5 panel composites the analysis
    thickness the same way (centred on the ERA5 MSL minimum) and sets the
    reference warm-core amplitude. Suppression = 1 - (baseline peak)/(ERA5 peak).
    """
    if baselines is None:
        baselines = BASELINES
    G = 9.80665  # m s^-2; convert geopotential (m^2 s^-2) to geopotential metres
    BOX = (260, 290, 12, 38)
    # Inner 3 deg disc used to read off the central warm-core amplitude.
    rel_lat = np.arange(-10.0, 10.25, 0.25)
    rel_lon = np.arange(-10.0, 10.25, 0.25)
    RLON, RLAT = np.meshgrid(rel_lon, rel_lat)
    inner = np.sqrt(RLON**2 + RLAT**2) <= 3.0
    init_t = np.datetime64(
        f"{init_tag[0:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
    )
    valid_t = init_t + np.timedelta64(120, "h")  # lead 120 h ~ 09 Oct 00 UTC

    composites: dict[str, np.ndarray] = {}
    # ERA5 control (analysis thickness, single realisation).
    era = xr.open_dataset(BASE / "era5_milton_window.nc")
    e_thick = era["z_thick_500m850"].sel(time=valid_t) / G
    e_psl = era["psl"].sel(time=valid_t)
    composites["ERA5"] = _storm_relative_thickness(e_thick, e_psl, rel_lat, rel_lon, BOX)

    def _composite_from_ds(ds):
        """Mean storm-relative thickness anomaly over the ensemble at valid_t."""
        if "init_time" in ds.dims:
            ds = ds.isel(init_time=0)
        if "lead_time" in ds.dims:
            lt = ds["lead_time"].values
            leads_valid = init_t + lt if np.issubdtype(lt.dtype, np.timedelta64) else lt
            if valid_t not in leads_valid:
                return None
            ds = ds.isel(lead_time=int(np.where(leads_valid == valid_t)[0][0]))
        thick_ens = (
            ds["geopotential"].sel(level=500) - ds["geopotential"].sel(level=850)
        ).load() / G
        msl_ens = ds["mean_sea_level_pressure"].load()
        n_mem = thick_ens.sizes.get("ensemble", 1)
        stack = np.full((n_mem, len(rel_lat), len(rel_lon)), np.nan)
        any_member = False
        for mi in range(n_mem):
            th = thick_ens.isel(ensemble=mi) if "ensemble" in thick_ens.dims else thick_ens
            ms = msl_ens.isel(ensemble=mi) if "ensemble" in msl_ens.dims else msl_ens
            # The IFS-ENS consolidated zarr has NaN MSL at scattered leads
            # (incl. 120 h for some inits); fall back to the thickness-max as
            # the storm centre there so the warm-core composite still renders.
            if np.isfinite(ms.values).any():
                a = _storm_relative_thickness(th, ms, rel_lat, rel_lon, BOX, center_on="min")
            else:
                a = _storm_relative_thickness(th, th, rel_lat, rel_lon, BOX, center_on="max")
            # Skip all-NaN members so a partial-NaN field still contributes.
            if a is not None and np.isfinite(a).any():
                stack[mi] = a
                any_member = True
        if not any_member:
            return None
        return np.nanmean(stack, axis=0)

    for b in baselines:
        if b == "ifs_ens":
            # IFS-ENS is not in the per-init baselines/ layout; read its
            # 500/850 geopotential + MSL from the consolidated multi-init zarr
            # (same source _msl_ensmean_at uses), selected to this init.
            try:
                ds = xr.open_zarr(
                    "/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr",
                    consolidated=False,
                    chunks={},
                ).sel(init_time=init_t)
                composites[b] = _composite_from_ds(ds)
            except Exception as e:
                print(f"  WARN ifs_ens F4: {type(e).__name__} {str(e)[:80]}")
                composites[b] = None
            continue
        zp = STORE / "baselines" / b / init_tag / "forecast.zarr"
        if not zp.exists():
            composites[b] = None
            continue
        ds = xr.open_zarr(zp, consolidated=True, chunks={})
        composites[b] = _composite_from_ds(ds)

    panels = ["ERA5"] + [b for b in baselines if composites.get(b) is not None]
    era_peak = np.nanmean(composites["ERA5"][inner])
    vmax = np.nanmax([np.nanmax(np.abs(c)) for c in composites.values() if c is not None])
    levels = np.linspace(-vmax, vmax, 21)

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, p in zip(axes.flat, panels):
        comp = composites.get(p)
        title = "ERA5 (analysis)" if p == "ERA5" else p.replace("_", " ")
        if comp is None:
            ax.text(0.5, 0.5, "no forecast", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            continue
        cf = ax.contourf(
            rel_lon, rel_lat, comp, levels=levels, cmap="RdBu_r", extend="both", rasterized=True
        )
        ax.contour(
            rel_lon, rel_lat, comp, levels=levels[::4], colors="black", linewidths=0.3
        ).set_rasterized(True)
        ax.plot(0, 0, marker="x", color="black", markersize=8)
        ax.set_xlabel("dlon (deg)", fontsize=8)
        ax.set_ylabel("dlat (deg)", fontsize=8)
        ax.set_aspect("equal")
        peak = np.nanmean(comp[inner])
        if p == "ERA5":
            lab = f"{title}  peak={peak:.0f} m"
        else:
            supp = 100 * (1 - peak / era_peak)
            lab = f"{title}  peak={peak:.0f} m ({supp:+.0f}%)"
            print(f"  {p}: warm-core peak {peak:.1f} m, suppression {supp:+.1f}% vs ERA5")
        ax.set_title(lab, fontsize=9)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    print(f"  ERA5 warm-core peak {era_peak:.1f} m")

    fig.suptitle(
        f"Storm-relative 500-850 hPa thickness anomaly at lead 120 h "
        f"(init {init_tag}; valid {str(valid_t)[:13]} UTC)",
        fontsize=11,
        y=0.99,
    )
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.015])
    fig.colorbar(cf, cax=cbar_ax, orientation="horizontal", label="thickness anomaly (m)")
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    for ext in ("png", "pdf"):
        out = FIGS / f"milton_F4_storm_relative_composite.{ext}"
        fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}")


def main():
    master = pd.read_csv(BASE / "milton_master_tracks.csv", parse_dates=["time"])
    verif = pd.read_csv(BASE / "milton_verification.csv", parse_dates=["time"])
    print(f"loaded {len(master)} track rows, {len(verif)} verification rows")
    f1_track_spaghetti(master)
    f2_intensity_vs_lead(master)
    f3_cascading_detection("aifsens")
    f3_cascading_detection("atlas")
    f3_cascading_detection("fcn3")
    f3_cascading_detection("aurora_encoder_ic")
    f3_cascading_detection("graphcast_all_ic")
    f3_cascading_detection("sfno_modes10_ic")
    f3_cascading_detection("aifs_perturbed_ic")
    f3_cascading_detection("ifs_ens")
    f4_storm_relative_composite()
    f5_track_intensity_err_vs_lead(verif)


if __name__ == "__main__":
    main()
