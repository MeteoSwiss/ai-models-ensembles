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
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

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
BASELINE_COLORS = {
    "aurora_encoder_ic": "#E67E22",
    "graphcast_all_ic": "#27AE60",
    "sfno_modes10_ic": "#2980B9",
    "aifs_perturbed_ic": "#D81B60",
    "aifsens": "#8B5A2B",
    "fcn3": "#D4A017",
    "atlas": "#C0392B",
    "ifs_ens": "#7F8C8D",
}


def load():
    master = pd.read_csv(BASE / "milton_master_tracks.csv", parse_dates=["time"])
    verif = pd.read_csv(BASE / "milton_verification.csv", parse_dates=["time"])
    return master, verif


def f1_track_spaghetti(master):
    """3 init rows x len(BASELINES) cols. Selected inits: 02 00 / 04 00 / 06 00."""
    init_picks = ["20241002_0000", "20241004_0000", "20241006_0000"]
    nrows, ncols = len(init_picks), len(BASELINES)
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
        for c, b in enumerate(BASELINES):
            ax = axes[r, c] if nrows > 1 else axes[c]
            sub = master[(master["baseline"] == b) & (master["init_tag"] == init_tag)]
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
            ax.set_extent([LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            # Truth
            ax.plot(
                ibt_lon,
                ibt_lat,
                color="red",
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
        "Milton track spaghetti (10 members) per baseline x init  (red: IBTrACS truth)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS / f"milton_F1_track_spaghetti.{ext}"
        fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}")


def f2_intensity_vs_lead(master):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, b in zip(axes.flat, BASELINES):
        sub = master[master["baseline"] == b]
        for (init_tag, m), grp in sub.groupby(["init_tag", "member"]):
            grp = grp.sort_values("time")
            init_t = pd.Timestamp(
                f"{init_tag[0:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
            )
            lead = [(t - init_t).total_seconds() / 3600 for t in grp["time"]]
            ax.plot(lead, grp["psl_hpa"], color="steelblue", alpha=0.18, linewidth=0.6)
        ax.set_title(b.replace("_", " "), fontsize=10)
        ax.set_xlabel("lead time (h)")
        ax.set_ylabel("min MSL (hPa)")
        ax.axhline(
            895,
            color="red",
            linestyle="--",
            linewidth=0.6,
            alpha=0.6,
            label="IBTrACS peak (895 hPa)",
        )
        ax.axhline(
            989,
            color="purple",
            linestyle="--",
            linewidth=0.6,
            alpha=0.6,
            label="ERA5 ceiling (~989 hPa)",
        )
        ax.set_ylim(890, 1015)
        ax.grid(True, alpha=0.3)
        if b == BASELINES[0]:
            ax.legend(loc="lower right", fontsize=7)
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
            marker="o",
            linewidth=1.5,
            label=b.replace("_", " "),
        )
        axes[1].plot(
            x,
            agg["psl_err_era5"],
            color=BASELINE_COLORS.get(b, "black"),
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
                "/capstor/store/cscs/mch/s83/IFS/ifs_ens.zarr",
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


def f4_storm_relative_composite(
    baselines: list[str] | None = None, init_tag: str = "20241004_0000"
):
    """Storm-relative MSL composite per baseline at a chosen valid time.

    For each member of each baseline at a given init, recentre the MSL field
    on the member's predicted MSL minimum (within a search box), bilinearly
    interpolate onto a fixed +-10 deg storm-relative grid, then mean across
    members. Shows the typical TC structure each baseline produces,
    independent of position bias. (Marks&Houze 1987 / Rogers 2013 standard.)
    """
    if baselines is None:
        baselines = BASELINES
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 260, 290, 12, 38
    # Storm-relative grid: +-10 deg, 0.25 deg = 81 x 81 cells.
    rel_lat = np.arange(-10.0, 10.25, 0.25)
    rel_lon = np.arange(-10.0, 10.25, 0.25)
    init_t = np.datetime64(
        f"{init_tag[0:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
    )
    valid_t = init_t + np.timedelta64(120, "h")  # lead 120 h ~ 09 Oct 00 UTC

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    composites = {}
    for ax, b in zip(axes.flat, baselines):
        zp = STORE / "baselines" / b / init_tag / "forecast.zarr"
        if not zp.exists():
            ax.text(0.5, 0.5, "no forecast", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(b.replace("_", " "), fontsize=10)
            continue
        try:
            ds = xr.open_zarr(zp, consolidated=True, chunks={})
            if "init_time" in ds.dims:
                ds = ds.isel(init_time=0)
            if "lead_time" in ds.dims:
                lt = ds["lead_time"].values
                if np.issubdtype(lt.dtype, np.timedelta64):
                    leads_valid = init_t + lt
                else:
                    leads_valid = lt
                if valid_t not in leads_valid:
                    ax.text(
                        0.5,
                        0.5,
                        "valid time out of range",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(b.replace("_", " "), fontsize=10)
                    continue
                ds = ds.isel(lead_time=int(np.where(leads_valid == valid_t)[0][0]))
            msl_ens = ds["mean_sea_level_pressure"].load() / 100  # to hPa
            n_mem = msl_ens.sizes.get("ensemble", 1)
            # Per-member: find MSL minimum within search box, then bilinear interp.
            stack = np.full((n_mem, len(rel_lat), len(rel_lon)), np.nan)
            for mi in range(n_mem):
                m = msl_ens.isel(ensemble=mi) if "ensemble" in msl_ens.dims else msl_ens
                m_box = m.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
                if m_box.size == 0:
                    continue
                flat_idx = int(np.argmin(m_box.values))
                ny, nx = m_box.shape
                iy, ix = flat_idx // nx, flat_idx % nx
                ctr_lat = float(m_box["latitude"].values[iy])
                ctr_lon = float(m_box["longitude"].values[ix])
                # Bilinear sample on shifted grid
                target_lat = ctr_lat + rel_lat
                target_lon = ctr_lon + rel_lon
                samp = m.interp(latitude=target_lat, longitude=target_lon, method="linear")
                stack[mi] = samp.values
            comp = np.nanmean(stack, axis=0)
            composites[b] = comp
            cf = ax.contourf(
                rel_lon,
                rel_lat,
                comp,
                levels=np.arange(960, 1020, 3),
                cmap="RdYlBu_r",
                extend="both",
                rasterized=True,
            )
            ax.contour(
                rel_lon,
                rel_lat,
                comp,
                levels=np.arange(960, 1020, 6),
                colors="black",
                linewidths=0.3,
            ).set_rasterized(True)
            ax.plot(0, 0, marker="x", color="black", markersize=8)
            ax.set_xlabel("dlon (deg)", fontsize=8)
            ax.set_ylabel("dlat (deg)", fontsize=8)
            ax.set_aspect("equal")
            ax.set_title(f"{b.replace('_', ' ')}  min={np.nanmin(comp):.1f}hPa", fontsize=10)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"ERR: {type(e).__name__}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(b.replace("_", " "), fontsize=10)

    fig.suptitle(
        f"Storm-relative MSL composite at lead 120 h (init {init_tag}; valid {str(valid_t)[:13]} UTC)",
        fontsize=11,
        y=0.99,
    )
    cbar_ax = fig.add_axes([0.25, -0.03, 0.5, 0.02])
    fig.colorbar(cf, cax=cbar_ax, orientation="horizontal", label="MSL (hPa)")
    fig.tight_layout()
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
    f4_storm_relative_composite()
    f5_track_intensity_err_vs_lead(verif)


if __name__ == "__main__":
    main()
