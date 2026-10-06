"""Milton z500 member plumes for the 2024-10-06 00 UTC initialisation (slide figure).

Four ensembles (SPW AIFS weight+IC, SPW AIFS weight-only, AIFS-ENS, IFS-ENS) as
member plumes of 500 hPa geopotential height (dam), cos-lat-weighted over a
2x2 deg box on the Siesta Key landfall point.
ERA5 verifies every panel; the unperturbed AIFS control (sigma=0, single member,
job 3592400) is drawn on the two SPW panels.

Stage 1 extracts the box means once per source into $STORE/analysis/milton_z500_plume
(small netCDFs); stage 2 plots from those. IFS-ENS whole-field NaN leads (WB2
chunk dropout) are left as gaps, not interpolated.

Each panel carries a signature-kernel skill score of the plotted box-mean path
against persistence (ERA5 at +0 h held constant):

    skill = 1 - D(ens) / D(persistence),  D = ||mean_i phi(X_i) - phi(ERA5)||^2

with phi the signature-kernel feature map, so 100 % is a perfect forecast and 0 %
is persistence. D keeps the member self-pairs (V-statistic), so it is >= 0 and the
skill can never exceed 100 %; unlike the paper's fair SIGK it slightly favours sharp
ensembles. Paper SIGK settings (tools/signature_kernel_score.py): 12-hourly path to
240 h, fixed 1990-2019 z500 scale, RBF sigma=1, dyadic=1, basepoint + time
augmentation. All four panels are scored on the same leads, the 12-hourly leads
where IFS-ENS has data.

Each panel also carries the within-run flip-flop ratio: the flip-flop index of
Griffiths et al. (2019) (as used run-to-run for AICON, arXiv:2608.24651 Sect. 6.3),
FFI = (sum |x_k+1 - x_k| - (max x - min x)) / (n - 2), applied along each member's
lead-time path, averaged over members and divided by the FFI of the ERA5 path. 1 is
as jumpy as ERA5, < 1 smoother, > 1 jumpier. Not a proper score (a flat path has
FFI 0), so it is read next to the SIGK skill. Computed on the 6-hourly leads where
IFS-ENS has data so all panels share the same leads.

Output: figures/slides/milton_z500_plume_box_{median|mean}[_nolegend].{png,pdf}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from _env import BASELINES, DATA, FIGURES, IFS_ENS, STORE, WB2_2024  # noqa: E402
from signature_kernel_score import _sig_kernel_batch  # noqa: E402

INIT = pd.Timestamp("2024-10-06T00:00")
INIT_TAG = "20241006_0000"
LEADS_H = np.arange(0, 241, 6)
LANDFALL_H = 96
G0 = 9.80665
IFS_STRATIFIED = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

CACHE = STORE / "analysis" / "milton_z500_plume"
CONTROL_ZARR = CACHE / f"aifs_control_{INIT_TAG}" / "forecast.zarr"
OUT_DIR = FIGURES / "slides"

# Landfall box: 2x2 deg centred on 27.3N, 82.6W.
BOX = (26.3, 28.3, 360 - 83.6, 360 - 81.6)

ENSEMBLES = {
    "aifs_perturbed_ic": ("SPW AIFS (weight+IC)", "#8E44AD"),
    "aifs_perturbed": ("SPW AIFS (weight only)", "#8E44AD"),
    "aifsens": ("AIFS-ENS", "#8C5A2B"),
    "ifs_ens": ("IFS-ENS", "#7F8C8D"),
}
SPW = ("aifs_perturbed_ic", "aifs_perturbed")
REPORT_LEADS = (24, 96, 168, 240)

# SIGK settings of the paper (tools/submit_table_metrics_fixedscale.sh).
SIGK_STEP_H = 12
SIGK_SIGMA = 1.0
SIGK_DYADIC = 1
Z500_SCALE_DAM = (
    json.loads((DATA / "channel_scale_1990_2019.json").read_text())["geopotential_500"] / G0 / 10
)


def _box(z: xr.DataArray) -> xr.DataArray:
    la0, la1, lo0, lo1 = BOX
    return z.sel(latitude=slice(la1, la0), longitude=slice(lo0, lo1))


def _box_mean(z: xr.DataArray) -> xr.Dataset:
    """z: geopotential (m2 s-2) on the box (..., latitude, longitude) -> cos-lat mean in dam."""
    w = np.cos(np.deg2rad(z["latitude"]))
    box = z.weighted(w).mean(["latitude", "longitude"], skipna=False) / G0 / 10
    return xr.Dataset({"box": box.astype("float64")})


def _lead_index(da: xr.DataArray) -> xr.DataArray:
    lead_h = (da["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    da = da.assign_coords(lead_time=lead_h).sel(lead_time=LEADS_H)
    return da.rename(lead_time="lead_h")


def extract(source: str) -> xr.Dataset:
    path = CACHE / f"{source}_{INIT_TAG}.nc"
    if path.exists():
        return xr.open_dataset(path).load()
    print(f"[extract] {source}")
    if source == "era5":
        era = xr.open_zarr(WB2_2024)["geopotential"].sel(level=500)
        valid = INIT + pd.to_timedelta(LEADS_H, unit="h")
        z = _box(era).sel(time=valid.values).load()
        z = z.assign_coords(time=LEADS_H).rename(time="lead_h")
        ds = _box_mean(z)
    elif source == "ifs_ens":
        z = xr.open_zarr(IFS_ENS, consolidated=False)["geopotential"]
        z = z.sel(init_time=INIT.to_datetime64(), level=500).isel(ensemble=IFS_STRATIFIED)
        z = _box(_lead_index(z)).load()
        nan_frac = z.isnull().mean(["latitude", "longitude"])
        partial = (nan_frac > 0) & (nan_frac < 1)
        if bool(partial.any()):
            raise RuntimeError(
                f"IFS-ENS has partially-NaN fields: {nan_frac.where(partial, drop=True)}"
            )
        ds = _box_mean(z)
        gone = ds["box"].isnull().all("ensemble")
        if bool((ds["box"].isnull() != gone).any()):
            raise RuntimeError("IFS-ENS NaN leads differ across members")
        ds["missing"] = gone
    elif source == "control":
        z = xr.open_zarr(CONTROL_ZARR, consolidated=True)["geopotential"]
        z = z.sel(level=500).isel(init_time=0, ensemble=0)
        ds = _box_mean(_box(_lead_index(z)).load())
    else:
        z = xr.open_zarr(BASELINES / source / INIT_TAG / "forecast.zarr", consolidated=True)[
            "geopotential"
        ]
        z = z.sel(level=500).isel(init_time=0)
        ds = _box_mean(_box(_lead_index(z)).load())
    CACHE.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return ds


def flip_flop(x: np.ndarray) -> float:
    """Griffiths et al. (2019) flip-flop index of a sequence, in its units."""
    return float((np.abs(np.diff(x)).sum() - (x.max() - x.min())) / (len(x) - 2))


def _centre(m: np.ndarray, kind: str) -> np.ndarray:
    return np.median(m, axis=0) if kind == "median" else m.mean(axis=0)


def _isolated(y: np.ndarray) -> np.ndarray:
    """Finite points with no finite neighbour; a line plot would not show them."""
    f = np.isfinite(y)
    return f & ~np.r_[False, f[:-1]] & ~np.r_[f[1:], False]


def _plot_gappy(ax, x, y, ms, **kw):
    ax.plot(x, y, **kw)
    iso = _isolated(y)
    if iso.any():
        ax.plot(
            x[iso], y[iso], ls="none", marker="o", ms=ms, color=kw["color"], alpha=kw.get("alpha")
        )


def sigk_dist(members: np.ndarray, truth: np.ndarray, lead_h: np.ndarray) -> tuple[float, float]:
    """Squared signature-kernel distance D of member paths (M, L) to the truth path (L), in dam.

    D = mean_ij K(X_i, X_j) - 2 mean_i K(X_i, o) + K(o, o): the paper's SIGK
    (tools/signature_kernel_score.py, Eq. 1) with the self-pairs i=j kept and the
    constant K(o, o) added back, so D >= 0 and D = 0 only if every member equals the
    truth. The time channel uses the actual lead hours so dropped leads keep their
    timing. Returns (D, leave-one-member-out jackknife SE).
    """
    t = np.r_[0.0, (lead_h + SIGK_STEP_H) / (LEADS_H[-1] + SIGK_STEP_H)]

    def path(v):
        return np.stack([t, np.r_[0.0, v / Z500_SCALE_DAM]], axis=-1)

    n = members.shape[0]
    mem = np.stack([path(v) for v in members])
    obs = path(truth)[None]
    i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    k_mm = _sig_kernel_batch(mem[i], mem[j], SIGK_SIGMA, SIGK_DYADIC).reshape(n, n)
    k_mo = _sig_kernel_batch(mem, np.repeat(obs, n, axis=0), SIGK_SIGMA, SIGK_DYADIC)
    k_oo = _sig_kernel_batch(obs, obs, SIGK_SIGMA, SIGK_DYADIC)[0]

    def dist(keep):
        return k_mm[np.ix_(keep, keep)].mean() - 2 * k_mo[keep].mean() + k_oo

    if n == 1:
        return float(dist([0])), 0.0
    loo = np.array([dist(np.delete(np.arange(n), d)) for d in range(n)])
    se = np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())
    return float(dist(np.arange(n))), float(se)


def plot(data, era5, control, skill, ffi, centre: str, legend: bool, stem: str):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (key, (title, colour)) in zip(axes, ENSEMBLES.items()):
        m = data[key]["box"].transpose("ensemble", "lead_h").values
        x = LEADS_H
        for row in m:
            _plot_gappy(ax, x, row, 3, color=colour, lw=1.0, alpha=0.35)
        ax.fill_between(
            x,
            np.percentile(m, 10, axis=0),
            np.percentile(m, 90, axis=0),
            color=colour,
            alpha=0.15,
            lw=0,
        )
        _plot_gappy(ax, x, _centre(m, centre), 6, color=colour, lw=2.8)
        ax.plot(x, era5["box"].values, color="#000000", lw=2.5)
        if key in SPW:
            ax.plot(x, control["box"].values, color="#000000", lw=1.8, ls="--")
        ax.axvline(LANDFALL_H, color="grey", ls=":", lw=1.5)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            LANDFALL_H + 3,
            0.98,
            "landfall",
            transform=trans,
            ha="left",
            va="top",
            fontsize=14,
            color="grey",
        )
        ax.text(
            0.5,
            1.015,
            f"SIGK skill = {100 * skill[key][0]:.0f}%\nflip-flop vs ERA5 = {ffi[key]:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            linespacing=1.3,
        )
        ax.set_title(title, fontsize=16, fontweight="bold", pad=48)
        ax.set_xlabel("lead time (h)", fontsize=16)
        ax.set_xticks(np.arange(0, 241, 48))
        ax.set_xlim(0, 240)
        ax.tick_params(labelsize=14)
        ax.grid(True, color="lightgrey", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("z500 (dam)", fontsize=16)
    fig.tight_layout()
    if legend:
        c = ENSEMBLES["aifs_perturbed_ic"][1]
        handles = [
            Line2D([], [], color=c, lw=1.0, alpha=0.35),
            Patch(facecolor=c, alpha=0.15, lw=0),
            Line2D([], [], color=c, lw=2.8),
            Line2D([], [], color="#000000", lw=2.5),
            Line2D([], [], color="#000000", lw=1.8, ls="--"),
        ]
        labels = ["member", "10-90 %", f"ensemble {centre}", "ERA5", "deterministic control"]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=5,
            frameon=False,
            fontsize=14,
        )
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"{stem}.{ext}",
            dpi=200,
            bbox_inches="tight",
            pad_inches=4 / 200,
            facecolor="white",
        )
    plt.close(fig)


def report(data, era5, skill, d_pers, sigk_leads, ffi, ffi_era5, ffi_leads):
    print("\n=== box: spread (std, ddof=1) | median-ERA5 | max-min, dam ===")
    print(f"{'ensemble':24s}" + "".join(f"{f'+{h}h':>24s}" for h in REPORT_LEADS))
    for key, (title, _) in ENSEMBLES.items():
        da = data[key]["box"]
        cells = []
        for h in REPORT_LEADS:
            v = da.sel(lead_h=h).values
            if np.isnan(v).any():
                cells.append("missing")
                continue
            bias = np.median(v) - float(era5["box"].sel(lead_h=h))
            cells.append(f"{v.std(ddof=1):6.2f} {bias:+7.2f} {v.max() - v.min():6.2f}")
        print(f"{title:24s}" + "".join(f"{c:>24s}" for c in cells))

    print(
        f"\n=== SIGK skill vs persistence (D_pers = {d_pers:.4f}) on {len(sigk_leads)} leads: "
        f"{sigk_leads.tolist()} ==="
    )
    for key, (sk, se) in skill.items():
        title = ENSEMBLES[key][0] if key in ENSEMBLES else key
        print(f"{title:24s} {100 * sk:6.1f}% +- {100 * se:.1f}% (jackknife SE)")

    print(
        f"\n=== within-run flip-flop ratio vs ERA5 (ERA5 FFI {ffi_era5:.3f} dam) on {len(ffi_leads)} leads ==="
    )
    for key, r in ffi.items():
        print(f"{ENSEMBLES[key][0] if key in ENSEMBLES else key:24s} {r:.3f}")

    print("\n=== member ordering (leads 6-240 h) ===")
    for key in SPW:
        m = data[key]["box"].sel(lead_h=slice(6, 240)).transpose("ensemble", "lead_h").values
        ranks = m.argsort(axis=0).argsort(axis=0)
        consec = [spearmanr(ranks[:, i], ranks[:, i + 1])[0] for i in range(ranks.shape[1] - 1)]
        r24 = ranks[:, list(LEADS_H[1:]).index(24)]
        same_as_24 = int((ranks == r24[:, None]).all(axis=0).sum())
        rho_24_240 = spearmanr(r24, ranks[:, -1])[0]
        # pairwise crossings: sign changes of member differences between consecutive leads
        d = m[:, None, :] - m[None, :, :]
        iu = np.triu_indices(m.shape[0], 1)
        crossings = int((np.diff(np.sign(d[iu]), axis=1) != 0).sum())
        print(
            f"{ENSEMBLES[key][0]:24s} rho(consecutive) mean={np.mean(consec):.3f} "
            f"min={np.min(consec):.3f} | rho(+24,+240)={rho_24_240:+.3f} | "
            f"leads with +24h ordering {same_as_24}/{ranks.shape[1]} | pair crossings {crossings}"
        )


def main():
    data = {k: extract(k) for k in ENSEMBLES}
    era5 = extract("era5")
    control = extract("control")
    print(f"IFS-ENS missing leads (h): {LEADS_H[data['ifs_ens']['missing'].values].tolist()}")
    grid = np.arange(0, LEADS_H[-1] + 1, SIGK_STEP_H)
    sigk_leads = grid[data["ifs_ens"]["box"].sel(lead_h=grid).notnull().all("ensemble").values]
    obs = era5["box"].sel(lead_h=sigk_leads).values
    paths = {
        k: data[k]["box"].sel(lead_h=sigk_leads).transpose("ensemble", "lead_h").values
        for k in ENSEMBLES
    }
    paths["AIFS control"] = control["box"].sel(lead_h=sigk_leads).values[None]
    d_pers = sigk_dist(np.full((1, obs.size), obs[0]), obs, sigk_leads)[0]
    skill = {}
    for key, m in paths.items():
        d, se = sigk_dist(m, obs, sigk_leads)
        skill[key] = (1 - d / d_pers, se / d_pers)
    ffi_leads = LEADS_H[data["ifs_ens"]["box"].notnull().all("ensemble").values]
    ffi_era5 = flip_flop(era5["box"].sel(lead_h=ffi_leads).values)
    ffi_paths = {
        k: data[k]["box"].sel(lead_h=ffi_leads).transpose("ensemble", "lead_h").values
        for k in ENSEMBLES
    }
    ffi_paths["AIFS control"] = control["box"].sel(lead_h=ffi_leads).values[None]
    ffi = {k: np.mean([flip_flop(x) for x in m]) / ffi_era5 for k, m in ffi_paths.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for centre in ("median", "mean"):
        for legend in (True, False):
            stem = f"milton_z500_plume_box_{centre}{'' if legend else '_nolegend'}"
            plot(data, era5, control, skill, ffi, centre, legend, stem)
    report(data, era5, skill, d_pers, sigk_leads, ffi, ffi_era5, ffi_leads)


if __name__ == "__main__":
    main()
