"""Ensemble member distributions vs a Gaussian and ERA5 climatology (T2m, Q850, Z500).

Two large 3-panel figures at one lead from tools/compute_ens_anomalies.py (28 inits x
2048 cos-lat points x 10 members per baseline, plus the ERA5 probabilistic climatology
era5_clim):

  figures/ens_anom/member_values_<lead>h.{png,pdf}
      density of the member values x_m = mean + z * std in physical units.
  figures/ens_anom/member_anomaly_qq_<lead>h.{png,pdf}
      detrended Q-Q of the member anomaly a = x_m - ensemble mean against Gaussian
      members with the same per-gridpoint std: M iid N(0,1) draws standardized like the
      forecasts (sample mean, ddof=1 std) and rescaled by that std.
  figures/ens_anom/member_zscore_qq_<lead>h.{png,pdf}
      the same for the standardized anomaly z = (x_m - mean) / std against those Gaussian
      members standardized the same way. z removes each forecast's location and scale, so
      Gaussian members at every gridpoint and init give exactly 0 even after pooling.
  figures/ens_anom/member_shapiro_rate.{png,pdf}
      share of the single ensembles (one init, one gridpoint) whose members are non-Gaussian
      in a Shapiro-Wilk test, Benjamini-Hochberg FDR 5% over all 112 inits x 8192 points
      per lead and variable (the --tag sw sample; Gaussian members give 0). Each test sees
      one forecast only, so a mixture over inits cannot raise it. IFS-ENS with all 50
      members and with 10 shows the effect of M on the test's power. The bootstrap band
      resamples inits at the FDR threshold of the full sample.

The styling helpers here are shared with tools/plot_point_ensemble.py.

Usage: python tools/plot_ens_anom_gaussianity.py [--lead 240] [--shapiro-only]
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import false_discovery_control, norm, shapiro

from _env import FIGURES, STORE
from model_colors import color_for, style_for

sys.stdout.reconfigure(line_buffering=True)

INDIR = STORE / "analysis" / "ens_anom_gaussianity"
FIGDIR = FIGURES / "ens_anom"
SPW = ["aurora_encoder", "graphcast_all", "sfno_modes10", "aifs_perturbed"]
TRAINED = ["aifsens", "atlas", "fcn3"]
IFS = ["ifs_ens", "ifs_ens10"]
CLIM = "era5_clim"
LABEL = {
    "aurora_encoder": "Aurora SPW",
    "graphcast_all": "GraphCast SPW",
    "sfno_modes10": "SFNO SPW",
    "aifs_perturbed": "AIFS SPW",
    "aifsens": "AIFS-ENS",
    "atlas": "Atlas",
    "fcn3": "FCN3",
    "ifs_ens": "IFS-ENS, 50 members",
    "ifs_ens10": "IFS-ENS, 10 members",
    CLIM: "ERA5 climatology",
}
FEATURE = ["2m_temperature", "specific_humidity_850", "geopotential_500"]
VAR_SHORT = {"2m_temperature": "T2m", "specific_humidity_850": "Q850", "geopotential_500": "Z500"}
# native units -> display units; Z stays geopotential (see .claude/rules/plot-units.md)
PHYS = {
    "2m_temperature": (1.0, "K"),
    "specific_humidity": (1000.0, "g/kg"),
    "geopotential": (1.0, "m²/s²"),
}
NULL_SEED = 1234
NULL_COLOR = "#9AA0A6"
SW_ALPHA = 0.05
SW_Q = 0.05
SW_TAG = "sw"  # compute_ens_anomalies.py --tag sw: all 112 inits x 8192 points, Shapiro-Wilk p
SW_BOOT = 2000
QQ_PROBS = norm.cdf(np.linspace(-3.3, 3.3, 121))
Z_POOL = "pooled over 28 inits x 2048 points x 10 members"
A_EQ = r"$a_m = x_m - \bar{x}$"
Z_EQ = r"$z_m = (x_m - \bar{x})\,/\,s_x$"
BIG_RC = {
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
}


def phys(label: str) -> tuple[float, str]:
    base, _, lev = label.rpartition("_")
    return PHYS[base if lev.isdigit() else label]


def gaussian_null(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Standardized anomalies of M iid N(0,1) members; members on axis -2."""
    x = np.random.default_rng(seed).standard_normal(shape)
    mu = x.mean(axis=-2, keepdims=True)
    sd = x.std(axis=-2, ddof=1, keepdims=True)
    return ((x - mu) / sd).astype(np.float32)


def style_axes(ax) -> None:
    ax.grid(True, color="#E0E0E0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def line_color(name: str) -> str:
    return "black" if name == CLIM else color_for(IFS[0] if name in IFS else name)


def model_line(ax, name, x, y, lw: float = 1.8) -> None:
    if name == CLIM:
        ax.plot(x, y, color="black", ls="--", lw=lw - 0.2, label=LABEL[name])
    elif name in IFS:
        ls = "-" if name == IFS[0] else ":"
        ax.plot(x, y, color=line_color(name), ls=ls, lw=lw, label=LABEL[name])
    else:
        ax.plot(x, y, color=color_for(name), ls=style_for(name), lw=lw, label=LABEL[name])


def panel_values(ax, data, k: int, scale: float) -> None:
    """Pooled density of x_m = mean + z * std. data: {name: (z, std, mean)} per FEATURE row."""
    arrs = {}
    for name, (z, sd, mu) in data.items():
        x = ((mu[k][:, None, :] + z[k] * sd[k][:, None, :]) * scale).ravel()
        arrs[name] = x[np.isfinite(x)]
    lo = min(np.quantile(x, 0.001) for x in arrs.values())
    hi = max(np.quantile(x, 0.999) for x in arrs.values())
    bins = np.linspace(lo, hi, 101)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    for name, x in arrs.items():
        h, _ = np.histogram(x, bins=bins, density=True)
        model_line(ax, name, ctr, h, lw=2.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    style_axes(ax)


def panel_qq(ax, data, nulls, k: int, scale: float) -> None:
    """Detrended Q-Q of a = z * std against Gaussian members with the same per-gridpoint std."""
    x = norm.ppf(QQ_PROBS)
    ax.axhline(0, color=NULL_COLOR, lw=3.6, label="Gaussian, same spread")
    for name, (z, sd, _) in data.items():
        s = sd[k][:, None, :] * scale
        a = (z[k] * s).ravel()
        an = (nulls[name][k] * s).ravel()
        ok = np.isfinite(a)
        y = np.quantile(a[ok], QQ_PROBS) - np.quantile(an[ok], QQ_PROBS)
        model_line(ax, name, x, y, lw=2.4)
    lim = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-lim, lim)
    style_axes(ax)


def panel_qq_z(ax, data, nulls, k: int) -> None:
    """Detrended Q-Q of the standardized anomaly z against standardized Gaussian members."""
    x = norm.ppf(QQ_PROBS)
    ax.axhline(0, color=NULL_COLOR, lw=3.6, label="Gaussian members")
    for name, (z, _, _) in data.items():
        a = z[k].ravel()
        ok = np.isfinite(a)
        y = np.quantile(a[ok], QQ_PROBS) - np.quantile(nulls[name][k].ravel(), QQ_PROBS)
        model_line(ax, name, x, y, lw=2.4)
    lim = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-lim, lim)
    style_axes(ax)


def bh_share(p: np.ndarray) -> np.ndarray:
    """p (ninit, npts) -> (ninit,) share of points discovered by Benjamini-Hochberg at SW_Q.

    The FDR family is every finite test in p; inits without data (IFS-ENS gaps) give NaN.
    """
    ok = np.isfinite(p)
    d = np.zeros(p.shape)
    d[ok] = false_discovery_control(p[ok]) < SW_Q
    n = ok.sum(axis=1)
    return np.divide(d.sum(axis=1), n, out=np.full(n.shape, np.nan), where=n > 0)


def plot_shapiro_rate() -> None:
    rng = np.random.default_rng(NULL_SEED)
    rates = {}
    for name in SPW + TRAINED + IFS + [CLIM]:
        f = np.load(INDIR / f"{name}_{SW_TAG}.npz")
        labels, p = list(f["labels"]), f["p"]
        rates[name] = np.stack(
            [[bh_share(p[labels.index(v), li]) for li in range(p.shape[1])] for v in FEATURE]
        )
        if name == SPW[0]:
            leads, n_init, npts = f["lead_hours"], f["inits"].size, p.shape[-1]
        print(f"  {name}: {np.round(100 * np.nanmean(rates[name][:, -1], -1), 2)} % at 360 h")
    p0 = shapiro(rng.standard_normal((n_init * npts, 10)), axis=-1).pvalue
    size = np.mean(false_discovery_control(p0) < SW_Q)
    print(f"  Gaussian members {100 * size:.3f} %")

    with plt.rc_context(BIG_RC):
        fig, axes = plt.subplots(1, len(FEATURE), figsize=(20, 7.8), sharey=True)
        for k, (ax, var) in enumerate(zip(axes, FEATURE, strict=True)):
            for name, rate in rates.items():
                r = rate[k]
                boot = rng.integers(0, r.shape[1], (SW_BOOT, r.shape[1]))
                lo, hi = np.percentile(np.nanmean(r[:, boot], -1), [2.5, 97.5], axis=1)
                ax.fill_between(leads, 100 * lo, 100 * hi, color=line_color(name), alpha=0.12, lw=0)
                model_line(ax, name, leads, 100 * np.nanmean(r, -1), lw=2.4)
            ax.set_title(VAR_SHORT[var])
            ax.set_xlabel("lead time [h]")
            ax.set_xticks(leads[1::2])
            style_axes(ax)
        axes[0].set_ylabel(f"ensembles non-Gaussian at FDR {SW_Q:.0%} [%]")
        axes[0].set_yscale("symlog", linthresh=1.0)  # 10-member shares are < 2%
        axes[0].set_ylim(0, 100)
        ticks = [0, 0.5, 1, 2, 5, 10, 20, 50, 100]
        axes[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
        axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=5, frameon=False)
        fig.suptitle("How often is a single ensemble non-Gaussian?")
        fig.text(
            0.5,
            0.905,
            "Shapiro-Wilk test on the members at one init and one gridpoint, Benjamini-Hochberg "
            f"FDR {SW_Q:.0%} over the {n_init} inits x {npts} points per lead;   band: 95% "
            f"bootstrap over inits;   10 Gaussian members: {100 * size:.0f}%",
            ha="center",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0.1, 1, 0.9))
        for ext in ("png", "pdf"):
            out = FIGDIR / f"member_shapiro_rate.{ext}"
            fig.savefig(out, dpi=200)
            print(f"saved {out}")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=240)
    ap.add_argument("--shapiro-only", action="store_true")
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plot_shapiro_rate()
    if args.shapiro_only:
        return

    data, nulls = {}, {}
    for name in SPW + TRAINED + [CLIM]:
        f = np.load(INDIR / f"{name}.npz")
        labels = list(f["labels"])
        ks = [labels.index(v) for v in FEATURE]
        li = int(np.flatnonzero(f["lead_hours"] == args.lead)[0])
        data[name] = tuple(f[key][ks, li] for key in ("z", "std", "mean"))
        seed = NULL_SEED + 3 if name == CLIM else NULL_SEED
        nulls[name] = gaussian_null(data[name][0].shape, seed)

    with plt.rc_context(BIG_RC):
        for kind in ("values", "qq", "qq_z"):
            fig, axes = plt.subplots(1, len(FEATURE), figsize=(20, 7.4))
            for k, (ax, var) in enumerate(zip(axes, FEATURE, strict=True)):
                scale, unit = phys(var)
                ax.set_title(VAR_SHORT[var])
                if kind == "values":
                    panel_values(ax, data, k, scale)
                    ax.set_xlabel(f"{VAR_SHORT[var]} [{unit}]")
                elif kind == "qq":
                    panel_qq(ax, data, nulls, k, scale)
                    ax.set_xlabel("standard normal quantile")
                    ax.set_ylabel(f"quantile minus Gaussian [{unit}]")
                else:
                    panel_qq_z(ax, data, nulls, k)
                    ax.set_xlabel("standard normal quantile")
                    ax.set_ylabel("quantile minus Gaussian [std]")
            if kind == "values":
                axes[0].set_ylabel("density")
                title = f"Ensemble member values at {args.lead} h"
                sub = (
                    r"$x_m$: member values, " + Z_POOL + ";   ERA5 climatology: 1990-2019, same "
                    "calendar day and hour, years as members"
                )
                stem = f"member_values_{args.lead}h"
            elif kind == "qq":
                title = f"Member anomalies at {args.lead} h: detrended Q-Q against Gaussian members"
                sub = (
                    f"{A_EQ},  " + r"$\bar{x}$" + f": mean over the 10 members per gridpoint;   "
                    f"{Z_POOL};   0 = Gaussian members with the same per-gridpoint std"
                )
                stem = f"member_anomaly_qq_{args.lead}h"
            else:
                title = (
                    f"Standardized member anomalies at {args.lead} h: detrended Q-Q against "
                    "Gaussian members"
                )
                sub = (
                    f"{Z_EQ},  " + r"$\bar{x}$, $s_x$" + ": mean and std over the 10 members per "
                    f"gridpoint;   {Z_POOL};   0 = 10 Gaussian members standardized the same way"
                )
                stem = f"member_zscore_qq_{args.lead}h"
            handles, names = axes[0].get_legend_handles_labels()
            fig.legend(handles, names, loc="lower center", ncol=len(handles), frameon=False)
            fig.suptitle(title)
            fig.text(0.5, 0.905, sub, ha="center", fontsize=13)
            fig.tight_layout(rect=(0, 0.07, 1, 0.9))
            for ext in ("png", "pdf"):
                out = FIGDIR / f"{stem}.{ext}"
                fig.savefig(out, dpi=200)
                print(f"saved {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()
