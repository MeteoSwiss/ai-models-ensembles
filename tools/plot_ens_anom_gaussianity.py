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
      share of the single ensembles (one init, one gridpoint) whose 10 members reject
      Gaussianity in a Shapiro-Wilk test at 5%, against lead time. Each test sees one
      forecast only, so a mixture over inits cannot raise it. References: Gaussian members
      (the test's own size) and three non-Gaussian member distributions at the same M.

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
from scipy.stats import norm, shapiro

from _env import FIGURES, STORE
from model_colors import color_for, style_for

sys.stdout.reconfigure(line_buffering=True)

INDIR = STORE / "analysis" / "ens_anom_gaussianity"
FIGDIR = FIGURES / "ens_anom"
SPW = ["aurora_encoder", "graphcast_all", "sfno_modes10", "aifs_perturbed"]
TRAINED = ["aifsens", "atlas", "fcn3"]
CLIM = "era5_clim"
LABEL = {
    "aurora_encoder": "Aurora SPW",
    "graphcast_all": "GraphCast SPW",
    "sfno_modes10": "SFNO SPW",
    "aifs_perturbed": "AIFS SPW",
    "aifsens": "AIFS-ENS",
    "atlas": "Atlas",
    "fcn3": "FCN3",
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
SW_SIMS = 200_000
SW_BOOT = 2000
SW_REFS = [  # non-Gaussian member distributions: (label, sampler(rng, shape))
    ("Laplace members", lambda rng, n: rng.laplace(size=n)),
    ("gamma k=4 members", lambda rng, n: rng.gamma(4.0, size=n)),
    (
        "two clusters at ±2 std",
        lambda rng, n: rng.standard_normal(n) + np.where(rng.random(n) < 0.5, -2.0, 2.0),
    ),
]
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


def model_line(ax, name, x, y, lw: float = 1.8) -> None:
    if name == CLIM:
        ax.plot(x, y, color="black", ls="--", lw=lw - 0.2, label=LABEL[name])
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


def sw_reject_rate(z: np.ndarray) -> np.ndarray:
    """z (nlead, ninit, M, npts) -> (nlead, ninit) share of points with Shapiro p < SW_ALPHA."""
    p = shapiro(np.moveaxis(z, 2, -1), axis=-1).pvalue
    return np.nanmean(np.where(np.isfinite(p), p < SW_ALPHA, np.nan), axis=-1)


def plot_shapiro_rate() -> None:
    rng = np.random.default_rng(NULL_SEED)
    rates, leads = {}, None
    for name in SPW + TRAINED + [CLIM]:
        f = np.load(INDIR / f"{name}.npz")
        labels = list(f["labels"])
        z = f["z"]
        rates[name] = np.stack([sw_reject_rate(z[labels.index(v)]) for v in FEATURE])
        leads, M = f["lead_hours"], z.shape[3]
        del z
        print(f"  {name}: {np.round(100 * rates[name][:, -1].mean(-1), 1)} % at {leads[-1]} h")
    size = np.mean(shapiro(rng.standard_normal((SW_SIMS, M)), axis=-1).pvalue < SW_ALPHA)
    refs = [
        (lab, np.mean(shapiro(draw(rng, (SW_SIMS, M)), axis=-1).pvalue < SW_ALPHA))
        for lab, draw in SW_REFS
    ]
    print(
        f"  Gaussian members {100 * size:.2f} %, "
        + ", ".join(f"{a} {100 * r:.1f} %" for a, r in refs)
    )

    with plt.rc_context(BIG_RC):
        fig, axes = plt.subplots(1, len(FEATURE), figsize=(20, 7.4), sharey=True)
        for k, (ax, var) in enumerate(zip(axes, FEATURE, strict=True)):
            ax.axhline(100 * size, color=NULL_COLOR, lw=3.6, label="Gaussian members")
            for lab, r in refs:
                ax.axhline(100 * r, color=NULL_COLOR, lw=1.2, ls=":")
                if k == len(FEATURE) - 1:
                    ax.annotate(
                        f"{lab} ({100 * r:.0f}%)",
                        xy=(1, 100 * r),
                        xycoords=("axes fraction", "data"),
                        xytext=(6, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=12,
                        color="#5F6368",
                    )
            for name, rate in rates.items():
                r = rate[k]
                boot = rng.integers(0, r.shape[1], (SW_BOOT, r.shape[1]))
                lo, hi = np.percentile(r[:, boot].mean(-1), [2.5, 97.5], axis=1)
                color = "black" if name == CLIM else color_for(name)
                ax.fill_between(leads, 100 * lo, 100 * hi, color=color, alpha=0.12, lw=0)
                model_line(ax, name, leads, 100 * r.mean(-1), lw=2.4)
            ax.set_title(VAR_SHORT[var])
            ax.set_xlabel("lead time [h]")
            ax.set_xticks(leads[1::2])
            style_axes(ax)
        axes[0].set_ylabel(f"ensembles rejecting Gaussianity at {SW_ALPHA:.0%} [%]")
        axes[0].set_ylim(0, None)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=len(handles), frameon=False)
        fig.suptitle(
            f"Shapiro-Wilk test on each single ensemble ({M} members at one init and one gridpoint)"
        )
        fig.text(
            0.5,
            0.905,
            f"share of the 28 inits x 2048 cos-lat points that reject at {SW_ALPHA:.0%};   "
            "band: 95% bootstrap over inits;   dotted: the same test on non-Gaussian members "
            f"(M = {M})",
            ha="center",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0.07, 0.9, 0.9))
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
