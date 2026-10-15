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

The styling helpers here are shared with tools/plot_point_ensemble.py.

Usage: python tools/plot_ens_anom_gaussianity.py [--lead 240]
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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
QQ_PROBS = norm.cdf(np.linspace(-3.3, 3.3, 121))
Z_POOL = "pooled over 28 inits x 2048 points x 10 members"
A_EQ = r"$a_m = x_m - \bar{x}$"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=240)
    args = ap.parse_args()

    data, nulls = {}, {}
    for name in SPW + TRAINED + [CLIM]:
        f = np.load(INDIR / f"{name}.npz")
        labels = list(f["labels"])
        ks = [labels.index(v) for v in FEATURE]
        li = int(np.flatnonzero(f["lead_hours"] == args.lead)[0])
        data[name] = tuple(f[key][ks, li] for key in ("z", "std", "mean"))
        seed = NULL_SEED + 3 if name == CLIM else NULL_SEED
        nulls[name] = gaussian_null(data[name][0].shape, seed)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(BIG_RC):
        for kind in ("values", "qq"):
            fig, axes = plt.subplots(1, len(FEATURE), figsize=(20, 7.4))
            for k, (ax, var) in enumerate(zip(axes, FEATURE, strict=True)):
                scale, unit = phys(var)
                ax.set_title(VAR_SHORT[var])
                if kind == "values":
                    panel_values(ax, data, k, scale)
                    ax.set_xlabel(f"{VAR_SHORT[var]} [{unit}]")
                else:
                    panel_qq(ax, data, nulls, k, scale)
                    ax.set_xlabel("standard normal quantile")
                    ax.set_ylabel(f"quantile minus Gaussian [{unit}]")
            if kind == "values":
                axes[0].set_ylabel("density")
                title = f"Ensemble member values at {args.lead} h"
                sub = (
                    r"$x_m$: member values, " + Z_POOL + ";   ERA5 climatology: 1990-2019, same "
                    "calendar day and hour, years as members"
                )
                stem = f"member_values_{args.lead}h"
            else:
                title = f"Member anomalies at {args.lead} h: detrended Q-Q against Gaussian members"
                sub = (
                    f"{A_EQ},  " + r"$\bar{x}$" + f": mean over the 10 members per gridpoint;   "
                    f"{Z_POOL};   0 = Gaussian members with the same per-gridpoint std"
                )
                stem = f"member_anomaly_qq_{args.lead}h"
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
