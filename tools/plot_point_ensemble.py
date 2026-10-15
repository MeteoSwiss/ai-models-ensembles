"""Member-value KDE grids (T2m, Q850, Z500) at single gridpoints and in latitude bands.

Rows of 3 KDE panels with the ERA5 probabilistic climatology (1990-2019, same calendar
day and hour, years as members):

  figures/ens_anom/point_grid_values_<lead>h.{png,pdf}
      POINTS, rows north to south, all 112 inits x 10 members per model
      (tools/compute_point_ensemble.py).
  figures/ens_anom/point_grid_opposite_values_<lead>h.{png,pdf}
      OPPOSITE_POINTS: the same latitudes on the opposite meridians.
  figures/ens_anom/band_grid_values_<lead>h.{png,pdf}
      four equal-area latitude bands pooled over the cos-lat point sample of
      tools/compute_ens_anomalies.py (28 inits), subsampled to BAND_MAX for the KDE.
  figures/ens_anom/point_members_<point>_<init>_<lead>h.{png,pdf}
      the 10 member values of each model at one point and one init on a value axis.
  figures/ens_anom/point_member_qq_<point>_<init>_<lead>h.{png,pdf}
      normal Q-Q of the members per model, IFS-ENS (50 and 10 members) and the 30-year ERA5
      climatology. One global envelope test per panel against Gaussian samples of the same
      size (the ensemble's own mean and std), Benjamini-Hochberg over the panels; the band is
      drawn at the BH level, so a member outside it marks a panel non-Gaussian at FDR 5%.

Usage: python tools/plot_point_ensemble.py [--lead 240] [--member-point zurich --member-init 20231005_0000]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import false_discovery_control, gaussian_kde

from plot_ens_anom_gaussianity import (
    BIG_RC,
    CLIM,
    FEATURE,
    FIGDIR,
    IFS,
    INDIR,
    LABEL,
    NULL_COLOR,
    NULL_SEED,
    SPW,
    SW_Q,
    TRAINED,
    VAR_SHORT,
    gaussian_null,
    line_color,
    model_line,
    phys,
    style_axes,
)

SEED = 2026
QQ_SIMS = 200_000
SIG_COLOR = "#B00020"
POINTS = [
    ("ny_alesund", "Ny-Ålesund"),
    ("zurich", "Zurich"),
    ("equator_pacific", "Equatorial Pacific"),
    ("mcmurdo", "McMurdo"),
]
OPPOSITE_POINTS = [
    ("arctic_168w", "Arctic Ocean"),
    ("npacific_171w", "North Pacific"),
    ("equator_0e", "Gulf of Guinea"),
    ("antarctic_13w", "Antarctica, 1880 m"),
]
BANDS = [  # equal-area quarters of the globe, north to south
    ("NH extratropics", 30, 90),
    ("NH tropics", 0, 30),
    ("SH tropics", -30, 0),
    ("SH extratropics", -90, -30),
]
BAND_MAX = 50_000


def coords(f) -> str:
    lat, lon = float(f["lat"]), float(f["lon"]) % 360
    return (
        f"{abs(lat):g}°{'N' if lat >= 0 else 'S'}, "
        f"{lon if lon <= 180 else 360 - lon:g}°{'E' if lon <= 180 else 'W'}"
    )


def point_values(f, var: str, li: int) -> dict[str, np.ndarray]:
    k = list(f["labels"]).index(var)
    scale, _ = phys(var)
    names = [n for n in SPW + TRAINED + [CLIM] if n in f.files]
    return {n: f[n][k, li].astype(np.float64).ravel() * scale for n in names}


def values_panel(ax, xs: dict[str, np.ndarray]) -> None:
    lo = min(np.quantile(x, 0.002) for x in xs.values())
    hi = max(np.quantile(x, 0.998) for x in xs.values())
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)
    for n, x in xs.items():
        model_line(ax, n, grid, gaussian_kde(x)(grid), lw=2.4)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(bottom=0)
    style_axes(ax)


def draw_grid(rows, title: str, sub: str, stem: str) -> None:
    """rows: [(label, sublabel, {var: {dataset: values}})], one row each, FEATURE columns."""
    with plt.rc_context(BIG_RC):
        fig, axes = plt.subplots(len(rows), len(FEATURE), figsize=(20, 5.2 * len(rows) + 1.6))
        for r, (label, sublabel, values) in enumerate(rows):
            for c, var in enumerate(FEATURE):
                ax = axes[r, c]
                values_panel(ax, values[var])
                ax.tick_params(axis="y", labelleft=False)
                if r == 0:
                    ax.set_title(f"{VAR_SHORT[var]} [{phys(var)[1]}]", fontsize=19, pad=12)
            axes[r, 0].set_ylabel(sublabel, fontsize=14)
            axes[r, 0].annotate(
                label,
                xy=(-0.1, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="right",
                va="center",
                fontsize=17,
                fontweight="bold",
            )
        handles, hnames = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, hnames, loc="lower center", ncol=len(handles), frameon=False)
        fig.suptitle(title, y=0.995)
        fig.text(0.5, 0.965, sub, ha="center", fontsize=14)
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        FIGDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            out = FIGDIR / f"{stem}.{ext}"
            fig.savefig(out, dpi=200)
            print(f"saved {out}")
        plt.close(fig)


def plot_points(lead: int, points: list[tuple[str, str]], stem: str, note: str = "") -> None:
    rows = []
    for name, label in points:
        f = np.load(INDIR / f"point_{name}.npz")
        li = int(np.flatnonzero(f["lead_hours"] == lead)[0])
        rows.append((label, coords(f), {v: point_values(f, v, li) for v in FEATURE}))
    n_init, M = f[SPW[0]].shape[2:]
    draw_grid(
        rows,
        f"Member values at {lead} h" + (f", {note}" if note else ""),
        f"{n_init} inits x {M} members per model;   ERA5 climatology 1990-2019",
        stem,
    )


def band_label(lo: int, hi: int) -> str:
    def fmt(v: int) -> str:
        return "0°" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

    return f"{fmt(lo)}-{fmt(hi)}" if hi > 0 else f"{fmt(hi)}-{fmt(lo)}"


def plot_bands(lead: int, n_max: int) -> None:
    rng = np.random.default_rng(SEED)
    values = {b: {v: {} for v in FEATURE} for b, _, _ in BANDS}
    n_pts = {}
    for name in SPW + TRAINED + [CLIM]:
        f = np.load(INDIR / f"{name}.npz")
        labels, lat = list(f["labels"]), f["lat"]
        li = int(np.flatnonzero(f["lead_hours"] == lead)[0])
        z, mu, sd = f["z"], f["mean"], f["std"]
        if name == SPW[0]:
            n_init, M = z.shape[2:4]
        for var in FEATURE:
            k = labels.index(var)
            x = (mu[k, li][:, None, :] + z[k, li] * sd[k, li][:, None, :]) * phys(var)[0]
            for band, lo, hi in BANDS:
                mask = (lat >= lo) & ((lat < hi) | (hi == 90))
                a = x[..., mask].ravel()
                a = a[np.isfinite(a)]
                if a.size > n_max:
                    a = rng.choice(a, n_max, replace=False)
                values[band][var][name] = a.astype(np.float64)
                if name == SPW[0]:
                    n_pts[band] = int(mask.sum())
        del z, mu, sd
    print("points per band:", n_pts)
    rows = [(b, band_label(lo, hi), values[b]) for b, lo, hi in BANDS]
    draw_grid(
        rows,
        f"Member values at {lead} h in four latitude bands",
        f"{n_init} inits x ~{round(np.mean(list(n_pts.values())))} area-weighted points x {M} "
        f"members per model, subsampled to {n_max:,};   ERA5 climatology 1990-2019",
        f"band_grid_values_{lead}h",
    )


def plot_members(lead: int, point: str, init: str) -> None:
    f = np.load(INDIR / f"point_{point}.npz")
    li = int(np.flatnonzero(f["lead_hours"] == lead)[0])
    j = list(f["inits"]).index(init)
    title = dict(POINTS + OPPOSITE_POINTS)[point]
    labels = list(f["labels"])
    with plt.rc_context(BIG_RC):
        names = SPW + TRAINED
        fig, axes = plt.subplots(1, len(FEATURE), figsize=(20, 5.6), sharey=True)
        for ax, var in zip(axes, FEATURE, strict=True):
            scale, unit = phys(var)
            k = labels.index(var)
            for r, name in enumerate(names):
                x = np.sort(f[name][k, li, j]) * scale
                model_line(ax, name, x, np.full(x.size, r), lw=1.6)
                ax.lines[-1].set_marker("o")
                ax.lines[-1].set_markersize(8)
            ax.set_title(VAR_SHORT[var])
            ax.set_xlabel(f"{VAR_SHORT[var]} [{unit}]")
            style_axes(ax)
        axes[0].set_yticks(range(len(names)), [LABEL[n] for n in names])
        axes[0].set_ylim(len(names) - 0.5, -0.5)
        t0 = f"{init[:4]}-{init[4:6]}-{init[6:8]} {init[9:11]} UTC"
        fig.suptitle(f"10 members, {title} ({coords(f)}), init {t0}, +{lead} h")
        fig.tight_layout()
        FIGDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            out = FIGDIR / f"point_members_{point}_{init}_{lead}h.{ext}"
            fig.savefig(out, dpi=200)
            print(f"saved {out}")
        plt.close(fig)


class EnvelopeNull:
    """Gaussian null of the global envelope test (Myllymaki et al. 2017) for M sorted members.

    Members are standardized by their own mean and std (ddof=1), like QQ_SIMS Gaussian samples
    of the same size. The statistic is the most extreme two-sided pointwise tail probability over
    the order statistics, so a band built from the pointwise null quantiles at one level contains
    a sample exactly when the test does not reject at the matching global level.
    """

    def __init__(self, M: int) -> None:
        z = np.sort(gaussian_null((QQ_SIMS, M, 1), NULL_SEED)[..., 0], axis=1)
        self.cols = np.sort(z, axis=0)  # null of each order statistic
        self.ref = z.mean(axis=0)  # Q-Q x positions
        self.e = np.sort(self.extreme(z))

    def extreme(self, z: np.ndarray) -> np.ndarray:
        n = self.cols.shape[0]
        F = np.stack(
            [np.searchsorted(self.cols[:, i], z[..., i], side="right") for i in range(z.shape[-1])],
            axis=-1,
        )
        F = F / n
        return np.min(2 * np.minimum(F, 1 - F + 1 / n), axis=-1)

    def pvalue(self, x: np.ndarray) -> float:
        z = (np.sort(x) - x.mean()) / x.std(ddof=1)
        k = np.searchsorted(self.e, self.extreme(z), side="right")
        return (1 + k) / (self.e.size + 1)

    def band(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Standardized band that holds a Gaussian sample with probability 1 - alpha."""
        b = np.quantile(self.e, alpha)
        return np.quantile(self.cols, [b / 2, 1 - b / 2], axis=0)


def fmt_p(p: float) -> str:
    return f"{p:.2f}" if p >= 0.01 else f"{p:.0e}"


def plot_member_qq(lead: int, point: str, init: str) -> None:
    f = np.load(INDIR / f"point_{point}.npz")
    li = int(np.flatnonzero(f["lead_hours"] == lead)[0])
    j = list(f["inits"]).index(init)
    title = dict(POINTS + OPPOSITE_POINTS)[point]
    labels = list(f["labels"])
    names = SPW + TRAINED + IFS + [CLIM]
    members = {
        (r, c): f[n][labels.index(v), li, j].astype(np.float64)
        for r, v in enumerate(FEATURE)
        for c, n in enumerate(names)
    }
    nulls = {M: EnvelopeNull(M) for M in {x.size for x in members.values()}}
    pvals = np.full((len(FEATURE), len(names)), np.nan)
    for rc, x in members.items():
        if np.isfinite(x).all():
            pvals[rc] = nulls[x.size].pvalue(x)
    ok = np.isfinite(pvals)
    qvals = np.full(pvals.shape, np.nan)
    qvals[ok] = false_discovery_control(pvals[ok])
    # BH rejects exactly the p <= SW_Q * k / m, so one band level matches its decisions
    alpha = SW_Q * max(int(np.sum(qvals[ok] < SW_Q)), 1) / int(ok.sum())
    bands = {M: null.band(alpha) for M, null in nulls.items()}
    with plt.rc_context(BIG_RC):
        fig, axes = plt.subplots(
            len(FEATURE), len(names), figsize=(38, 13), sharex=True, sharey="row"
        )
        for r, var in enumerate(FEATURE):
            scale, unit = phys(var)
            for c, name in enumerate(names):
                ax = axes[r, c]
                style_axes(ax)
                if r == 0:
                    M = f[name].shape[-1]
                    head = LABEL[name] + (f", {M} years" if name == CLIM else "")
                    ax.set_title(head, color=line_color(name), fontweight="bold")
                if r == len(FEATURE) - 1:
                    ax.set_xlabel("Gaussian quantile")
                if not ok[r, c]:
                    ax.text(
                        0.5,
                        0.5,
                        "no data\n(WB2 gap)",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="#5F6368",
                    )
                    continue
                x = np.sort(members[r, c]) * scale
                ref, (lo, hi) = nulls[x.size].ref, bands[x.size]
                mu, sd = x.mean(), x.std(ddof=1)
                out = (x < mu + sd * lo) | (x > mu + sd * hi)
                ax.fill_between(ref, mu + sd * lo, mu + sd * hi, color=NULL_COLOR, alpha=0.3, lw=0)
                ax.plot(ref, mu + sd * ref, color=NULL_COLOR, lw=2)
                ax.plot(ref, x, "o", color=line_color(name), ms=8, mec="white", mew=0.8)
                ax.plot(ref[out], x[out], "o", mfc="none", mec=SIG_COLOR, ms=13, mew=2.2)
                sig = qvals[r, c] < SW_Q
                ax.text(
                    0.04,
                    0.96,
                    f"adj. p = {fmt_p(qvals[r, c])}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=14,
                    color=SIG_COLOR if sig else "black",
                    fontweight="bold" if sig else "normal",
                )
            axes[r, 0].set_ylabel(f"{VAR_SHORT[var]} [{unit}]")
        t0 = f"{init[:4]}-{init[4:6]}-{init[6:8]} {init[9:11]} UTC"
        fig.suptitle(
            f"Normal Q-Q of the ensemble members, {title} ({coords(f)}), init {t0}, +{lead} h"
        )
        fig.text(
            0.5,
            0.95,
            "dots: sorted members (ERA5: the same calendar day and hour in each year);   grey line: "
            "Gaussian with the ensemble's own mean and std\n"
            "grey band: range of Gaussian samples of the same size, widened for testing "
            f"{int(ok.sum())} panels at once (Benjamini-Hochberg, FDR {SW_Q:.0%});   "
            f"red ring: member outside the band = non-Gaussian;   adj. p: the same test as a "
            "p-value (global envelope test, BH-adjusted)",
            ha="center",
            va="top",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        FIGDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            out = FIGDIR / f"point_member_qq_{point}_{init}_{lead}h.{ext}"
            fig.savefig(out, dpi=200)
            print(f"saved {out} (band level 1 - {alpha:.4f})")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=240)
    ap.add_argument("--band-max", type=int, default=BAND_MAX)
    ap.add_argument("--member-point", default="zurich")
    ap.add_argument("--member-init", default="20231005_0000")
    ap.add_argument("--members-only", action="store_true")
    args = ap.parse_args()
    plot_members(args.lead, args.member_point, args.member_init)
    plot_member_qq(args.lead, args.member_point, args.member_init)
    if args.members_only:
        return
    plot_points(args.lead, POINTS, f"point_grid_values_{args.lead}h")
    plot_points(
        args.lead, OPPOSITE_POINTS, f"point_grid_opposite_values_{args.lead}h", "opposite meridians"
    )
    plot_bands(args.lead, args.band_max)


if __name__ == "__main__":
    main()
