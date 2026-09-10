"""Zonal power spectra of the Phase-1 magnitude sweep at a single lead time.

Row 1 is the raw log-log spectrum (ERA5 truth, the unperturbed sigma=0 run, and
every perturbed sigma); row 2 is the log10 ratio to truth, where the added
high-wavenumber power is visible - on the raw axes the curves lie on top of each
other across five decades.

Reads the NPZ written by tools/compute_psd_lead.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
import model_colors  # noqa: F401  (sets shared font rcParams on import)

from _env import FIGURES  # noqa: E402

# PSD carries the variable's units squared, so row 1 is only ever comparable
# within one variable (ERA5 z500 sits ~11 decades above specific humidity at the
# same wavenumber). The row 2 log ratio is dimensionless and cross-comparable.
PSD_UNITS = {
    "geopotential": r"m$^4$ s$^{-4}$",
    "temperature": r"K$^2$",
    "specific_humidity": r"(kg kg$^{-1}$)$^2$",
    "2m_temperature": r"K$^2$",
    "mean_sea_level_pressure": r"Pa$^2$",
    "10m_u_component_of_wind": r"m$^2$ s$^{-2}$",
    "10m_v_component_of_wind": r"m$^2$ s$^{-2}$",
}

MODEL_LABELS = {
    "aurora": "Aurora",
    "graphcast_operational": "GraphCast",
    "sfno": "SFNO",
    "aifs": "AIFS",
}


def sigma_of(run: str) -> float:
    return float(run.split("_")[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="tools/data/psd_phase1_lead006h.npz")
    ap.add_argument("--variable", default="geopotential@500hPa")
    ap.add_argument(
        "--free-y",
        action="store_true",
        help="autoscale each panel instead of sharing y within a row; the shared "
        "default keeps the four backbones directly comparable",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.npz)
    k = d["wavenumber"]
    lead = int(d["lead_hours"][0])
    models = [m for m in MODEL_LABELS if any(key.startswith(f"{m}|") for key in d.files)]

    k_cut = float(np.nanmax(k)) / 2.0
    fig, axs = plt.subplots(
        2,
        len(models),
        figsize=(4.2 * len(models), 7.6),
        sharex=True,
        sharey=("none" if args.free_y else "row"),
    )
    if len(models) == 1:
        axs = axs.reshape(2, 1)

    ratio_range: list[tuple[float, float]] = []
    for col, model in enumerate(models):
        runs = sorted(
            {key.split("|")[1] for key in d.files if key.startswith(f"{model}|")} - {"__truth__"},
            key=sigma_of,
        )
        truth = d[f"{model}|__truth__|{args.variable}"]
        perturbed = [r for r in runs if sigma_of(r) > 0]
        cmap = plt.get_cmap("plasma")
        colors = {
            r: cmap(0.05 + 0.75 * i / max(len(perturbed) - 1, 1)) for i, r in enumerate(perturbed)
        }

        top, bot = axs[0, col], axs[1, col]
        top.loglog(k, truth, color="black", lw=2.0, label="ERA5", zorder=5)
        for run in runs:
            s = d[f"{model}|{run}|{args.variable}"]
            sig = sigma_of(run)
            if sig == 0:
                top.loglog(k, s, color="0.45", lw=1.6, ls="--", label=r"$\sigma=0$ (unperturbed)")
                bot.semilogx(k, np.log10(s / truth), color="0.45", lw=1.6, ls="--")
            else:
                top.loglog(k, s, color=colors[run], lw=1.3, label=rf"$\sigma={sig:g}$")
                bot.semilogx(k, np.log10(s / truth), color=colors[run], lw=1.3)

        # ERA5 is spectrally truncated at the last few wavenumbers, so the ratio
        # explodes past the 4dx cutoff. Scale the ratio axis on the resolved range.
        resolved = k <= k_cut
        ratios = np.stack(
            [np.log10(d[f"{model}|{r}|{args.variable}"] / truth)[resolved] for r in runs]
        )
        ratio_range.append((ratios.min(), ratios.max()))
        if args.free_y:
            pad = 0.08 * (ratios.max() - ratios.min())
            bot.set_ylim(ratios.min() - pad, ratios.max() + pad)
        bot.axhline(0.0, color="black", lw=1.0, zorder=1)
        for ax in (top, bot):
            ax.axvline(k_cut, color="gold", ls=":", lw=2, alpha=0.85)
            ax.grid(True, which="both", ls="--", alpha=0.35)
            ax.set_xlim(k.min(), k.max())
        top.set_title(MODEL_LABELS[model], pad=34)
        top.legend(fontsize=8, loc="lower left")
        bot.set_xlabel("Wavenumber (cycles km$^{-1}$)")

        sec = top.secondary_xaxis("top", functions=(lambda x: 1.0 / x, lambda x: 1.0 / x))
        sec.set_xlabel("Wavelength (km)", fontsize=9)
        sec.tick_params(labelsize=8)

    if not args.free_y:
        lo = min(r[0] for r in ratio_range)
        hi = max(r[1] for r in ratio_range)
        pad = 0.08 * (hi - lo)
        axs[1, 0].set_ylim(lo - pad, hi + pad)

    unit = PSD_UNITS.get(args.variable.split("@")[0], "")
    axs[0, 0].set_ylabel(f"Zonal PSD [{unit}]" if unit else "Zonal PSD")
    axs[1, 0].set_ylabel(r"$\log_{10}$(forecast / ERA5)")
    fig.suptitle(f"{args.variable} zonal PSD at +{lead:03d} h (Phase-1 magnitude sweep)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = args.out or str(FIGURES / f"psd_lead{lead:03d}h_{args.variable.replace('@', '_')}_phase1")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
        print(f"Wrote {out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
