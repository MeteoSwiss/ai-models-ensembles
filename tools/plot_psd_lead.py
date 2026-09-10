"""Zonal power spectra of an ablation sweep at a single lead time.

Row 1 is the raw log-log spectrum (ERA5 truth, the unperturbed sigma=0 run, and
every perturbed run); row 2 is the log10 ratio to truth, where the added
high-wavenumber power is visible - on the raw axes the curves lie on top of each
other across five decades.

When the NPZ holds more than one perturbation target (the Phase-2 layer groups,
the Phase-3 coarse-scale targets) curves are coloured by target and shaded by
sigma; a single-target sweep falls back to the plasma sigma ramp.

Reads the NPZ written by tools/compute_psd_lead.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
from model_colors import color_for  # noqa: E402  (import also sets shared font rcParams)

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

# Colour always encodes sigma and the line style encodes the perturbation target,
# so a panel holding several targets at several magnitudes stays readable. The
# ramp is plasma truncated before its pale yellow end: it spans blue to orange,
# so neighbouring sigmas differ in hue rather than only in lightness. Curves
# thin out as sigma grows, which lets a weaker run show from underneath where
# two spectra coincide (most of Phase 3 is exactly that).
GROUP_STYLES = ["-", "--", "-.", ":"]

# --model-colors draws every perturbed curve in the backbone's canonical paper
# colour, which is what the production overlay wants: one run per panel, and the
# colour then matches that baseline everywhere else in the paper.
PRODUCTION_IDS = {
    "aurora": "aurora_encoder",
    "graphcast_operational": "graphcast_all",
    "sfno": "sfno_modes10",
    "aifs": "aifs_perturbed",
}
GROUP_LABELS = {
    "all": "all weights",
    "unet_bottom": "U-Net bottom",
    "gcnodes42": "mesh 42",
    "gcnodes162": "mesh 162",
    "gcnodes642": "mesh 642",
    "modes10": r"modes $\ell<10$",
    "modes20": r"modes $\ell<20$",
    "modes40": r"modes $\ell<40$",
}
EARTH_CIRCUMFERENCE_KM = 40030.0


def parse_run(run: str) -> tuple[float, str]:
    """``[<phase>:]<prefix>_<sigma>_<target>`` -> (sigma, target)."""
    tail = run.split(":")[-1]
    parts = tail.split("_")
    target = "_".join(parts[2:]).removeprefix("layer_") or "all"
    return float(parts[1]), target


def sigma_of(run: str) -> float:
    return parse_run(run)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", default=["tools/data/psd_phase1_lead006h.npz"])
    ap.add_argument("--variable", default="geopotential@500hPa")
    ap.add_argument("--tag", default="phase1", help="suffix of the output figure name")
    ap.add_argument("--models", nargs="+", default=None, help="subset of backbones to plot")
    ap.add_argument(
        "--model-colors",
        action="store_true",
        help="colour the perturbed curves by backbone instead of by sigma",
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="draw only these runs, as <model>=<run key>; the sigma=0 run is always kept",
    )
    ap.add_argument(
        "--harmonics",
        type=float,
        default=None,
        help="draw guides at multiples of this zonal wavenumber (Aurora patch grid: 180)",
    )
    ap.add_argument(
        "--free-y",
        action="store_true",
        help="autoscale each panel instead of sharing y within a row; the shared "
        "default keeps the four backbones directly comparable",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    merged: dict[str, np.ndarray] = {}
    for path in args.npz:
        merged.update(dict(np.load(path).items()))
    # the unperturbed run is written by every phase bundle, prefixed there and bare
    # in the Phase-1 one; keep a single copy so it is drawn once
    d = {
        key: val
        for key, val in merged.items()
        if not (
            key.count("|") == 2
            and ":" in key.split("|")[1]
            and "|".join(
                [key.split("|")[0], key.split("|")[1].split(":", 1)[1], key.split("|", 2)[2]]
            )
            in merged
        )
    }
    k = d["wavenumber"]
    lead = int(d["lead_hours"][0])
    models = [m for m in args.models or MODEL_LABELS if any(key.startswith(f"{m}|") for key in d)]

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
            {key.split("|")[1] for key in d if key.startswith(f"{model}|")} - {"__truth__"},
            key=sigma_of,
        )
        if args.runs:
            want = {r.split("=", 1)[1] for r in args.runs if r.split("=", 1)[0] == model}
            runs = [r for r in runs if sigma_of(r) == 0 or r in want]
        truth = d[f"{model}|__truth__|{args.variable}"]
        perturbed = [r for r in runs if sigma_of(r) > 0]
        groups = [parse_run(r)[1] for r in perturbed]
        by_group = len(set(groups)) > 1
        cmap = plt.get_cmap("plasma")
        span = max(len(perturbed) - 1, 1)
        if args.model_colors:
            colors = {r: color_for(PRODUCTION_IDS.get(model, model)) for r in perturbed}
        else:
            colors = {r: cmap(0.80 * i / span) for i, r in enumerate(perturbed)}
        widths = {r: 2.4 - 1.1 * i / span for i, r in enumerate(perturbed)}
        # the target with the most magnitudes carries the plain solid line
        ranked = sorted(dict.fromkeys(groups), key=lambda g: -groups.count(g))
        styles = {g: GROUP_STYLES[i % len(GROUP_STYLES)] for i, g in enumerate(ranked)}

        top, bot = axs[0, col], axs[1, col]
        top.loglog(k, truth, color="black", lw=2.0, label="ERA5", zorder=5)
        for run in runs:
            s = d[f"{model}|{run}|{args.variable}"]
            sig = sigma_of(run)
            if sig == 0:
                top.loglog(k, s, color="0.45", lw=1.6, ls="--", label=r"$\sigma=0$ (unperturbed)")
                bot.semilogx(k, np.log10(s / truth), color="0.45", lw=1.6, ls="--")
            else:
                group = parse_run(run)[1]
                label = rf"$\sigma={sig:g}$"
                ls = "-"
                if by_group:
                    label = f"{GROUP_LABELS.get(group, group)}, {label}"
                    ls = styles[group]
                lw = widths[run]
                top.loglog(k, s, color=colors[run], ls=ls, lw=lw, label=label)
                bot.semilogx(k, np.log10(s / truth), color=colors[run], ls=ls, lw=lw)

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
            if args.harmonics:
                n = 1
                while args.harmonics * n / EARTH_CIRCUMFERENCE_KM <= k.max():
                    ax.axvline(
                        args.harmonics * n / EARTH_CIRCUMFERENCE_KM,
                        color="0.75",
                        ls="-",
                        lw=0.8,
                        alpha=0.6,
                        zorder=0,
                    )
                    n += 1
            ax.grid(True, which="both", ls="--", alpha=0.2)
            ax.set_xlim(k.min(), k.max())
        top.set_title(MODEL_LABELS[model], pad=34)
        top.legend(fontsize=7 if by_group else 8, loc="lower left")
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
    fig.suptitle(f"{args.variable} zonal PSD at +{lead:03d} h ({args.tag})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = args.out or str(
        FIGURES / f"psd_lead{lead:03d}h_{args.variable.replace('@', '_')}_{args.tag}"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
        print(f"Wrote {out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
