"""Band-integrated log spectral distance against lead time, Phase-1 sweep.

Rows are backbones, columns the four wavelength bands, one line per perturbation
magnitude including sigma=0.

Two sources. ``--source pooled`` (default) reads the recomputed CSV from
tools/compute_banded_lsd.py, a true mean of per-member spectra. ``--source eval``
reads the SwissClim intercomparison CSVs, which despite their "enspooled" token
hold member 0 only: the target carries a singleton ensemble dim, and
_compute_spectra_pair's xr.align(..., join="inner") intersects the ensemble
coordinate down to {0}. The eval source also has no AIFS sigma=0 row, whose
eval never ran.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
import model_colors  # noqa: F401  (sets shared font rcParams on import)

from _env import DATA, FIGURES, STORE  # noqa: E402

MODEL_LABELS = {
    "aurora": "Aurora",
    "graphcast_operational": "GraphCast",
    "sfno": "SFNO",
    "aifs": "AIFS",
}
BANDS = [
    ("planetary", "Planetary\n5000-20000 km"),
    ("synoptic", "Synoptic\n1000-5000 km"),
    ("upper_mesoscale", "Upper mesoscale\n250-1000 km"),
    ("lower_mesoscale", "Lower mesoscale\n10-250 km"),
]


def sigma_of(run: str) -> float:
    return float(run.split("_")[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variable", default="geopotential@500hPa")
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--models", nargs="+", default=list(MODEL_LABELS))
    ap.add_argument(
        "--source",
        choices=["pooled", "eval"],
        default="pooled",
        help="pooled: recomputed mean of per-member spectra; eval: the SwissClim "
        "intercomparison CSVs, which are member-0 only (see compute_banded_lsd.py)",
    )
    ap.add_argument(
        "--free-y",
        action="store_true",
        help="autoscale every panel instead of sharing y down each band column; "
        "LSD is dimensionless, so the shared default keeps backbones comparable",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pooled = None
    if args.source == "pooled":
        parts = sorted(DATA.glob("lsd_bands_phase1_pooled*.csv"))
        if not parts:
            raise SystemExit("no pooled CSV found; run tools/submit_banded_lsd_sweep.sh first")
        pooled = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
        pooled = pooled.drop_duplicates(
            subset=["backbone", "model", "lead_time_hours", "variable", "band"], keep="last"
        )

    fig, axs = plt.subplots(
        len(args.models),
        len(BANDS),
        figsize=(4.0 * len(BANDS), 3.0 * len(args.models)),
        sharex=True,
        sharey=("none" if args.free_y else "col"),
    )
    axs = axs.reshape(len(args.models), len(BANDS))

    for row, model in enumerate(args.models):
        csv = (
            STORE
            / "ablation"
            / args.phase
            / model
            / "intercomparison"
            / "energy_spectra"
            / "lsd_metrics_banded_lead_time_combined.csv"
        )
        if pooled is not None:
            df = pooled[pooled["backbone"] == model]
        else:
            df = pd.read_csv(csv)
        df = df[df["variable"] == args.variable]
        runs = sorted(df["model"].unique(), key=sigma_of)
        perturbed = [r for r in runs if sigma_of(r) > 0]
        cmap = plt.get_cmap("plasma")
        colors = {
            r: cmap(0.05 + 0.75 * i / max(len(perturbed) - 1, 1)) for i, r in enumerate(perturbed)
        }

        for col, (band, band_label) in enumerate(BANDS):
            ax = axs[row, col]
            sub = df[df["band"] == band]
            for run in runs:
                s = sub[sub["model"] == run].sort_values("lead_time_hours")
                sig = sigma_of(run)
                if sig == 0:
                    ax.plot(
                        s["lead_time_hours"],
                        s["LSD"],
                        color="0.45",
                        lw=1.8,
                        ls="--",
                        label=r"$\sigma=0$ (unperturbed)",
                    )
                else:
                    ax.plot(
                        s["lead_time_hours"],
                        s["LSD"],
                        color=colors[run],
                        lw=1.4,
                        label=rf"$\sigma={sig:g}$",
                    )
            ax.grid(True, ls="--", alpha=0.35)
            if row == 0:
                ax.set_title(band_label, fontsize=10)
            if row == len(args.models) - 1:
                ax.set_xlabel("Lead time (h)")
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nLSD")
        axs[row, -1].legend(fontsize=8, loc="best")

    src = "ensemble-pooled" if args.source == "pooled" else "member 0 only"
    fig.suptitle(f"Band-integrated log spectral distance, {args.variable} (Phase-1 sweep, {src})")
    fig.tight_layout()

    out = args.out or str(
        FIGURES / f"lsd_bands_vs_lead_{args.variable.replace('@', '_')}_phase1_{args.source}"
    )
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
        print(f"Wrote {out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
