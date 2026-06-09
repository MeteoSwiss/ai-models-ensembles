"""Phase 5 diagnostic: lead-resolved SSR for aifs_perturbed (weight-only)
vs aifs_perturbed_ic (real IFS-ENS IC + same weight perturbation).

Per [[phase5_perturbed_ic_plan]], the scientific test of Phase 5 is whether
SSR(lead) stays close to 1 across the whole 0-360h forecast. Production
aifs_perturbed (weight-only) is under-dispersed at early leads then drifts
toward / over 1 by day 10. The hypothesis: adding real EDA-derived IC
spread per member raises early-lead SSR to ~1 without inflating long-lead
SSR (the two sources are lead-time complementary, not additive).

Inputs: per-baseline ssr_line_<var>_by_lead_ensprob.csv files from each
baseline's eval/probabilistic/.

Output: one panel per paper variable + a variable-mean panel, two curves
each (aifs_perturbed dashed, aifs_perturbed_ic solid). Includes a
horizontal reference line at SSR=1.

Run after aifs_perturbed_ic eval/probabilistic lands.
"""

from __future__ import annotations
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PDF = Path("/users/sadamov/pyprojects/ai-models-ensembles/figures/ssr_phase5_aifs_lead.pdf")

BASELINES = {
    "aifs_perturbed": {
        "root": Path(
            "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/"
            "aifs_perturbed/eval/probabilistic"
        ),
        "label": "aifs_perturbed (weight only)",
        "style": "--",
        "colour": "#8E44AD",
    },
    "aifs_perturbed_ic": {
        "root": Path(
            "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/"
            "aifs_perturbed_ic/eval/probabilistic"
        ),
        "label": "aifs_perturbed_ic (IFS-ENS IC + weight)",
        "style": "-",
        "colour": "#D81B60",
    },
}

VARS_2D = [
    ("2m_temperature", "2m_temperature", None, "T2m"),
    ("mean_sea_level_pressure", "mean_sea_level_pressure", None, "MSL"),
]
VARS_3D = [
    ("geopotential", "geopotential", 500, "Z500"),
    ("temperature", "temperature", 850, "T850"),
    ("u_component_of_wind", "u_component_of_wind", 850, "U850"),
    ("v_component_of_wind", "v_component_of_wind", 850, "V850"),
    ("specific_humidity", "specific_humidity", 500, "q500"),
]


def load_ssr(root: Path, var_stem: str) -> tuple[list[int], list[float]]:
    p = root / f"ssr_line_{var_stem}_by_lead_ensprob.csv"
    if not p.exists():
        return [], []
    leads, ssrs = [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                leads.append(int(row["lead_time_hours"]))
                ssrs.append(float(row["SSR"]))
            except (ValueError, KeyError):
                continue
    return leads, ssrs


panels = []
for v_long, v_stem_2d, _, pretty in VARS_2D:
    panels.append((pretty, v_stem_2d))
for v_long, v_stem_3d, lvl, pretty in VARS_3D:
    panels.append((pretty, f"{v_stem_3d}_{lvl}"))

# Add a variable-mean panel as the last subplot
panels.append(("variable mean", None))

n_panels = len(panels)
n_cols = 4
n_rows = math.ceil(n_panels / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), sharex=True)
axes_flat = axes.flatten() if n_rows > 1 else axes

per_var_curves: dict[str, dict[str, tuple[list[int], list[float]]]] = {}

for ax, (pretty, var_stem) in zip(axes_flat, panels):
    if var_stem is None:
        # variable-mean panel
        for bl_key, bl in BASELINES.items():
            # Build per-lead mean across all per-var curves we plotted
            agg: dict[int, list[float]] = {}
            for stem, curves in per_var_curves.items():
                if bl_key not in curves:
                    continue
                leads_, ssrs_ = curves[bl_key]
                for lead_h, s in zip(leads_, ssrs_):
                    agg.setdefault(lead_h, []).append(s)
            leads = sorted(agg)
            ssrs = [sum(agg[h]) / len(agg[h]) for h in leads]
            ax.plot(
                leads,
                ssrs,
                color=bl["colour"],
                linestyle=bl["style"],
                linewidth=1.5,
                label=bl["label"],
            )
        ax.set_title(pretty, fontsize=9)
    else:
        per_var_curves[var_stem] = {}
        for bl_key, bl in BASELINES.items():
            leads, ssrs = load_ssr(bl["root"], var_stem)
            if not leads:
                continue
            per_var_curves[var_stem][bl_key] = (leads, ssrs)
            ax.plot(leads, ssrs, color=bl["colour"], linestyle=bl["style"], linewidth=1.4)
        ax.set_title(pretty, fontsize=9)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.3)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, max(1.6, ax.get_ylim()[1]))
    ax.set_xticks([0, 72, 144, 240, 360])
    ax.grid(True, linewidth=0.3, alpha=0.5)

# Hide unused trailing axes if any
for j in range(n_panels, len(axes_flat)):
    axes_flat[j].axis("off")

# X labels and legend on the last used axis
for ax in axes_flat[-n_cols:]:
    ax.set_xlabel("Lead (h)")
for r in range(n_rows):
    axes_flat[r * n_cols].set_ylabel("SSR")

# Single legend at the top
handles_labels = []
for bl_key, bl in BASELINES.items():
    handles_labels.append(
        (
            plt.Line2D([], [], color=bl["colour"], linestyle=bl["style"], linewidth=1.6),
            bl["label"],
        )
    )
handles_labels.append(
    (
        plt.Line2D([], [], color="black", linestyle="--", linewidth=0.6, alpha=0.6),
        "SSR = 1 (calibrated)",
    )
)
fig.legend(
    *zip(*handles_labels),
    loc="lower center",
    ncol=3,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle("Phase 5 AIFS: lead-resolved SSR, weight-only vs IFS-ENS-IC + weight", fontsize=10)
plt.tight_layout(rect=(0, 0.03, 1, 0.97))
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
plt.savefig(str(OUT_PDF).replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print(f"-> {OUT_PDF}")
print(f"-> {OUT_PDF.with_suffix('.png')}")

# Sanity check: variable-mean SSR at key leads
print("\nVariable-mean SSR (mean across the 7 panels):")
for bl_key, bl in BASELINES.items():
    agg: dict[int, list[float]] = {}
    for stem, curves in per_var_curves.items():
        if bl_key not in curves:
            continue
        leads_, ssrs_ = curves[bl_key]
        for lead_h, s in zip(leads_, ssrs_):
            agg.setdefault(lead_h, []).append(s)
    if not agg:
        print(f"  {bl['label']:60s}  -- no data on disk yet --")
        continue
    msg = []
    for lead_h in (24, 72, 120, 240, 360):
        if lead_h in agg:
            msg.append(f"{lead_h}h: {sum(agg[lead_h])/len(agg[lead_h]):.3f}")
        else:
            msg.append(f"{lead_h}h: --")
    print(f"  {bl['label']:60s}  " + "  ".join(msg))
