"""Produce the headline CRPSS vs lead-time PDF for the paper.

Trained-probabilistic baselines (AIFS-ENS, FCN3, Atlas) and the classical
IFS-ENS reference come from the SwissClim intercomparison temporal-metrics
CSV; the post-hoc weight-only baselines are read
from per-baseline probabilistic CSVs.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--persistence-json",
    type=Path,
    default=None,
    help="Persistence MAE JSON (default: 112-init if present, else 20-init 2024)",
)
parser.add_argument(
    "--climatology-json",
    type=Path,
    default=None,
    help="WB2 clim-CRPS denominator JSON (default: tools/data/crps_clim_eval_1990_2019.json)",
)
cli = parser.parse_args()

CSV = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/intercomparison/probabilistic/temporal_metrics_combined.csv"
CRPS_CLIM = (
    str(cli.climatology_json)
    if cli.climatology_json is not None
    else str(Path(__file__).resolve().parent / "data" / "crps_clim_eval_1990_2019.json")
)
AIFS_PERT_PROB = Path(
    "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/"
    "aifs_perturbed/eval/probabilistic"
)
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/headline_crpss_vs_lead_8way.pdf"

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]

MODELS = [
    "aifsens",
    "atlas",
    "fcn3",
    "ifs_ens",
    "graphcast_all",
    "aurora_encoder",
    "sfno_modes10",
    "aifs_perturbed",
]

PRETTY = {
    "aifsens": "AIFS-ENS",
    "atlas": "Atlas",
    "fcn3": "FCN3",
    "ifs_ens": "IFS-ENS",
    "graphcast_all": "graphcast_all",
    "aurora_encoder": "aurora_encoder",
    "sfno_modes10": "sfno_modes10",
    "aifs_perturbed": "aifs_perturbed",
}

COLOUR = {
    "aurora_encoder": "#E67E22",
    "graphcast_all": "#27AE60",
    "sfno_modes10": "#2980B9",
    "aifs_perturbed": "#8E44AD",
    "aifsens": "#8B5A2B",
    "atlas": "#C0392B",
    "fcn3": "#D4A017",
    "ifs_ens": "#7F8C8D",
}

STYLE = {
    "aurora_encoder": "-",
    "graphcast_all": "-",
    "sfno_modes10": "-",
    "aifs_perturbed": "-",
    "aifsens": "--",
    "atlas": "--",
    "fcn3": "--",
    "ifs_ens": ":",
}

crps_clim = json.load(open(CRPS_CLIM))


def clim_for(var, lvl, lead):
    """WB2 probabilistic-climatology CRPS denominator at this lead."""
    key = (
        var
        if var in VARS_2D
        else f"{var}_{int(lvl)}"
        if (var in VARS_3D and lvl is not None)
        else None
    )
    if key is None or key not in crps_clim:
        return None
    return crps_clim[key].get(str(lead))


# Load the 7-way CRPS values from the SwissClim intercomp temporal-metrics CSV.
data = {}
with open(CSV) as f:
    for row in csv.DictReader(f):
        if row["metric"] != "CRPS" or row["model"] not in MODELS:
            continue
        try:
            lead = int(row["lead_time"])
            lvl = float(row["level"]) if row["level"] else None
            val = float(row["value"])
        except ValueError:
            continue
        data[(row["model"], row["variable"], lead, lvl)] = val


def _load_perbase(root, model):
    """Per-baseline `crps_line_<stem>_by_lead_ensprob.csv` (lead_time_hours, variable, CRPS).
    Used for the post-hoc aifs_perturbed baseline, which was added after the
    most recent combined-CSV regen.
    """

    def _load(stem, var_label, level):
        path = root / f"crps_line_{stem}_by_lead_ensprob.csv"
        if not path.exists():
            return
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    lead = int(row["lead_time_hours"])
                    val = float(row["CRPS"])
                except ValueError:
                    continue
                data[(model, var_label, lead, level)] = val

    for v in VARS_2D:
        _load(v, v, None)
    for v in VARS_3D:
        for lvl in (500, 850):
            _load(f"{v}_{lvl}", v, float(lvl))


_load_perbase(AIFS_PERT_PROB, "aifs_perturbed")
LEADS = sorted({k[2] for k in data})


def crpss(model, lead):
    per_var = []
    for v in VARS_2D:
        candidates = [
            val for k, val in data.items() if k[0] == model and k[1] == v and k[2] == lead
        ]
        if not candidates:
            continue
        c = clim_for(v, None, lead)
        if c is None:
            continue
        per_var.append(1 - candidates[0] / c)
    for v in VARS_3D:
        skills = []
        for lvl in (500.0, 850.0):
            crps_v = data.get((model, v, lead, lvl))
            if crps_v is None:
                continue
            c = clim_for(v, lvl, lead)
            if c is None:
                continue
            skills.append(1 - crps_v / c)
        if skills:
            per_var.append(sum(skills) / len(skills))
    if not per_var:
        return None
    return sum(per_var) / len(per_var)


fig, ax = plt.subplots(figsize=(7.0, 7.5))

for m in MODELS:
    xs, ys = [], []
    for lead in LEADS:
        v = crpss(m, lead)
        if v is None or lead == 0:
            continue
        xs.append(lead)
        ys.append(v)
    ax.plot(xs, ys, label=PRETTY[m], color=COLOUR[m], linestyle=STYLE[m], linewidth=1.6)

# Reference baselines:
# (a) Climatology: CRPSS = 0 exactly, by definition of the CRPSS denominator.
# (b) Persistence: forecast(t+h) = analysis(t). Empirical MAE from
#     tools/compute_persistence_mae.py against the local 2022-2025 ERA5 zarr;
#     converted to CRPSS using the same WB2 clim-CRPS denominator as the models.
if cli.persistence_json is not None:
    PERSISTENCE_JSON = cli.persistence_json
elif Path("/iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.json").exists():
    PERSISTENCE_JSON = Path("/iopsstor/scratch/cscs/sadamov/persistence_mae_112inits.json")
else:
    PERSISTENCE_JSON = Path("/iopsstor/scratch/cscs/sadamov/persistence_mae_2024.json")
if PERSISTENCE_JSON.exists():
    print(f"Using persistence: {PERSISTENCE_JSON}")
    pers_mae = json.load(open(PERSISTENCE_JSON))

    def persistence_crpss(lead):
        per_var = []
        for v in VARS_2D:
            mae_dict = pers_mae.get(v)
            if not mae_dict:
                continue
            mae = mae_dict.get(str(lead))
            c = clim_for(v, None, lead)
            if mae is None or c is None:
                continue
            per_var.append(1 - mae / c)
        for v in VARS_3D:
            skills = []
            for lvl in (500, 850):
                mae_dict = pers_mae.get(f"{v}_{lvl}")
                if not mae_dict:
                    continue
                mae = mae_dict.get(str(lead))
                c = clim_for(v, float(lvl), lead)
                if mae is None or c is None:
                    continue
                skills.append(1 - mae / c)
            if skills:
                per_var.append(sum(skills) / len(skills))
        return sum(per_var) / len(per_var) if per_var else None

    pers_x, pers_y = [], []
    for lead in sorted(int(h) for h in next(iter(pers_mae.values())).keys()):
        v = persistence_crpss(lead)
        if v is not None and 0 < lead <= 360:
            pers_x.append(lead)
            pers_y.append(v)
    ax.plot(
        pers_x, pers_y, label="Persistence", color="black", linestyle=":", linewidth=1.0, alpha=0.7
    )
else:
    print(f"WARN: {PERSISTENCE_JSON} missing; skipping persistence reference curve")
ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label="Climatology")
ax.set_xlim(0, 360)
ax.set_ylim(-0.65, 1.0)
ax.set_xticks([0, 24, 72, 120, 168, 240, 312, 360])
ax.set_xlabel("Lead time (h)")
ax.set_ylabel("CRPSS (variable-mean, 7 paper variables)")
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.set_title("Headline intercomparison on the 112-init production grid", fontsize=10)
# Legend below the plot in 2 horizontal rows -- keeps the y-axis full-height.
handles, labels = ax.get_legend_handles_labels()
ncol = math.ceil(len(handles) / 2)
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=ncol,
    fontsize=7,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout(rect=(0, 0.10, 1, 1))
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print(f"-> {OUT}")
print(f"-> {OUT.replace('.pdf', '.png')}")

# Sanity check: report 240 h CRPSS in order
print("\nCRPSS @ 240 h:")
ordered = sorted(MODELS, key=lambda m: -(crpss(m, 240) or -99))
for m in ordered:
    v = crpss(m, 240)
    print(f"  {PRETTY[m]:25s} {v:.3f}" if v is not None else f"  {PRETTY[m]:25s} --")
