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
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
from model_colors import MODEL_COLORS as COLOUR
from model_colors import LINESTYLE as STYLE

from _env import BASELINES, FIGURES, INTERCOMP, SCRATCH  # noqa: E402

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
parser.add_argument(
    "--with-aifs-det",
    action="store_true",
    help="Talk variant: add the unperturbed AIFS (sigma=0, single-member CRPS = MAE) as a grey line",
)
parser.add_argument(
    "--out", type=Path, default=None, help="Output PDF path (a PNG is written alongside)"
)
cli = parser.parse_args()

CSV = str(INTERCOMP / "probabilistic" / "temporal_metrics_combined.csv")
CRPS_CLIM = (
    str(cli.climatology_json)
    if cli.climatology_json is not None
    else str(Path(__file__).resolve().parent / "data" / "crps_clim_eval_1990_2019.json")
)
AIFS_PERT_PROB = BASELINES / "aifs_perturbed/eval/probabilistic"
AIFS_DET_DET = BASELINES / "aifs_det/eval/deterministic"
if cli.out is not None:
    OUT = str(cli.out)
elif cli.with_aifs_det:
    OUT = str(FIGURES / "talk" / "headline_crpss_vs_lead_8way_aifs_det.pdf")
else:
    OUT = str(FIGURES / "headline_crpss_vs_lead_8way.pdf")

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


def _load_det_mae(root, model):
    """SwissClim deterministic long CSV (lead_time_hours, variable, level, MAE).
    For a single member the CRPS reduces to the MAE, so the MAE goes into
    `data` in place of the CRPS.
    """
    (path,) = root.glob("deterministic_metrics_by_lead_long_*.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["variable"] not in VARS_2D + VARS_3D:
                continue
            lvl = None if row["variable"] in VARS_2D else float(row["level"])
            data[(model, row["variable"], int(row["lead_time_hours"]), lvl)] = float(row["MAE"])


if cli.with_aifs_det:
    _load_det_mae(AIFS_DET_DET, "aifs_det")
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

if cli.with_aifs_det:
    det = [(lead, crpss("aifs_det", lead)) for lead in LEADS if lead > 0]
    det = [(x, y) for x, y in det if y is not None]
    ax.plot(
        [x for x, _ in det],
        [y for _, y in det],
        label=r"AIFS ($\sigma=0$)",
        color="#A6A6A6",
        linestyle="-",
        linewidth=1.6,
    )

# Reference baselines:
# (a) Climatology: CRPSS = 0 exactly, by definition of the CRPSS denominator.
# (b) Persistence: forecast(t+h) = analysis(t). Empirical MAE from
#     tools/compute_persistence_mae.py against the local 2022-2025 ERA5 zarr;
#     converted to CRPSS using the same WB2 clim-CRPS denominator as the models.
if cli.persistence_json is not None:
    PERSISTENCE_JSON = cli.persistence_json
elif (Path(__file__).resolve().parent / "data" / "persistence_mae_112inits.json").exists():
    PERSISTENCE_JSON = Path(__file__).resolve().parent / "data" / "persistence_mae_112inits.json"
elif (SCRATCH / "persistence_mae_112inits.json").exists():
    PERSISTENCE_JSON = SCRATCH / "persistence_mae_112inits.json"
else:
    PERSISTENCE_JSON = SCRATCH / "persistence_mae_2024.json"
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
ax.set_ylim(-0.1, 1.0)  # cut below -0.1: persistence dives off-axis after the first 2 leads
if cli.with_aifs_det:
    # Single-member MAE drops well below the probabilistic climatology at long leads.
    ax.set_ylim(min(-0.1, math.floor(min(y for _, y in det) * 10 - 0.5) / 10), 1.0)
ax.set_xticks([0, 24, 72, 120, 168, 240, 312, 360])
ax.set_xlabel("Lead time (h)")
ax.set_ylabel("CRPSS (variable-mean, 7 paper variables)")
ax.grid(True, linewidth=0.4, alpha=0.5)
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
print(f"-> {OUT}")
if cli.with_aifs_det:
    plt.savefig(str(Path(OUT).with_suffix(".png")), dpi=300, bbox_inches="tight")

# Sanity check: report 240 h CRPSS in order
print("\nCRPSS @ 240 h:")
ordered = sorted(MODELS, key=lambda m: -(crpss(m, 240) or -99))
for m in ordered:
    v = crpss(m, 240)
    print(f"  {PRETTY[m]:25s} {v:.3f}" if v is not None else f"  {PRETTY[m]:25s} --")
if cli.with_aifs_det:
    print(f"  {'AIFS (sigma=0, MAE)':25s} {crpss('aifs_det', 240):.3f}")
