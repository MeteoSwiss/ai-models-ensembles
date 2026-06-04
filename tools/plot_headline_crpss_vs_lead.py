"""Produce the headline 7-way CRPSS vs lead-time PDF for the paper.

Reuses the climatology + CRPSS logic of headline_7way_table.py but produces
a line plot over the full 0-360 h lead range (61 leads, 6 h step).
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

CSV = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/intercomparison/probabilistic/temporal_metrics_combined.csv"
SIGMA = "/iopsstor/scratch/cscs/sadamov/sigma_clim_ablation.json"
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/headline_crpss_vs_lead_7way.pdf"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--persistence-json", type=Path, default=None)
parser.add_argument("--climatology-json", type=Path, default=None)
args = parser.parse_args()

VARS_2D = ["2m_temperature"]  # MSL excluded per the ifs_ens MSL bug
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]

MODELS = ["aifsens", "atlas", "fcn3", "ifs_ens", "graphcast_all", "aurora_encoder", "sfno_modes10"]

PRETTY = {
    "aifsens": "AIFS-ENS",
    "atlas": "Atlas",
    "fcn3": "FCN3",
    "ifs_ens": "IFS-ENS",
    "graphcast_all": "graphcast_all",
    "aurora_encoder": "aurora_encoder",
    "sfno_modes10": "sfno_modes10",
}

# Match the paper's table colour palette (experiments_tables.tex)
COLOUR = {
    "aurora_encoder": "#E67E22",
    "graphcast_all": "#27AE60",
    "sfno_modes10": "#2980B9",
    "aifsens": "#8B5A2B",
    "atlas": "#C0392B",
    "fcn3": "#D4A017",
    "ifs_ens": "#7F8C8D",
}

STYLE = {
    "aurora_encoder": "-",
    "graphcast_all": "-",
    "sfno_modes10": "-",
    "aifsens": "--",
    "atlas": "--",
    "fcn3": "--",
    "ifs_ens": ":",
}

sigma = json.load(open(args.climatology_json if args.climatology_json is not None else SIGMA))


def sig_for(var, lvl):
    if var in VARS_2D:
        return sigma.get(var)
    if var in VARS_3D and lvl is not None:
        return sigma.get(f"{var}_{int(lvl)}")
    return None


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

LEADS = sorted({k[2] for k in data})


def crpss(model, lead):
    per_var = []
    for v in VARS_2D:
        candidates = [
            val for k, val in data.items() if k[0] == model and k[1] == v and k[2] == lead
        ]
        if not candidates:
            continue
        s = sig_for(v, None)
        if s is None:
            continue
        per_var.append(1 - candidates[0] / (s / math.sqrt(math.pi)))
    for v in VARS_3D:
        skills = []
        for lvl in (500.0, 850.0):
            crps_v = data.get((model, v, lead, lvl))
            if crps_v is None:
                continue
            s = sig_for(v, lvl)
            if s is None:
                continue
            skills.append(1 - crps_v / (s / math.sqrt(math.pi)))
        if skills:
            per_var.append(sum(skills) / len(skills))
    if not per_var:
        return None
    return sum(per_var) / len(per_var)


fig, ax = plt.subplots(figsize=(6.5, 4.2))

for m in MODELS:
    xs, ys = [], []
    for lead in LEADS:
        v = crpss(m, lead)
        if v is None or lead == 0:
            continue
        xs.append(lead)
        ys.append(v)
    ax.plot(xs, ys, label=PRETTY[m], color=COLOUR[m], linestyle=STYLE[m], linewidth=1.6)

if args.persistence_json is not None and args.persistence_json.exists():
    pers_mae = json.load(open(args.persistence_json))

    def persistence_crpss(lead):
        per_var = []
        for v in VARS_2D:
            mae_dict = pers_mae.get(v)
            if not mae_dict:
                continue
            mae = mae_dict.get(str(lead))
            s = sig_for(v, None)
            if mae is None or s is None:
                continue
            per_var.append(1 - mae / (s / math.sqrt(math.pi)))
        for v in VARS_3D:
            skills = []
            for lvl in (500, 850):
                mae_dict = pers_mae.get(f"{v}_{lvl}")
                if not mae_dict:
                    continue
                mae = mae_dict.get(str(lead))
                s = sig_for(v, float(lvl))
                if mae is None or s is None:
                    continue
                skills.append(1 - mae / (s / math.sqrt(math.pi)))
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
elif args.persistence_json is not None:
    print(f"WARN: {args.persistence_json} missing; skipping persistence reference curve")

if args.climatology_json is not None:
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label="Climatology")
else:
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.5)
ax.set_xlim(0, 360)
ax.set_ylim(-0.05, 1.0)
ax.set_xticks([0, 24, 72, 120, 168, 240, 312, 360])
ax.set_xlabel("Lead time (h)")
ax.set_ylabel("CRPSS (variable-mean, 6 paper variables)")
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.95)
ax.set_title("Headline 7-way intercomparison on the 112-init production grid", fontsize=10)

plt.tight_layout()
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"-> {OUT}")

# Sanity check: report 240 h CRPSS in order
print("\nCRPSS @ 240 h:")
ordered = sorted(MODELS, key=lambda m: -(crpss(m, 240) or -99))
for m in ordered:
    v = crpss(m, 240)
    print(f"  {PRETTY[m]:25s} {v:.3f}" if v is not None else f"  {PRETTY[m]:25s} --")
