"""Recompute the CRPSS columns of the calibration table (Tab. calibration) with
the exact-WB2 ablation-grid climatology denominator.

NOTE (2026-06-22): the full calibration table is now regenerated end-to-end by
tools/assemble_calibration_table.py (all 9 metrics, single current vintage, frozen
GraphCast Phase 3/3b). This script remains a CRPSS-only diagnostic over the
allphases tree; its GraphCast Phase 3/3b rows are the historical FRESH sweep
(the frozen run_tags live under phase3/phase3b, not allphases).

The calibration table is hand-maintained (figures/calibration_basis_table.tex);
this prints the variable-mean CRPSS at 24/72/120/240 h for every (model, phase)
row so the CRPSS cells can be updated, with the within-model per-lead argmax
marked (the table's bold rule for CRPSS).

Denominator: crps_clim_eval_ablation_1990_2019.json (4 mid-season inits,
fair CRPS of the 1990-2019 ensemble vs eval truth, lead-resolved).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

CRPS_CLIM = Path(__file__).resolve().parent / "data" / "crps_clim_eval_ablation_1990_2019.json"
ROOT = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation/allphases"

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEADS = [24, 72, 120, 240]

# (model_dir, [(row_label, run_key), ...]) in calibration-table row order.
ROWS = {
    "aurora": [
        ("Phase 1  0.03 / all", "mag_0.03_layer_all"),
        ("Phase 2  0.044 / encoder", "mag_0.044176_layer_encoder"),
        ("Phase 2b 0.025 / encoder *", "mag_0.025_layer_encoder"),
        ("Phase 3  0.40 / enc_2", "mag_0.40_layer_unet_bottom"),
        ("Phase 3b 0.015 / enc_012", "mag_0.015_layer_enc_012"),
    ],
    "graphcast_operational": [
        ("Phase 1  0.01 / all *", "mag_0.01_layer_all"),
        ("Phase 2  0.030 / m2g", "mag_0.029665_layer_m2g"),
        ("Phase 2b 0.014 / g2m", "mag_0.014_layer_g2m"),
        ("Phase 3  1.80 / n_coarse_42", "gcsigma_1.80_gcnodes42"),
        ("Phase 3b 0.159 / n_coarse_162", "gcsigma_0.159_gcnodes162"),
    ],
    "sfno": [
        ("Phase 1  0.03 / all", "mag_0.03_layer_all"),
        ("Phase 2  0.054 / encoder", "mag_0.053852_layer_encoder"),
        ("Phase 2b 0.035 / encoder", "mag_0.035_layer_encoder"),
        ("Phase 3  0.25 / Lcut_10 *", "mag_0.25_modes10"),
        ("Phase 3b 0.035 / Lcut_20", "mag_0.035_modes20"),
    ],
    "aifs": [
        ("Phase 1  0.01 / all", "mag_0.01_layer_all"),
        ("Phase 2  0.028 / decoder *", "mag_0.027500_layer_decoder"),
    ],
}

clim = json.load(open(CRPS_CLIM))


def clim_for(var, lvl, lead):
    key = var if var in VARS_2D else f"{var}_{int(lvl)}"
    d = clim.get(key)
    return d.get(str(lead)) if d else None


def load_crps(model_dir):
    """(run_key, variable, lead, level) -> CRPS for one ablation model."""
    path = f"{ROOT}/{model_dir}/intercomparison/probabilistic/temporal_metrics_combined.csv"
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "CRPS":
                continue
            try:
                lead = int(row["lead_time"])
                lvl = float(row["level"]) if row["level"] else None
                val = float(row["value"])
            except ValueError:
                continue
            out[(row["model"], row["variable"], lead, lvl)] = val
    return out


def crpss(data, run_key, lead):
    per_var = []
    for v in VARS_2D:
        crps_v = data.get((run_key, v, lead, None)) or data.get((run_key, v, lead, 0.0))
        c = clim_for(v, None, lead)
        if crps_v is None or c is None:
            continue
        per_var.append(1 - crps_v / c)
    for v in VARS_3D:
        skills = []
        for lvl in (500.0, 850.0):
            crps_v = data.get((run_key, v, lead, lvl))
            c = clim_for(v, lvl, lead)
            if crps_v is None or c is None:
                continue
            skills.append(1 - crps_v / c)
        if skills:
            per_var.append(sum(skills) / len(skills))
    return sum(per_var) / len(per_var) if per_var else None


for model_dir, rows in ROWS.items():
    data = load_crps(model_dir)
    table = {label: {L: crpss(data, key, L) for L in LEADS} for label, key in rows}
    best = {
        L: max((table[lbl][L] for lbl, _ in rows if table[lbl][L] is not None), default=None)
        for L in LEADS
    }
    print(f"\n=== {model_dir}  (CRPSS @ {LEADS} h; * = argmax within model) ===")
    for label, _ in rows:
        cells = []
        for L in LEADS:
            v = table[label][L]
            if v is None:
                cells.append("   -  ")
            else:
                mark = "*" if best[L] is not None and abs(v - best[L]) < 1e-9 else " "
                cells.append(f"{v:+.3f}{mark}")
        print(f"  {label:32s} " + "  ".join(cells))
