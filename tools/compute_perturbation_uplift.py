"""Compute per-model CRPSS uplift: CRPSS(perturbed) - CRPSS(mag_0 MAE).

For mag_0 (unperturbed N=1), fair CRPS == MAE, so CRPSS is well-defined.
Variable mean over the 7 paper variables (2m_t, MSL, Z, T, u, v, q),
3D variables averaged over 500 + 850 hPa first. Ablation grid: 4 inits.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Exact-WB2 ablation-grid CRPS_clim denominator (4 mid-season inits, lead-resolved).
CRPS_CLIM = str(Path(__file__).resolve().parent / "data" / "crps_clim_eval_ablation_1990_2019.json")
ROOT = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation/allphases"

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]

# Production picks
PICKS = {
    "aurora": ("mag_0.025_layer_encoder", "Aurora (aurora_encoder)"),
    "graphcast_operational": ("mag_0.01_layer_all", "GraphCast (graphcast_all)"),
    "sfno": ("mag_0.25_modes10", "SFNO (sfno_modes10)"),
}
LEAD = 240

clim = json.load(open(CRPS_CLIM))


def crps_clim(v, lvl):
    key = v if v in VARS_2D else f"{v}_{int(lvl)}"
    d = clim.get(key)
    return d.get(str(LEAD)) if d else None


def load_metric(model_dir, key_model, metric):
    csv_path = (
        f"{ROOT}/{model_dir}/intercomparison/"
        f"{'probabilistic' if metric == 'CRPS' else 'deterministic'}/"
        f"temporal_metrics_combined.csv"
    )
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["model"] != key_model or row["metric"] != metric:
                continue
            try:
                lead = int(row["lead_time"])
                if lead != LEAD:
                    continue
                lvl = float(row["level"]) if row["level"] else None
                out[(row["variable"], lvl)] = float(row["value"])
            except ValueError:
                continue
    return out


def crpss_var_mean(model_score):
    per_var = []
    for v in VARS_2D:
        val = model_score.get((v, None)) or model_score.get((v, 0.0))
        if val is None:
            continue
        c = crps_clim(v, None)
        if c is None:
            continue
        per_var.append(1 - val / c)
    for v in VARS_3D:
        ss = []
        for lvl in (500.0, 850.0):
            val = model_score.get((v, lvl))
            if val is None:
                continue
            c = crps_clim(v, lvl)
            if c is None:
                continue
            ss.append(1 - val / c)
        if ss:
            per_var.append(sum(ss) / len(ss))
    return sum(per_var) / len(per_var) if per_var else None


rows = []
for model_dir, (key_perturbed, pretty) in PICKS.items():
    perturbed = load_metric(model_dir, key_perturbed, "CRPS")
    mag0 = load_metric(model_dir, "mag_0_layer_all", "MAE")
    crpss_pert = crpss_var_mean(perturbed)
    crpss_mag0 = crpss_var_mean(mag0)
    delta = crpss_pert - crpss_mag0
    rows.append((pretty, crpss_pert, crpss_mag0, delta))

print(f"{'Model':35s} {'CRPSS(pert)':>12s} {'CRPSS(mag_0)':>13s} {'Delta':>10s}")
print("-" * 75)
for pretty, p, m, d in rows:
    print(f"{pretty:35s} {p:>12.4f} {m:>13.4f} {d:>+10.4f}")
print()
print("Note: mag_0 CRPSS uses MAE (fair-CRPS reduces to MAE when M=1 effectively).")
print(f"      Lead = {LEAD} h, 7 paper variables, 3D averaged over 500+850 first.")
