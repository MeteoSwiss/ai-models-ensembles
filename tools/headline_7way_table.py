"""Generate the headline 7-way CRPSS table and write it as a standalone
.tex file ready to \\input into main.tex.

Output: /users/sadamov/pyprojects/ai-models-ensembles/figures/headline_7way_table.tex
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

CSV = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/intercomparison/probabilistic/temporal_metrics_combined.csv"
# Superseded by headline_8way_table.py (the paper table); kept for the 7-baseline
# view. Uses the same exact-WB2 lead-resolved CRPS_clim denominator.
CRPS_CLIM = str(Path(__file__).resolve().parent / "data" / "crps_clim_eval_1990_2019.json")
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/headline_7way_table.tex"

# Paper variables, 3D need 500 + 850 averaging
VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEADS = [72, 120, 240, 360]

# Models to include (paper-headline 7)
MODELS = ["aifsens", "atlas", "fcn3", "ifs_ens", "graphcast_all", "aurora_encoder", "sfno_modes10"]

# Pretty names
PRETTY = {
    "aifsens": "AIFS-ENS",
    "atlas": "Atlas",
    "fcn3": "FCN3",
    "ifs_ens": "IFS-ENS",
    "graphcast_all": "graphcast\\_all",
    "aurora_encoder": "aurora\\_encoder",
    "sfno_modes10": "sfno\\_modes10",
}

crps_clim = json.load(open(CRPS_CLIM))


def clim_for(var: str, lvl: float | None, lead: int) -> float | None:
    key = var if var in VARS_2D else (f"{var}_{int(lvl)}" if lvl is not None else None)
    if key is None or key not in crps_clim:
        return None
    return crps_clim[key].get(str(lead))


# Pull CRPS per (model, var, lead, level)
data: dict[tuple, float] = {}
with open(CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        if row["metric"] != "CRPS":
            continue
        if row["model"] not in MODELS:
            continue
        try:
            lead = int(row["lead_time"])
            lvl = float(row["level"]) if row["level"] else None
            val = float(row["value"])
        except ValueError:
            continue
        if lead not in LEADS:
            continue
        if row["variable"] in VARS_2D and lvl not in (0.0, 1000.0, None) and not (lvl == lvl):
            pass  # 2D should not have a meaningful level; some CSVs put 0
        data[(row["model"], row["variable"], lead, lvl)] = val


# Compute per-(var, lead) CRPSS per model, then per-variable (averaging levels for 3D),
# then mean across the 7 paper vars.
def crpss(model: str, lead: int) -> float | None:
    per_var = []
    for v in VARS_2D:
        # find any 2D entry (lvl could be 0 or None or NaN)
        candidates = [
            (k, val) for k, val in data.items() if k[0] == model and k[1] == v and k[2] == lead
        ]
        if not candidates:
            return None
        crps_v = candidates[0][1]
        c = clim_for(v, None, lead)
        if c is None:
            return None
        per_var.append(1 - crps_v / c)
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
        if not skills:
            return None
        per_var.append(sum(skills) / len(skills))
    if not per_var:
        return None
    return sum(per_var) / len(per_var)


# Build the table
rows = {m: {lead: crpss(m, lead) for lead in LEADS} for m in MODELS}

# Find per-lead max for bolding
maxes = {
    lead: max((v for v in (rows[m][lead] for m in MODELS) if v is not None), default=None)
    for lead in LEADS
}

# Sort models by CRPSS at 240h descending (for the table row order)
order = sorted(MODELS, key=lambda m: -(rows[m][240] or -99))


# Render LaTeX
def fmt(v: float | None, m: str, lead: int) -> str:
    if v is None:
        return "--"
    s = f"{v:.3f}"
    if maxes[lead] is not None and abs(v - maxes[lead]) < 1e-9:
        s = f"\\textbf{{{s}}}"
    return s


# Compose colour pills for "perturbed" vs "trained-prob" vs "classical"
ROLE = {
    "aurora_encoder": "post-hoc",
    "graphcast_all": "post-hoc",
    "sfno_modes10": "post-hoc",
    "aifsens": "trained-prob",
    "atlas": "trained-prob",
    "fcn3": "trained-prob",
    "ifs_ens": "classical",
}

lines = []
lines.append("% Headline 7-way CRPSS table. Generated from")
lines.append("%   $STORE/baselines/intercomparison/probabilistic/temporal_metrics_combined.csv")
lines.append("% via tools_or_scratch/headline_7way_table.py.")
lines.append("% 112 inits x 10 members, 7 paper vars (MSL EXCLUDED due to ifs_ens MSL bug,")
lines.append("% see memory ifs-ens-msl-bug.md; re-run after patch). CRPSS = 1 - CRPS/CRPS_clim")
lines.append("% against the WB2 probabilistic climatology (1990-2019, fair CRPS vs eval truth).")
lines.append("")
lines.append("\\begin{table}[t]")
lines.append("  \\centering")
lines.append("  \\small")
lines.append("  \\setlength{\\tabcolsep}{6pt}")
lines.append("  \\caption{Headline 7-way CRPSS@(72,120,240,360)h on the production grid")
lines.append("           (112 weekly initialisations $\\times$ 10 members in 2024).")
lines.append("           Higher is better; 1 = perfect, 0 = climatology, $<0$ = worse")
lines.append("           than climatology. \\textbf{Bold} marks the per-lead optimum across")
lines.append("           all seven baselines. MSL excluded due to an IFS-ENS MSL bug")
lines.append("           (see Sec.~\\ref{sec:methods-eval}); re-run with the patched data")
lines.append("           is pending. Derived variables (geopotential height, wind speed,")
lines.append("           gradient) excluded because no WB2 climatology $\\sigma$ is")
lines.append("           available for them.}")
lines.append("  \\label{tab:headline7way}")
lines.append("  \\begin{tabular}{@{}l l rrrr@{}}")
lines.append("    \\toprule")
lines.append(
    "    \\textbf{Role} & \\textbf{Model} & \\textbf{72\\,h} & \\textbf{120\\,h} & \\textbf{240\\,h} & \\textbf{360\\,h} \\\\"
)
lines.append("    \\midrule")
# Group by role: trained-prob first, then post-hoc, then classical (for narrative)
group_order = ["trained-prob", "post-hoc", "classical"]
last_role = None
for role_g in group_order:
    for m in order:
        if ROLE[m] != role_g:
            continue
        role_label = {
            "post-hoc": "Post-hoc weight perturbation (this work)",
            "trained-prob": "Trained probabilistic",
            "classical": "Classical reference",
        }[role_g]
        if last_role != role_g:
            lines.append("    \\rowcolor{phaseBg}")
            lines.append(f"    \\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{role_label}}}}} \\\\")
            last_role = role_g
        cells = [fmt(rows[m][lead], m, lead) for lead in LEADS]
        lines.append(f"    & {PRETTY[m]} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
    if role_g != group_order[-1]:
        lines.append("    \\midrule")
lines.append("    \\bottomrule")
lines.append("  \\end{tabular}")
lines.append("\\end{table}")
Path(OUT).write_text("\n".join(lines))
print(f"-> {OUT}")
print("\nFinal table:")
for m in order:
    print(
        f"  {PRETTY[m]:25s}  "
        + "  ".join(
            f"{(rows[m][lead] if rows[m][lead] is not None else float('nan')):.3f}"
            for lead in LEADS
        )
    )
