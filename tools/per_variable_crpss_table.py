"""Per-variable CRPSS breakdown table (Appendix B).

Replicates tools/headline_8way_table.py exactly, but KEEPS the per-variable
breakdown instead of averaging over the 7 paper variables. 3D variables are
averaged over levels 500/850 first (as in the pipeline). The mean of the 7
per-variable CRPSS values reproduces the variable-mean CRPSS of the headline
table (verified in this script).

Output: /users/sadamov/pyprojects/ai-models-ensembles/figures/per_variable_crpss_table.tex
"""

from __future__ import annotations
import csv
import json
from pathlib import Path

CSV_COMBINED = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/intercomparison/probabilistic/temporal_metrics_combined.csv"
CRPS_CLIM = str(Path(__file__).resolve().parent / "data" / "crps_clim_eval_1990_2019.json")
OUT = "/users/sadamov/pyprojects/ai-models-ensembles/figures/per_variable_crpss_table.tex"

AIFS_PERT_PROB = Path(
    "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/"
    "aifs_perturbed/eval/probabilistic"
)

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
# Ordered list of the 7 paper variables (2D + 3D).
VAR_ORDER = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
VAR_HEADER = {
    "2m_temperature": "2m\\_t",
    "mean_sea_level_pressure": "MSL",
    "geopotential": "$Z$",
    "temperature": "$T$",
    "u_component_of_wind": "$u$",
    "v_component_of_wind": "$v$",
    "specific_humidity": "$q$",
}
LEADS = [120, 240]

MODELS_FROM_COMBINED = [
    "aifsens",
    "atlas",
    "fcn3",
    "ifs_ens",
    "graphcast_all",
    "aurora_encoder",
    "sfno_modes10",
]
MODELS_FROM_PERBASE = ["aifs_perturbed"]
MODELS = MODELS_FROM_COMBINED + MODELS_FROM_PERBASE

PRETTY = {
    "aifsens": "AIFS-ENS",
    "atlas": "Atlas",
    "fcn3": "FCN3",
    "ifs_ens": "IFS-ENS",
    "graphcast_all": "graphcast\\_all",
    "aurora_encoder": "aurora\\_encoder",
    "sfno_modes10": "sfno\\_modes10",
    "aifs_perturbed": "aifs\\_perturbed",
}

ROLE = {
    "aurora_encoder": "post-hoc",
    "graphcast_all": "post-hoc",
    "sfno_modes10": "post-hoc",
    "aifs_perturbed": "post-hoc",
    "aifsens": "trained-prob",
    "atlas": "trained-prob",
    "fcn3": "trained-prob",
    "ifs_ens": "classical",
}

crps_clim = json.load(open(CRPS_CLIM))


def clim_for(var: str, lvl: float | None, lead: int) -> float | None:
    key = (
        var
        if var in VARS_2D
        else (f"{var}_{int(lvl)}" if (var in VARS_3D and lvl is not None) else None)
    )
    if key is None or key not in crps_clim:
        return None
    return crps_clim[key].get(str(lead))


data: dict[tuple, float] = {}
with open(CSV_COMBINED) as f:
    for row in csv.DictReader(f):
        if row["metric"] != "CRPS" or row["model"] not in MODELS_FROM_COMBINED:
            continue
        try:
            lead = int(row["lead_time"])
            lvl = float(row["level"]) if row["level"] else None
            val = float(row["value"])
        except ValueError:
            continue
        if lead not in LEADS:
            continue
        data[(row["model"], row["variable"], lead, lvl)] = val


def _load_perbase(root: Path, model: str) -> None:
    def _load(stem: str, var_label: str, level: float | None) -> None:
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
                if lead not in LEADS:
                    continue
                data[(model, var_label, lead, level)] = val

    for v in VARS_2D:
        _load(v, v, None)
    for v in VARS_3D:
        for lvl in (500, 850):
            _load(f"{v}_{lvl}", v, float(lvl))


_load_perbase(AIFS_PERT_PROB, "aifs_perturbed")


def crpss_var(model: str, var: str, lead: int) -> float | None:
    """Per-variable CRPSS. 3D vars averaged over levels 500/850 first."""
    if var in VARS_2D:
        candidates = [
            (k, val) for k, val in data.items() if k[0] == model and k[1] == var and k[2] == lead
        ]
        if not candidates:
            return None
        crps_v = candidates[0][1]
        c = clim_for(var, None, lead)
        if c is None:
            return None
        return 1 - crps_v / c
    skills = []
    for lvl in (500.0, 850.0):
        crps_v = data.get((model, var, lead, lvl))
        if crps_v is None:
            continue
        c = clim_for(var, lvl, lead)
        if c is None:
            continue
        skills.append(1 - crps_v / c)
    if not skills:
        return None
    return sum(skills) / len(skills)


per_var = {
    m: {lead: {v: crpss_var(m, v, lead) for v in VAR_ORDER} for lead in LEADS} for m in MODELS
}


def var_mean(model: str, lead: int) -> float | None:
    vals = [per_var[model][lead][v] for v in VAR_ORDER]
    vals = [x for x in vals if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


# --- Verification against the headline table -----------------------------
HEADLINE_240 = {
    "aifsens": 0.256,
    "atlas": 0.238,
    "fcn3": 0.218,
    "aifs_perturbed": 0.221,
    "graphcast_all": 0.167,
    "aurora_encoder": 0.134,
    "sfno_modes10": 0.125,
    "ifs_ens": 0.207,
}
print("Verification: mean(per-variable CRPSS@240h) vs headline_8way_table.tex")
print(f"  {'model':18s}  {'computed':>9s}  {'headline':>9s}  {'diff':>7s}")
ok = True
for m in MODELS:
    comp = var_mean(m, 240)
    head = HEADLINE_240[m]
    diff = comp - head
    flag = "" if abs(diff) < 0.01 else "  <-- MISMATCH"
    if abs(diff) >= 0.01:
        ok = False
    print(f"  {m:18s}  {comp:9.3f}  {head:9.3f}  {diff:+7.3f}{flag}")
print(f"  ALL WITHIN 0.01: {ok}\n")

print("Per-variable CRPSS@240h:")
print("  " + f"{'model':18s}" + "".join(f"{v:>10s}" for v in VAR_ORDER))
for m in MODELS:
    print(
        "  "
        + f"{m:18s}"
        + "".join(
            f"{(per_var[m][240][v] if per_var[m][240][v] is not None else float('nan')):10.3f}"
            for v in VAR_ORDER
        )
    )


def fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if v >= 0 else f"$-${abs(v):.3f}"


# Per-variable best (highest CRPSS) at each lead, for bolding.
best = {
    lead: {
        v: max(
            (per_var[m][lead][v] for m in MODELS if per_var[m][lead][v] is not None),
            default=None,
        )
        for v in VAR_ORDER
    }
    for lead in LEADS
}


def fmt_bold(v: float | None, lead: int, var: str) -> str:
    s = fmt(v)
    if v is None:
        return s
    b = best[lead][var]
    if b is not None and abs(v - b) < 1e-9:
        return f"\\textbf{{{s}}}"
    return s


order = sorted(MODELS, key=lambda m: -(var_mean(m, 240) or -99))
group_order = ["trained-prob", "post-hoc", "classical"]
role_label = {
    "post-hoc": "Post-hoc weight perturbation (this work)",
    "trained-prob": "Trained probabilistic",
    "classical": "Classical reference",
}

lines = []
lines.append("% Per-variable CRPSS table (Appendix B). Generated from")
lines.append("%   $STORE/baselines/intercomparison/probabilistic/temporal_metrics_combined.csv")
lines.append("% plus per-baseline aifs_perturbed probabilistic CSVs at")
lines.append(f"%   {AIFS_PERT_PROB}")
lines.append("% via tools/per_variable_crpss_table.py.")
lines.append("% Same pipeline as tools/headline_8way_table.py; the variable mean of each row")
lines.append("% reproduces that baseline's variable-mean CRPSS in figures/headline_8way_table.tex.")
lines.append("")
lines.append("\\begin{table}[t]")
lines.append("  \\centering")
lines.append("  \\small")
lines.append("  \\setlength{\\tabcolsep}{4.5pt}")
lines.append("  \\caption{Per-variable CRPSS at 240\\,h on the 112-init production grid")
lines.append("           (112 initialisations $\\times$ 10 members, 2023-2024), broken out by")
lines.append("           the seven paper variables. Three-dimensional variables ($Z$, $T$,")
lines.append("           $u$, $v$, $q$) are averaged over 500 and 850\\,hPa first. Higher is")
lines.append("           better; 1 = perfect, 0 = climatology, $<0$ = worse than climatology.")
lines.append("           \\textbf{Bold} marks the per-variable optimum. The mean of each row")
lines.append("           reproduces that baseline's variable-mean CRPSS@240\\,h in")
lines.append("           Tab.~\\ref{tab:headline8way}. CRPSS $= 1 - \\mathrm{CRPS}/\\mathrm{CRPS_{clim}}$")
lines.append("           against the WB2 probabilistic climatology (1990-2019 years as members,")
lines.append("           lead-resolved). IFS-ENS surface fields carry WB2-archive NaN gaps")
lines.append("           (T2m $\\sim20\\%$, MSL $\\sim30\\%$); skipna averaging is applied.}")
lines.append("  \\label{tab:per-variable-crpss}")
lines.append("  \\begin{tabular}{@{}l l rrrrrrr@{}}")
lines.append("    \\toprule")
hdr = " & ".join(f"\\textbf{{{VAR_HEADER[v]}}}" for v in VAR_ORDER)
lines.append(f"    \\textbf{{Role}} & \\textbf{{Model}} & {hdr} \\\\")
lines.append("    \\midrule")
last_role = None
for role_g in group_order:
    for m in order:
        if ROLE[m] != role_g:
            continue
        if last_role != role_g:
            lines.append("    \\rowcolor{phaseBg}")
            lines.append(
                f"    \\multicolumn{{9}}{{@{{}}l}}{{\\textbf{{{role_label[role_g]}}}}} \\\\"
            )
            last_role = role_g
        cells = " & ".join(fmt_bold(per_var[m][240][v], 240, v) for v in VAR_ORDER)
        lines.append(f"    & {PRETTY[m]} & {cells} \\\\")
    if role_g != group_order[-1]:
        lines.append("    \\hline")
lines.append("    \\bottomrule")
lines.append("  \\end{tabular}")
lines.append("\\end{table}")
Path(OUT).write_text("\n".join(lines) + "\n")
print(f"\n-> {OUT}")
