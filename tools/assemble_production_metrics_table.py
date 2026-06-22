"""Assemble the production-grid calibration pendant (Tab. production-metrics, App. C).

Production analogue of figures/calibration_basis_table.tex: one row per production
baseline (the per-model picks + trained-prob + classical reference), two lead blocks
(120, 240 h), each block carrying CRPSS, per-pixel SSR, SSIM, LSD, FSS 95% and W1
(variable-mean over the 7 paper variables; 3D variables averaged over 500/850 hPa
first). The 240 h block additionally carries the published multivariate ES / VS / SIGK.

Single source per metric, all on the 112-init production grid:
  CRPSS  - $STORE/baselines/intercomparison/probabilistic/temporal_metrics_combined.csv
           (+ aifs_perturbed per-baseline crps_line CSVs), per_variable_crpss_table.py logic.
  SSR    - per-baseline probabilistic/ssr_line_<var>[_<lvl>]_by_lead_ensprob.csv (per-pixel,
           lead-resolved; the intercomparison ssr_combined.csv is time-mean only).
  SSIM   - per-baseline ssim/ssim_ssim_by_lead_ensmean.csv.
  LSD    - per-baseline energy_spectra/energy_ratios_3d_lead_time_*_enspooled.csv (lsd_mean).
  FSS95  - per-baseline fss/fss_metrics_per_member_per_lead_*_ensmembers.csv (mean over members).
  W1     - per-baseline wd_kde/wd_kde_wasserstein_averaged_enspooled.csv (global hemisphere).
The per-baseline files reproduce the intercomparison combined CSVs to machine precision
(verified); they are used uniformly so aifs_perturbed (absent from the combined CSVs) is
handled by the same code path.

ES / VS / SIGK @ 240 h are taken verbatim from the published verified values (not recomputed).

Writes figures/table_c1_production_metrics.tex and prints the CRPSS reproduction check
against figures/headline_8way_table.tex.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

STORE = Path("/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles")
BASE = STORE / "baselines"
INTER = BASE / "intercomparison"
CRPS_CLIM = Path(__file__).resolve().parent / "data" / "crps_clim_eval_1990_2019.json"
OUT = Path("/users/sadamov/pyprojects/ai-models-ensembles/figures/table_c1_production_metrics.tex")

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
VAR_ORDER = VARS_2D + VARS_3D
LEADS = [120, 240]

MODELS = [
    "aurora_encoder",
    "graphcast_all",
    "sfno_modes10",
    "aifs_perturbed",
    "aifsens",
    "atlas",
    "fcn3",
    "ifs_ens",
]
MODELS_FROM_COMBINED = [m for m in MODELS if m != "aifs_perturbed"]

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
GROUP_ORDER = ["post-hoc", "trained-prob", "classical"]
ROLE_LABEL = {
    "post-hoc": "Post-hoc weight perturbation (this work)",
    "trained-prob": "Trained probabilistic",
    "classical": "Classical reference",
}

# Verified ES / VS / SIGK at 120 and 240 h on the 112-init production grid,
# from tools/submit_table_metrics_multilead.sh (job 2579121, 2026-06-21):
#   scratch/table_metrics/{esvs,sigk}_<model>_L{120,240}.csv.
# The four ablation-winner cells reproduce figures/calibration_basis_table.tex
# exactly; aifs_perturbed ES@240 = 1.054 supersedes the earlier untraceable 1.050.
# IFS-ENS is absent here (only ~21-23/112 inits carry all 7 variables jointly).
ES_120 = {
    "aifsens": 0.623,
    "atlas": 0.651,
    "aifs_perturbed": 0.669,
    "fcn3": 0.675,
    "aurora_encoder": 0.730,
    "graphcast_all": 0.735,
    "sfno_modes10": 0.850,
}
ES_240 = {
    "aifsens": 1.018,
    "atlas": 1.038,
    "aifs_perturbed": 1.054,
    "fcn3": 1.061,
    "graphcast_all": 1.111,
    "aurora_encoder": 1.117,
    "sfno_modes10": 1.156,
}
VS_120 = {
    "aifsens": 0.0426,
    "atlas": 0.0462,
    "aifs_perturbed": 0.0470,
    "fcn3": 0.0479,
    "graphcast_all": 0.0500,
    "aurora_encoder": 0.0511,
    "sfno_modes10": 0.0646,
}
VS_240 = {
    "aifsens": 0.0840,
    "atlas": 0.0858,
    "aifs_perturbed": 0.0869,
    "fcn3": 0.0880,
    "graphcast_all": 0.0908,
    "aurora_encoder": 0.0915,
    "sfno_modes10": 0.0958,
}
SIGK_120 = {
    "aifsens": -26.2,
    "atlas": -24.2,
    "fcn3": -22.1,
    "aifs_perturbed": -16.2,
    "aurora_encoder": -13.9,
    "graphcast_all": -6.9,
    "sfno_modes10": -4.1,
}
SIGK_240 = {
    "aifsens": -37.1,
    "atlas": -33.1,
    "aifs_perturbed": -12.2,
    "fcn3": -29.4,
    "graphcast_all": 18.1,
    "aurora_encoder": -9.9,
    "sfno_modes10": 1.0,
}
MVAR = {
    120: {"ES": ES_120, "VS": VS_120, "SIGK": SIGK_120},
    240: {"ES": ES_240, "VS": VS_240, "SIGK": SIGK_240},
}

crps_clim = json.load(open(CRPS_CLIM))


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------- CRPSS
def clim_for(var, lvl, lead):
    key = (
        var
        if var in VARS_2D
        else (f"{var}_{int(lvl)}" if (var in VARS_3D and lvl is not None) else None)
    )
    if key is None or key not in crps_clim:
        return None
    return crps_clim[key].get(str(lead))


def load_crps():
    data = {}
    path = INTER / "probabilistic" / "temporal_metrics_combined.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "CRPS" or row["model"] not in MODELS_FROM_COMBINED:
                continue
            try:
                lead = int(row["lead_time"])
                lvl = float(row["level"]) if row["level"] else None
                val = float(row["value"])
            except ValueError:
                continue
            if lead in LEADS:
                data[(row["model"], row["variable"], lead, lvl)] = val
    # aifs_perturbed from per-baseline crps_line.
    root = BASE / "aifs_perturbed" / "eval" / "probabilistic"

    def load_line(stem, var_label, level):
        p = root / f"crps_line_{stem}_by_lead_ensprob.csv"
        if not p.exists():
            return
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    lead = int(row["lead_time_hours"])
                    val = float(row["CRPS"])
                except ValueError:
                    continue
                if lead in LEADS:
                    data[("aifs_perturbed", var_label, lead, level)] = val

    for v in VARS_2D:
        load_line(v, v, None)
    for v in VARS_3D:
        for lvl in (500, 850):
            load_line(f"{v}_{lvl}", v, float(lvl))
    return data


def crpss(data, model, lead):
    per_var = []
    for v in VARS_2D:
        crps_v = data.get((model, v, lead, None))
        if crps_v is None:
            crps_v = data.get((model, v, lead, 0.0))
        c = clim_for(v, None, lead)
        if crps_v is None or c is None:
            continue
        per_var.append(1 - crps_v / c)
    for v in VARS_3D:
        skills = []
        for lvl in (500.0, 850.0):
            crps_v = data.get((model, v, lead, lvl))
            c = clim_for(v, lvl, lead)
            if crps_v is None or c is None:
                continue
            skills.append(1 - crps_v / c)
        if skills:
            per_var.append(sum(skills) / len(skills))
    return _mean(per_var)


# --------------------------------------------------------------------------- SSR (per-pixel)
def ssr(model, lead):
    root = BASE / model / "eval" / "probabilistic"

    def read(stem):
        p = root / f"ssr_line_{stem}_by_lead_ensprob.csv"
        if not p.exists():
            return None
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    if int(row["lead_time_hours"]) == lead:
                        return float(row["SSR"])
                except (ValueError, KeyError):
                    continue
        return None

    per_var = []
    for v in VARS_2D:
        per_var.append(read(v))
    for v in VARS_3D:
        lv = _mean([read(f"{v}_500"), read(f"{v}_850")])
        per_var.append(lv)
    return _mean(per_var)


# --------------------------------------------------------------------------- SSIM
def ssim(model, lead):
    p = BASE / model / "eval" / "ssim" / "ssim_ssim_by_lead_ensmean.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))

    def get(var):
        for r in rows:
            try:
                if r["variable"] == var and int(r["lead_time_hours"]) == lead:
                    return float(r["SSIM"])
            except (ValueError, KeyError):
                continue
        return None

    per_var = [get(v) for v in VAR_ORDER]
    return _mean(per_var)


# --------------------------------------------------------------------------- LSD
def lsd(model, lead):
    d = BASE / model / "eval" / "energy_spectra"
    matches = list(d.glob("energy_ratios_3d_lead_time_*_enspooled.csv"))
    if not matches:
        return None
    rows = list(csv.DictReader(open(matches[0])))
    tag = f"{lead:03d}h"

    def get(var):
        for r in rows:
            if r["variable"] == var and r["lead_time"] == tag:
                try:
                    return float(r["lsd_mean"])
                except ValueError:
                    return None
        return None

    # LSD per-baseline 3d file carries the 3D variables only.
    per_var = [get(v) for v in VARS_3D]
    return _mean(per_var)


# --------------------------------------------------------------------------- FSS 95%
def fss95(model, lead):
    d = BASE / model / "eval" / "fss"
    matches = list(d.glob("fss_metrics_per_member_per_lead_*_ensmembers.csv"))
    if not matches:
        return None
    rows = list(csv.DictReader(open(matches[0])))

    def get(var):
        vals = []
        for r in rows:
            try:
                if r["variable"] != var or float(r["lead_time_hours"]) != float(lead):
                    continue
                v = float(r["FSS 95%"])
            except (ValueError, KeyError):
                continue
            if not math.isnan(v):
                vals.append(v)
        return _mean(vals)

    per_var = [get(v) for v in VAR_ORDER]
    return _mean(per_var)


# --------------------------------------------------------------------------- W1
def w1(model, lead):
    p = BASE / model / "eval" / "wd_kde" / "wd_kde_wasserstein_averaged_enspooled.csv"
    if not p.exists():
        # IFS-ENS is not run through the wd_kde re-eval (its WB2 surface fields
        # carry NaN gaps); report "-" like its multivariate scores.
        return "omit" if model == "ifs_ens" else "pending"
    rows = list(csv.DictReader(open(p)))

    def get(var, level):
        for r in rows:
            try:
                if (
                    r["variable"] == var
                    and r["hemisphere"] == "global"
                    and int(r["lead_time_hours"]) == lead
                ):
                    lvl = r["level"]
                    if level is None and lvl == "":
                        return float(r["wasserstein"])
                    if level is not None and lvl != "" and int(float(lvl)) == int(level):
                        return float(r["wasserstein"])
            except (ValueError, KeyError):
                continue
        return None

    per_var = []
    for v in VARS_2D:
        per_var.append(get(v, None))
    for v in VARS_3D:
        per_var.append(_mean([get(v, 500), get(v, 850)]))
    return _mean(per_var)


# --------------------------------------------------------------------------- assemble
def main():
    data = load_crps()
    M = {m: {L: {} for L in LEADS} for m in MODELS}
    for m in MODELS:
        for L in LEADS:
            M[m][L]["CRPSS"] = crpss(data, m, L)
            M[m][L]["SSR"] = ssr(m, L)
            M[m][L]["SSIM"] = ssim(m, L)
            M[m][L]["LSD"] = lsd(m, L)
            M[m][L]["FSS"] = fss95(m, L)
            M[m][L]["W1"] = w1(m, L)
            M[m][L]["ES"] = MVAR[L]["ES"].get(m)
            M[m][L]["VS"] = MVAR[L]["VS"].get(m)
            M[m][L]["SIGK"] = MVAR[L]["SIGK"].get(m)

    # ---- CRPSS reproduction check vs headline_8way_table.tex --------------
    headline = {
        120: {
            "aifsens": 0.580,
            "atlas": 0.558,
            "fcn3": 0.534,
            "aifs_perturbed": 0.542,
            "graphcast_all": 0.489,
            "aurora_encoder": 0.471,
            "sfno_modes10": 0.388,
            "ifs_ens": 0.518,
        },
        240: {
            "aifsens": 0.256,
            "atlas": 0.238,
            "fcn3": 0.218,
            "aifs_perturbed": 0.221,
            "graphcast_all": 0.167,
            "aurora_encoder": 0.134,
            "sfno_modes10": 0.125,
            "ifs_ens": 0.207,
        },
    }
    print("CRPSS reproduction vs figures/headline_8way_table.tex (must be <0.005):")
    print(f"  {'model':16s} {'lead':>4s} {'computed':>9s} {'headline':>9s} {'diff':>7s}")
    worst = 0.0
    for L in LEADS:
        for m in MODELS:
            c = M[m][L]["CRPSS"]
            h = headline[L][m]
            d = c - h
            worst = max(worst, abs(d))
            flag = "" if abs(d) < 0.005 else "  <-- MISMATCH"
            print(f"  {m:16s} {L:4d} {c:9.3f} {h:9.3f} {d:+7.3f}{flag}")
    print(f"  worst |diff| = {worst:.4f}\n")

    pending = [m for m in MODELS if M[m][120]["W1"] == "pending"]
    print(f"W1 pending (no wd_kde CSV yet): {pending}\n")

    # ---- per-(lead, metric) winners for bolding ---------------------------
    METRIC_ORDER = ["CRPSS", "SSR", "SSIM", "LSD", "FSS", "W1", "ES", "VS", "SIGK"]
    HIGHER = {"CRPSS", "SSIM", "FSS"}  # argmax; SSR is argmin|x-1|; rest argmin

    def best_model(lead, metric):
        cand = []
        for m in MODELS:
            v = M[m][lead][metric]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                cand.append((m, float(v)))
        if not cand:
            return None
        if metric in HIGHER:
            return max(cand, key=lambda t: t[1])[0]
        if metric == "SSR":
            return min(cand, key=lambda t: abs(t[1] - 1.0))[0]
        return min(cand, key=lambda t: t[1])[0]

    WIN = {(lead, mt): best_model(lead, mt) for lead in LEADS for mt in METRIC_ORDER}

    # ---- emit LaTeX -------------------------------------------------------
    FMT = {
        "CRPSS": "{:.3f}",
        "SSR": "{:.2f}",
        "SSIM": "{:.3f}",
        "LSD": "{:.3f}",
        "FSS": "{:.3f}",
        "ES": "{:.3f}",
        "VS": "{:.4f}",
        "SIGK": "{:.1f}",
    }

    def cell(m, lead, metric):
        v = M[m][lead][metric]
        if metric == "W1":
            if v == "pending":
                return "(pending)"
            if v == "omit" or v is None:
                return "-"
            s = f"{v:.3f}"
        else:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "-"
            s = FMT[metric].format(v)
        return f"\\textbf{{{s}}}" if WIN[(lead, metric)] == m else s

    L = []
    L.append("% Production-grid calibration pendant to Tab.~\\ref{tab:calibration}.")
    L.append("% Generated by tools/assemble_production_metrics_table.py from the cached 112-init")
    L.append("% production intercomparison + per-baseline eval CSVs under")
    L.append("%   $STORE/baselines/{intercomparison,<model>/eval}/...")
    L.append(
        "% CRPSS reproduces figures/headline_8way_table.tex to <0.005 (per_variable_crpss_table.py"
    )
    L.append(
        "% logic). SSR = per-pixel Fortin (M+1)/M (per-baseline ssr_line CSVs). SSIM/LSD/FSS from"
    )
    L.append(
        "% the per-baseline by-lead CSVs (= intercomparison combined CSVs to machine precision)."
    )
    L.append(
        "% W1 from per-baseline wd_kde wasserstein (global). IFS-ENS W1 = - (not in the re-eval; NaN-gap surface fields)."
    )
    L.append(
        "% ES/VS/SIGK at 120+240h from tools/submit_table_metrics_multilead.sh (job 2579121, 2026-06-21);"
    )
    L.append("% bold = per-metric per-lead optimum over the eight rows.")
    L.append("% 7-variable mean (3D vars averaged over 500/850 hPa first; LSD over the 3D vars).")
    L.append("")
    L.append("\\begin{sidewaystable*}[p]")
    L.append("  \\centering")
    L.append("  \\scriptsize")
    L.append("  \\setlength{\\tabcolsep}{3pt}")
    L.append("  \\renewcommand{\\arraystretch}{1.15}")
    L.append("  \\caption{Production-grid (112 initialisations $\\times$ 10 members, 2023-2024)")
    L.append("           multi-metric calibration for the seven production baselines plus the")
    L.append("           classical IFS-ENS reference, the deployment-scale pendant to")
    L.append("           Tab.~\\ref{tab:calibration}. Two lead blocks (120, 240\\,h); 7-variable")
    L.append(
        "           mean (3D variables averaged over $\\ell\\in\\{500,850\\}$\\,hPa first; LSD over"
    )
    L.append("           the five 3D variables). Column arrows give each metric's optimum")
    L.append("           direction (SSR targets 1; SIGK on its native scale, ranking only).")
    L.append(
        "           \\textbf{Bold}: per-metric per-lead optimum across the eight rows. IFS-ENS"
    )
    L.append("           is excluded from ES/VS/SIGK and $W_1$ (its WeatherBench-2 forecasts carry")
    L.append(
        "           archive NaN gaps in every variable, leaving too few jointly-complete initialisations).}"
    )
    L.append("  \\label{tab:production-metrics}")
    L.append("  \\begin{tabular}{@{}l l rrrrrrrrr rrrrrrrrr@{}}")
    L.append("    \\toprule")
    L.append("    \\multirow{2}{*}{\\textbf{Role}} & \\multirow{2}{*}{\\textbf{Model}}")
    L.append("        & \\multicolumn{9}{c}{\\textbf{120\\,h}}")
    L.append("        & \\multicolumn{9}{c}{\\textbf{240\\,h}} \\\\")
    L.append("    \\cmidrule(lr){3-11} \\cmidrule(lr){12-20}")
    L.append(
        "     &  & CRPSS$\\uparrow$ & SSR & SSIM$\\uparrow$ & LSD$\\downarrow$ & FSS$\\uparrow$ & $W_1\\downarrow$ & ES$\\downarrow$ & VS$\\downarrow$ & SIGK"
    )
    L.append(
        "        & CRPSS$\\uparrow$ & SSR & SSIM$\\uparrow$ & LSD$\\downarrow$ & FSS$\\uparrow$ & $W_1\\downarrow$ & ES$\\downarrow$ & VS$\\downarrow$ & SIGK \\\\"
    )
    L.append("    \\midrule")

    first_group = True
    for g in GROUP_ORDER:
        members = [m for m in MODELS if ROLE[m] == g]
        if not members:
            continue
        if not first_group:
            L.append("    \\hline")
        first_group = False
        L.append("    \\rowcolor{phaseBg}")
        L.append(f"    \\multicolumn{{20}}{{@{{}}l}}{{\\textbf{{{ROLE_LABEL[g]}}}}} \\\\")
        for m in members:
            cells = [cell(m, lead, mt) for lead in LEADS for mt in METRIC_ORDER]
            L.append(f"    & {PRETTY[m]} & " + " & ".join(cells) + " \\\\")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("\\end{sidewaystable*}")
    OUT.write_text("\n".join(L) + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
