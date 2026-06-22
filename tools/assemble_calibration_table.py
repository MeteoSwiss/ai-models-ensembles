"""Assemble the calibration-basis table (Tab. calibration, App. C) -
figures/calibration_basis_table.tex - on a single current eval vintage.

Ablation analogue of tools/assemble_production_metrics_table.py: one row per
(model, phase) calibration pick, two lead blocks (120, 240 h), each carrying
CRPSS, per-pixel Fortin SSR, SSIM, LSD, FSS 95%, W1, and the multivariate
ES / VS_0.5 / SIGK. Variable-mean over the 7 paper variables (3D variables
averaged over 500/850 hPa first); LSD over the 5 3D variables only.

Single source per metric (all the ablation 4-init x 10-member grid):
  CRPSS  - <intercomp>/probabilistic/temporal_metrics_combined.csv + WB2 ablation
           climatology (tools/data/crps_clim_eval_ablation_1990_2019.json),
           calibration_crpss.py logic. Reproduces the published CRPSS exactly.
  SSIM   - <intercomp>/ssim/ssim_by_lead_combined.csv (level-collapsed, 7-var mean).
  LSD    - <intercomp>/energy_spectra/lsd_metrics_3d_lead_time_combined.csv
           (lsd_mean, 5 3D vars).
  FSS95  - <intercomp>/fss/fss_per_member_per_lead_combined.csv (mean over members, 7 vars).
  W1     - per-run eval/<run>/wd_kde/wd_kde_wasserstein_averaged_enspooled.csv
           (global hemisphere, 7-var, 3D avg 500/850).
  SSR    - the 15 non-frozen rows keep the published current-Fortin values
           (FortinSpreadSkillRatio; the 2026-06-16 scratch source was purged), the
           two frozen GraphCast rows are read fresh from their Fortin ssr_line CSVs.
  ES/VS  - scratch/table_metrics/esvs_<label>_L{120,240}.csv.
  SIGK   - scratch/table_metrics/sigk_<label>_L{120,240}.csv.

<intercomp> is allphases/<model>/intercomparison for the non-frozen rows and
phase3{,b}/graphcast_operational/intercomparison for the two frozen GraphCast
rows (which exist only in the frozen tree). The per-variable values are
byte-identical across the 05-27 and 06-21 eval vintages (verified), so the mix
is a single consistent vintage.

Prints a per-cell comparison against the values currently in
figures/calibration_basis_table.tex and the recomputed bold marks (within-model
column optimum: argmax CRPSS/SSIM/FSS95; argmin|x-1| SSR; argmin LSD/W1/ES/VS/SIGK).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

STORE = Path("/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles")
ABL = STORE / "ablation"
SCRATCH = Path("/iopsstor/scratch/cscs/sadamov/ai-models-ensembles/scratch/table_metrics")
CRPS_CLIM = Path(__file__).resolve().parent / "data" / "crps_clim_eval_ablation_1990_2019.json"

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
VAR7 = VARS_2D + VARS_3D
LEADS = [120, 240]

crps_clim = json.load(open(CRPS_CLIM))


# Per-run eval files (W1, ssr_line) live under the phase-specific tree; the
# combined intercomparison CSVs (CRPSS/SSIM/LSD/FSS) under `phase_dir` (allphases
# for the non-frozen rows, phase3{,b} for the two frozen GraphCast rows).
_PHASE_EVAL = {
    "Phase 1": "phase1",
    "Phase 2": "phase2",
    "Phase 2b": "phase2b",
    "Phase 3": "phase3",
    "Phase 3b": "phase3b",
}


# row: model_dir, phase, cfg (LaTeX), run_key, intercomp subpath, eval_run subpath,
#      ssr ("verified" tuple or "ssr_line"), esvs label, winner
def R(model_dir, phase, cfg, run_key, phase_dir, ssr, esvs, winner=False):
    inter = ABL / phase_dir / model_dir / "intercomparison"
    evrun = ABL / _PHASE_EVAL[phase] / model_dir / "eval" / run_key
    return dict(
        model=model_dir,
        phase=phase,
        cfg=cfg,
        key=run_key,
        inter=inter,
        evrun=evrun,
        ssr=ssr,
        esvs=esvs,
        winner=winner,
    )


ROWS = [
    # aurora
    R(
        "aurora",
        "Phase 1",
        "0.03 / all",
        "mag_0.03_layer_all",
        "allphases",
        ("verified", 1.55, 1.34),
        "abl_aurora_p1",
    ),
    R(
        "aurora",
        "Phase 2",
        "0.044 / encoder",
        "mag_0.044176_layer_encoder",
        "allphases",
        ("verified", 1.63, 1.34),
        "abl_aurora_p2",
    ),
    R(
        "aurora",
        "Phase 2b",
        r"0.025 / encoder \,\winner",
        "mag_0.025_layer_encoder",
        "allphases",
        ("verified", 1.14, 1.16),
        "abl_aurora_enc",
        winner=True,
    ),
    R(
        "aurora",
        "Phase 3",
        r"0.40 / \texttt{enc\_2}",
        "mag_0.40_layer_unet_bottom",
        "allphases",
        ("verified", 0.53, 0.75),
        "abl_aurora_p3",
    ),
    R(
        "aurora",
        "Phase 3b",
        r"0.015 / \texttt{enc\_012}",
        "mag_0.015_layer_enc_012",
        "allphases",
        ("verified", 0.35, 0.62),
        "abl_aurora_p3b",
    ),
    # graphcast
    R(
        "graphcast_operational",
        "Phase 1",
        r"0.01 / all \,\winner",
        "mag_0.01_layer_all",
        "allphases",
        ("verified", 1.13, 1.34),
        "abl_graphcast_all",
        winner=True,
    ),
    R(
        "graphcast_operational",
        "Phase 2",
        "0.030 / m2g",
        "mag_0.029665_layer_m2g",
        "allphases",
        ("verified", 1.09, 1.29),
        "abl_graphcast_p2",
    ),
    R(
        "graphcast_operational",
        "Phase 2b",
        "0.014 / g2m",
        "mag_0.014_layer_g2m",
        "allphases",
        ("verified", 1.27, 1.46),
        "abl_graphcast_p2b",
    ),
    R(
        "graphcast_operational",
        "Phase 3",
        r"1.00 / \texttt{n\_coarse\_42} (frozen)",
        "gcsigma_1.0_gcnodes42_frozen",
        "phase3",
        "ssr_line",
        "abl_graphcast_p3frozen",
    ),
    R(
        "graphcast_operational",
        "Phase 3b",
        r"0.159 / \texttt{n\_coarse\_162} (frozen)",
        "gcsigma_0.159_gcnodes162_frozen",
        "phase3b",
        "ssr_line",
        "abl_graphcast_p3bfrozen",
    ),
    # sfno
    R(
        "sfno",
        "Phase 1",
        "0.03 / all",
        "mag_0.03_layer_all",
        "allphases",
        ("verified", 1.83, 1.69),
        "abl_sfno_p1",
    ),
    R(
        "sfno",
        "Phase 2",
        "0.054 / encoder",
        "mag_0.053852_layer_encoder",
        "allphases",
        ("verified", 1.25, 1.27),
        "abl_sfno_p2",
    ),
    R(
        "sfno",
        "Phase 2b",
        "0.035 / encoder",
        "mag_0.035_layer_encoder",
        "allphases",
        ("verified", 0.94, 1.12),
        "abl_sfno_p2b",
    ),
    R(
        "sfno",
        "Phase 3",
        r"0.25 / \texttt{Lcut\_10} \,\winner",
        "mag_0.25_modes10",
        "allphases",
        ("verified", 1.07, 1.22),
        "abl_sfno_modes10",
        winner=True,
    ),
    R(
        "sfno",
        "Phase 3b",
        r"0.035 / \texttt{Lcut\_20}",
        "mag_0.035_modes20",
        "allphases",
        ("verified", 0.25, 0.54),
        "abl_sfno_p3b",
    ),
    # aifs
    R(
        "aifs",
        "Phase 1",
        "0.01 / all",
        "mag_0.01_layer_all",
        "allphases",
        ("verified", 0.74, 0.92),
        "abl_aifs_p1",
    ),
    R(
        "aifs",
        "Phase 2",
        r"0.028 / decoder \,\winner",
        "mag_0.027500_layer_decoder",
        "allphases",
        ("verified", 1.29, 1.37),
        "abl_aifs_decoder",
        winner=True,
    ),
]


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def _rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def _L(x):
    return int(float(str(x).replace("h", "")))


def clim_for(var, lvl, lead):
    key = var if var in VARS_2D else f"{var}_{int(lvl)}"
    d = crps_clim.get(key)
    return d.get(str(lead)) if d else None


def crpss(row, lead):
    rs = _rows(row["inter"] / "probabilistic" / "temporal_metrics_combined.csv")
    data = {}
    for r in rs:
        if r["metric"] != "CRPS" or r["model"] != row["key"]:
            continue
        try:
            lv = float(r["level"]) if r["level"] else None
            data[(r["variable"], int(r["lead_time"]), lv)] = float(r["value"])
        except ValueError:
            continue
    per = []
    for v in VARS_2D:
        c = data.get((v, lead, None)) or data.get((v, lead, 0.0))
        cl = clim_for(v, None, lead)
        if c is not None and cl:
            per.append(1 - c / cl)
    for v in VARS_3D:
        sk = []
        for lvl in (500.0, 850.0):
            c = data.get((v, lead, lvl))
            cl = clim_for(v, lvl, lead)
            if c is not None and cl:
                sk.append(1 - c / cl)
        if sk:
            per.append(_mean(sk))
    return _mean(per)


def ssim(row, lead):
    rs = _rows(row["inter"] / "ssim" / "ssim_by_lead_combined.csv")
    out = []
    for v in VAR7:
        xs = [
            float(r["SSIM"])
            for r in rs
            if r["model"] == row["key"] and r["variable"] == v and _L(r["lead_time_hours"]) == lead
        ]
        out.append(_mean(xs))
    return _mean(out)


def lsd(row, lead):
    rs = _rows(row["inter"] / "energy_spectra" / "lsd_metrics_3d_lead_time_combined.csv")
    out = []
    for v in VARS_3D:
        xs = [
            float(r["lsd_mean"])
            for r in rs
            if r["model"] == row["key"] and r["variable"] == v and _L(r["lead_time"]) == lead
        ]
        out.append(_mean(xs))
    return _mean(out)


def fss95(row, lead):
    rs = _rows(row["inter"] / "fss" / "fss_per_member_per_lead_combined.csv")
    out = []
    for v in VAR7:
        xs = []
        for r in rs:
            if r["model"] != row["key"] or r["variable"] != v or _L(r["lead_time_hours"]) != lead:
                continue
            try:
                x = float(r["FSS 95%"])
            except (ValueError, KeyError):
                continue
            if not math.isnan(x):
                xs.append(x)
        out.append(_mean(xs))
    return _mean(out)


def w1(row, lead):
    p = row["evrun"] / "wd_kde" / "wd_kde_wasserstein_averaged_enspooled.csv"
    if not p.exists():
        return None
    rs = _rows(p)

    def g(v, level):
        for r in rs:
            if (
                r["variable"] != v
                or r["hemisphere"] != "global"
                or _L(r["lead_time_hours"]) != lead
            ):
                continue
            lvl = r["level"]
            if level is None and lvl == "":
                return float(r["wasserstein"])
            if level is not None and lvl != "" and int(float(lvl)) == int(level):
                return float(r["wasserstein"])
        return None

    return _mean([g(v, None) for v in VARS_2D] + [_mean([g(v, 500), g(v, 850)]) for v in VARS_3D])


def ssr(row, lead):
    if row["ssr"] != "ssr_line":
        return row["ssr"][1 if lead == 120 else 2]
    root = row["evrun"] / "probabilistic"

    def read(stem):
        p = root / f"ssr_line_{stem}_by_lead_ensprob.csv"
        if not p.exists():
            return None
        for r in _rows(p):
            try:
                if int(float(r["lead_time_hours"])) == lead:
                    return float(r["SSR"])
            except (ValueError, KeyError):
                continue
        return None

    return _mean(
        [read(v) for v in VARS_2D] + [_mean([read(f"{v}_500"), read(f"{v}_850")]) for v in VARS_3D]
    )


def _scratch(prefix, row, lead, score):
    p = SCRATCH / f"{prefix}_{row['esvs']}_L{lead}.csv"
    if not p.exists():
        return None
    for r in _rows(p):
        if r["score"] == score and int(float(r["lead_hours"])) == lead:
            return float(r["value"])
    return None


def es(row, lead):
    return _scratch("esvs", row, lead, "energy_score_mvar")


def vs(row, lead):
    return _scratch("esvs", row, lead, "variogram_score_p05")


def sigk(row, lead):
    return _scratch("sigk", row, lead, "signature_kernel_score")


METRICS = [
    ("CRPSS", crpss, "max"),
    ("SSR", ssr, "one"),
    ("SSIM", ssim, "max"),
    ("LSD", lsd, "min"),
    ("FSS", fss95, "max"),
    ("W1", w1, "min"),
    ("ES", es, "min"),
    ("VS", vs, "min"),
    ("SIGK", sigk, "min"),
]


def main():
    # compute every cell
    grid = {}  # (row_idx, metric, lead) -> value
    for i, row in enumerate(ROWS):
        for name, fn, _ in METRICS:
            for L in LEADS:
                grid[(i, name, L)] = fn(row, L)

    # bold marks: within-model column optimum
    bold = set()
    by_model = {}
    for i, row in enumerate(ROWS):
        by_model.setdefault(row["model"], []).append(i)
    for idxs in by_model.values():
        for name, _, rule in METRICS:
            for L in LEADS:
                cells = [(i, grid[(i, name, L)]) for i in idxs if grid[(i, name, L)] is not None]
                if not cells:
                    continue
                if rule == "max":
                    best = max(cells, key=lambda t: t[1])[0]
                elif rule == "min":
                    best = min(cells, key=lambda t: t[1])[0]
                else:  # closest to 1
                    best = min(cells, key=lambda t: abs(t[1] - 1))[0]
                bold.add((best, name, L))

    fmt = {
        "CRPSS": "{:.3f}",
        "SSR": "{:.2f}",
        "SSIM": "{:.2f}",
        "LSD": "{:.2f}",
        "FSS": "{:.2f}",
        "W1": "{:.3f}",
        "ES": "{:.3f}",
        "VS": "{:.4f}",
        "SIGK": "{:.1f}",
    }

    def cell(i, name, L):
        v = grid[(i, name, L)]
        if v is None:
            return "(NA)"
        s = fmt[name].format(v)
        return f"\\textbf{{{s}}}" if (i, name, L) in bold else s

    print("=" * 110)
    print("RECOMPUTED calibration table (current vintage). (NA) = job output not on disk yet.")
    print("=" * 110)
    hdr = f"{'row':22s}" + "".join(f"{n+str(L):>10s}" for n, _, _ in METRICS for L in LEADS)
    print(hdr)
    last_model = None
    for i, row in enumerate(ROWS):
        if row["model"] != last_model:
            print(f"--- {row['model']} ---")
            last_model = row["model"]
        label = f"{row['phase']} {row['cfg'][:12]}"
        line = f"{label[:22]:22s}"
        for name, _, _ in METRICS:
            for L in LEADS:
                v = grid[(i, name, L)]
                line += f"{(fmt[name].format(v) if v is not None else 'NA'):>10s}"
        print(line)

    print("\n" + "=" * 110)
    print("LaTeX tabular body (replace the block between \\midrule and \\bottomrule):")
    print("=" * 110)
    badge = {
        "aurora": r"\mbadge{auroraC}{aurora}",
        "graphcast_operational": r"\mbadge{graphcastC}{graphcast}",
        "sfno": r"\mbadge{sfnoC}{sfno}",
        "aifs": r"\mbadge{aifsC}{aifs}",
    }
    order = ["aurora", "graphcast_operational", "sfno", "aifs"]
    out = []
    for mi, model in enumerate(order):
        idxs = by_model[model]
        out.append(r"    \rowcolor{phaseBg}")
        out.append(r"    \multicolumn{20}{@{}l}{" + badge[model] + r"} \\")
        for pos, i in enumerate(idxs):
            if pos % 2 == 1:
                out.append(r"    \rowcolor{altRow}")
            cells = " & ".join(cell(i, n, L) for n, _, _ in METRICS for L in LEADS)
            out.append(f"    {ROWS[i]['phase']} & {ROWS[i]['cfg']} & {cells} \\\\")
        out.append(r"    \hline" if mi < len(order) - 1 else "")
    print("\n".join(x for x in out if x).rstrip())


if __name__ == "__main__":
    main()
