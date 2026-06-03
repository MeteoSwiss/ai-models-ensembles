"""Compare the 2022-2024 (Gaussian) vs 1990-2019 (Gaussian + empirical) CRPS
denominators and re-emit the three paper tables / one figure with the new
numbers.

Inputs (all JSON, scalar per (var, level) keyed as e.g. ``geopotential_500``
or ``2m_temperature``):
  /iopsstor/scratch/cscs/sadamov/sigma_clim_ablation.json        (old)
  /iopsstor/scratch/cscs/sadamov/sigma_clim_1990_2019.json       (new gauss)
  /iopsstor/scratch/cscs/sadamov/empirical_crps_clim_1990_2019.json (new emp)

Outputs (under /iopsstor/scratch/cscs/sadamov/recomputed_tables/, never
overwriting the canonical .tex in figures/):
  sigma_clim_comparison.csv
  crpss_old_vs_new_per_table.csv
  headline_7way_table_NEW_gauss1990.tex
  headline_7way_table_NEW_empirical1990.tex
  headline_crpss_vs_lead_7way_NEW_gauss1990.pdf
  headline_crpss_vs_lead_7way_NEW_empirical1990.pdf
  calibration_basis_table_NEW_*.tex      (one per recipe)
  uplift_NEW_*.txt                       (uplift table, two recipes)

And one diff-report markdown at
  /iopsstor/scratch/cscs/sadamov/climatology_recompute_diff.md
listing per-cell (old, new_gauss, new_empirical, delta) for the headline
table and the per-variable sigma table.

This script does NOT touch the paper repo.  The user reviews the diff
report and decides which numbers to integrate.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SCRATCH = Path("/iopsstor/scratch/cscs/sadamov")
OLD_SIGMA = SCRATCH / "sigma_clim_ablation.json"
NEW_SIGMA = SCRATCH / "sigma_clim_1990_2019.json"
NEW_EMP_CRPS = SCRATCH / "empirical_crps_clim_1990_2019.json"
OUTDIR = SCRATCH / "recomputed_tables"
DIFF_REPORT = SCRATCH / "climatology_recompute_diff.md"

CSV_PROB = (
    "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/"
    "baselines/intercomparison/probabilistic/temporal_metrics_combined.csv"
)

VARS_2D = ["2m_temperature", "mean_sea_level_pressure"]
VARS_3D = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
LEADS = [72, 120, 240, 360]
MODELS = ["aifsens", "atlas", "fcn3", "ifs_ens", "graphcast_all", "aurora_encoder", "sfno_modes10"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def keys_per_field() -> list[str]:
    return VARS_2D + [f"{v}_{lvl}" for v in VARS_3D for lvl in (500, 850)]


def crps_clim_from_sigma(sigma: float) -> float:
    return sigma / math.sqrt(math.pi)


def load_baseline_crps() -> dict:
    """(model, var, lead, lvl) -> CRPS from the headline 7-way CSV."""
    out: dict = {}
    with open(CSV_PROB) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "CRPS" or row["model"] not in MODELS:
                continue
            try:
                lead = int(row["lead_time"])
                lvl = float(row["level"]) if row["level"] else None
                val = float(row["value"])
            except ValueError:
                continue
            out[(row["model"], row["variable"], lead, lvl)] = val
    return out


def crpss_table(crps_data: dict, denom: dict) -> dict:
    """rows[model][lead] = variable-mean CRPSS using denom = CRPS_clim per field."""
    rows: dict = {m: {} for m in MODELS}
    for m in MODELS:
        for lead in LEADS + sorted({k[2] for k in crps_data} - set(LEADS)):
            per_var = []
            for v in VARS_2D:
                candidates = [
                    val for k, val in crps_data.items() if k[0] == m and k[1] == v and k[2] == lead
                ]
                if not candidates:
                    continue
                d = denom.get(v)
                if not d:
                    continue
                per_var.append(1 - candidates[0] / d)
            for v in VARS_3D:
                skills = []
                for lvl in (500.0, 850.0):
                    val = crps_data.get((m, v, lead, lvl))
                    if val is None:
                        continue
                    d = denom.get(f"{v}_{int(lvl)}")
                    if not d:
                        continue
                    skills.append(1 - val / d)
                if skills:
                    per_var.append(sum(skills) / len(skills))
            if per_var:
                rows[m][lead] = sum(per_var) / len(per_var)
    return rows


def build_denom(sigma: dict) -> dict:
    return {k: crps_clim_from_sigma(v) for k, v in sigma.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    old_sigma_raw = json.load(open(OLD_SIGMA))
    # Old schema is {key: scalar}; new schema is {key: {unconditional, doy_conditional}}.
    old_sigma = {
        k: (v if isinstance(v, (int, float)) else v.get("unconditional"))
        for k, v in old_sigma_raw.items()
    }
    if not NEW_SIGMA.exists():
        raise SystemExit(f"Missing {NEW_SIGMA} -- run compute_climatology_1990_2019.py first")
    new_sigma_raw = json.load(open(NEW_SIGMA))
    new_sigma_uncond = {
        k: (v["unconditional"] if isinstance(v, dict) else v) for k, v in new_sigma_raw.items()
    }
    new_sigma_doy = {
        k: (v["doy_conditional"] if isinstance(v, dict) else None) for k, v in new_sigma_raw.items()
    }
    new_emp_crps = json.load(open(NEW_EMP_CRPS)) if NEW_EMP_CRPS.exists() else None
    # Re-bind for backward-compatible refs below.
    new_sigma = new_sigma_uncond

    # 1) sigma_clim comparison
    sigma_csv = OUTDIR / "sigma_clim_comparison.csv"
    with open(sigma_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "key",
                "sigma_old_2022_2024",
                "sigma_new_1990_2019",
                "ratio_new_over_old",
                "gauss_crps_old",
                "gauss_crps_new",
                "empirical_crps_new",
                "empirical_over_gauss_new",
            ]
        )
        for k in keys_per_field():
            so = old_sigma.get(k)
            sn = new_sigma.get(k)
            eg = crps_clim_from_sigma(so) if so else None
            ng = crps_clim_from_sigma(sn) if sn else None
            ne = new_emp_crps.get(k) if new_emp_crps else None
            ratio = sn / so if so and sn else None
            eo_ratio = ne / ng if ne and ng else None
            w.writerow([k, so, sn, ratio, eg, ng, ne, eo_ratio])
    print(f"-> {sigma_csv}")

    # 2) Build the four CRPSS denominators (apples-apples requires DOY-cond.)
    denoms = {
        "gauss_old_2022_2024_uncond": build_denom(old_sigma),
        "gauss_new_1990_2019_uncond": build_denom(new_sigma_uncond),
        "gauss_new_1990_2019_doy": build_denom(new_sigma_doy),
    }
    if new_emp_crps:
        denoms["empirical_new_1990_2019_doy"] = new_emp_crps

    # 3) Headline 7-way CRPSS per recipe
    crps_data = load_baseline_crps()
    crpss_per_recipe = {name: crpss_table(crps_data, d) for name, d in denoms.items()}

    # Cell-by-cell diff CSV
    diff_csv = OUTDIR / "crpss_old_vs_new_per_table.csv"
    with open(diff_csv, "w", newline="") as f:
        w = csv.writer(f)
        recipe_names = list(crpss_per_recipe.keys())
        ref_recipe = "gauss_old_2022_2024_uncond"
        w.writerow(
            ["model", "lead_h"]
            + recipe_names
            + [f"delta_{r}_vs_old" for r in recipe_names if r != ref_recipe]
        )
        for m in MODELS:
            for lead in LEADS:
                vals = [crpss_per_recipe[r][m].get(lead) for r in recipe_names]
                ref = crpss_per_recipe[ref_recipe][m].get(lead)
                deltas = [
                    (v - ref) if (v is not None and ref is not None) else None
                    for r, v in zip(recipe_names, vals)
                    if r != ref_recipe
                ]
                w.writerow([m, lead] + vals + deltas)
    print(f"-> {diff_csv}")

    # 4) Re-emit headline 7-way LaTeX tables (one per new recipe)
    ref_recipe = "gauss_old_2022_2024_uncond"
    for recipe_name, rows in crpss_per_recipe.items():
        if recipe_name == ref_recipe:
            continue
        out_tex = OUTDIR / f"headline_7way_table_NEW_{recipe_name}.tex"
        write_headline_tex(out_tex, rows, recipe_name)
        print(f"-> {out_tex}")

    # 5) Re-emit headline CRPSS-vs-lead figure (one per new recipe)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401 -- imported for plot_headline_crpss_vs_lead

        for recipe_name, rows in crpss_per_recipe.items():
            if recipe_name == ref_recipe:
                continue
            out_pdf = OUTDIR / f"headline_crpss_vs_lead_7way_NEW_{recipe_name}.pdf"
            plot_headline_crpss_vs_lead(out_pdf, rows, recipe_name)
            print(f"-> {out_pdf}")
    except ImportError:
        print("(matplotlib missing; skipped figure regen)")

    # 6) Diff report markdown
    write_diff_report(
        DIFF_REPORT,
        old_sigma,
        new_sigma_uncond,
        new_sigma_doy,
        new_emp_crps,
        crpss_per_recipe,
        ref_recipe,
    )
    print(f"-> {DIFF_REPORT}")


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
PRETTY = {
    "aifsens": "AIFS-ENS",
    "atlas": "Atlas",
    "fcn3": "FCN3",
    "ifs_ens": "IFS-ENS",
    "graphcast_all": "graphcast\\_all",
    "aurora_encoder": "aurora\\_encoder",
    "sfno_modes10": "sfno\\_modes10",
}
ROLE = {
    "aurora_encoder": "post-hoc",
    "graphcast_all": "post-hoc",
    "sfno_modes10": "post-hoc",
    "aifsens": "trained-prob",
    "atlas": "trained-prob",
    "fcn3": "trained-prob",
    "ifs_ens": "classical",
}


def write_headline_tex(out_path: Path, rows: dict, recipe_name: str):
    maxes = {
        lead: max(
            (rows[m].get(lead) for m in MODELS if rows[m].get(lead) is not None), default=None
        )
        for lead in LEADS
    }
    order = sorted(MODELS, key=lambda m: -(rows[m].get(240) or -99))

    def fmt(v, lead):
        if v is None:
            return "--"
        s = f"{v:.3f}"
        if maxes[lead] is not None and abs(v - maxes[lead]) < 1e-9:
            s = f"\\textbf{{{s}}}"
        return s

    L = []
    L.append(f"% Headline 7-way CRPSS table -- recipe {recipe_name}")
    L.append("% Auto-generated by tools/diff_climatology_recompute.py")
    L.append("\\begin{table}[t]")
    L.append("  \\centering \\small \\setlength{\\tabcolsep}{6pt}")
    L.append(
        f"  \\caption{{Headline 7-way CRPSS @ (72,120,240,360)h. "
        f"Denominator: \\textbf{{{recipe_name.replace('_', ' ')}}}.}}"
    )
    L.append(f"  \\label{{tab:headline7way_{recipe_name}}}")
    L.append("  \\begin{tabular}{@{}l l rrrr@{}}")
    L.append("    \\toprule")
    L.append(
        "    \\textbf{Role} & \\textbf{Model} & "
        "\\textbf{72\\,h} & \\textbf{120\\,h} & \\textbf{240\\,h} & \\textbf{360\\,h} \\\\"
    )
    L.append("    \\midrule")
    group_order = ["trained-prob", "post-hoc", "classical"]
    last = None
    for g in group_order:
        for m in order:
            if ROLE[m] != g:
                continue
            label = {
                "post-hoc": "Post-hoc weight perturbation (this work)",
                "trained-prob": "Trained probabilistic",
                "classical": "Classical reference",
            }[g]
            if last != g:
                L.append("    \\rowcolor{phaseBg}")
                L.append(f"    \\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{label}}}}} \\\\")
                last = g
            cells = [fmt(rows[m].get(lead), lead) for lead in LEADS]
            L.append(f"    & {PRETTY[m]} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
        if g != group_order[-1]:
            L.append("    \\midrule")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("\\end{table}")
    out_path.write_text("\n".join(L))


def plot_headline_crpss_vs_lead(out_pdf: Path, rows: dict, recipe_name: str):
    import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    all_leads = sorted({lead for m in MODELS for lead in rows[m]})
    for m in MODELS:
        xs, ys = [], []
        for lead in all_leads:
            v = rows[m].get(lead)
            if v is None or lead == 0:
                continue
            xs.append(lead)
            ys.append(v)
        if xs:
            ax.plot(xs, ys, label=m, color=COLOUR[m], linestyle=STYLE[m], linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xticks([0, 24, 72, 120, 168, 240, 312, 360])
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel(f"CRPSS ({recipe_name})")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"Headline 7-way: denom = {recipe_name}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_diff_report(
    out_md: Path,
    old_sigma: dict,
    new_sigma_uncond: dict,
    new_sigma_doy: dict,
    new_emp: dict | None,
    crpss_per_recipe: dict,
    ref_recipe: str,
):
    L = []
    L.append("# Climatology recompute diff report")
    L.append("")
    L.append("Baseline: paper currently uses Gaussian CRPS_clim = sigma_clim / sqrt(pi)")
    L.append("with sigma_clim from WB2 ERA5 2022-2024 (file sigma_clim_ablation.json).")
    L.append("")
    L.append("Recipes evaluated here:")
    L.append("  - `gauss_old_2022_2024_uncond` (current paper baseline)")
    L.append("  - `gauss_new_1990_2019_uncond` (reviewer-fix temporal window only;")
    L.append("    annual cycle + interannual variability both in sigma)")
    L.append("  - `gauss_new_1990_2019_doy` (DOY-conditional sigma; annual cycle removed,")
    L.append("    apples-to-apples with the empirical recipe below)")
    L.append("  - `empirical_new_1990_2019_doy` (non-parametric LOO fair-CRPS of a")
    L.append("    per-pixel, per-(DOY, hour) 30-member climatological ensemble)")
    L.append("")
    L.append("## 1. sigma_clim per (variable, level)")
    L.append("")
    L.append(
        "| field | old uncond (2022-2024) | new uncond (1990-2019) | new DOY-cond (1990-2019) | uncond ratio new/old |"
    )
    L.append("|---|---:|---:|---:|---:|")
    for k in keys_per_field():
        so = old_sigma.get(k)
        snu = new_sigma_uncond.get(k)
        snd = new_sigma_doy.get(k)
        r = snu / so if so and snu else float("nan")
        L.append(
            f"| `{k}` | {so:.4g} | {snu:.4g} | "
            f"{(snd if snd is not None else float('nan')):.4g} | {r:.3f} |"
        )
    L.append("")

    if new_emp:
        L.append("## 2. Gaussian vs empirical denominator (DOY-conditional, 1990-2019)")
        L.append("")
        L.append("Note: Gaussian uses sigma_clim_doy / sqrt(pi); empirical is the")
        L.append("LOO fair-CRPS of the 30-member per-(pixel, DOY, hour) ensemble.")
        L.append("Both share the same conditioning, so the ratio isolates non-Gaussianity.")
        L.append("")
        L.append(
            "| field | gauss_doy = sigma_doy/sqrt(pi) | empirical (LOO) | empirical / gauss_doy |"
        )
        L.append("|---|---:|---:|---:|")
        for k in keys_per_field():
            snd = new_sigma_doy.get(k)
            ng = snd / math.sqrt(math.pi) if snd else None
            ne = new_emp.get(k)
            r = ne / ng if ne and ng else float("nan")
            L.append(
                f"| `{k}` | {(ng if ng is not None else float('nan')):.4g} | "
                f"{(ne if ne is not None else float('nan')):.4g} | {r:.3f} |"
            )
        L.append("")

    L.append("## 3. Headline 7-way CRPSS per (model, lead) -- old vs new")
    L.append("")
    recipes = list(crpss_per_recipe.keys())
    header = "| model | lead h | " + " | ".join(recipes) + " |"
    L.append(header)
    L.append("|---|---:|" + "---:|" * len(recipes))
    for m in MODELS:
        for lead in LEADS:
            cells = [f"{crpss_per_recipe[r][m].get(lead, float('nan')):.3f}" for r in recipes]
            L.append(f"| {m} | {lead} | " + " | ".join(cells) + " |")
    L.append("")

    # Highlight: largest deltas
    L.append("## 4. Largest CRPSS deltas (new vs old)")
    L.append("")
    deltas = []
    for m in MODELS:
        for lead in LEADS:
            ref = crpss_per_recipe[ref_recipe][m].get(lead)
            for rname in recipes:
                if rname == ref_recipe:
                    continue
                v = crpss_per_recipe[rname][m].get(lead)
                if v is None or ref is None:
                    continue
                deltas.append((abs(v - ref), m, lead, rname, ref, v, v - ref))
    deltas.sort(reverse=True)
    L.append("| rank | model | lead h | recipe | old | new | delta |")
    L.append("|---:|---|---:|---|---:|---:|---:|")
    for i, (_, m, lead, rname, ref, v, d) in enumerate(deltas[:25], 1):
        L.append(f"| {i} | {m} | {lead} | {rname} | {ref:.3f} | {v:.3f} | {d:+.3f} |")
    L.append("")
    L.append("## 5. Files written")
    L.append("")
    L.append("- /iopsstor/scratch/cscs/sadamov/recomputed_tables/sigma_clim_comparison.csv")
    L.append("- /iopsstor/scratch/cscs/sadamov/recomputed_tables/crpss_old_vs_new_per_table.csv")
    L.append("- /iopsstor/scratch/cscs/sadamov/recomputed_tables/headline_7way_table_NEW_*.tex")
    L.append(
        "- /iopsstor/scratch/cscs/sadamov/recomputed_tables/headline_crpss_vs_lead_7way_NEW_*.pdf"
    )
    out_md.write_text("\n".join(L))


if __name__ == "__main__":
    main()
