"""Produce paper-ready Milton case study tracker stats.

Reads milton_verification.csv (the master per-(baseline, init, member, time)
table joined against IBTrACS + ERA5). Outputs:

  milton_per_baseline_summary.csv  -- one row per (baseline) with:
      detection_rate          fraction of (init, member) pairs that produced
                              any track touching the Milton lifecycle window
      mean_pos_err_km         mean position error vs IBTrACS over all matched
                              (init, member, lead) rows
      pos_err_by_lead_km      mean position error binned in 24 h lead windows
      mean_psl_err_hpa        mean intensity bias vs IBTrACS
      mean_pos_err_vs_era5_km the same against ERA5 (intensity ceiling baseline)
      n_rows                  number of matched rows used

  milton_pos_err_by_lead.csv -- long-form table (baseline, lead_bin_h, mean_pos_err_km, n)
                                for Sec 4.5 figure / table.

These numbers go into the Sec 4.5 prose draft; nothing is committed back into
the paper tex from this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

CASE = Path("/iopsstor/scratch/cscs/sadamov/milton_case_study")
VERIF = CASE / "milton_verification.csv"
TRACKS = CASE / "milton_master_tracks.csv"

OUT_SUMMARY = CASE / "milton_per_baseline_summary.csv"
OUT_BY_LEAD = CASE / "milton_pos_err_by_lead.csv"

LEAD_BINS = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240]
LEAD_LABELS = [f"{a}-{b}" for a, b in zip(LEAD_BINS[:-1], LEAD_BINS[1:])]


def main() -> None:
    df = pd.read_csv(VERIF)
    print(f"verification rows: {len(df)}", flush=True)
    print(f"baselines: {sorted(df['baseline'].unique())}", flush=True)

    tracks = pd.read_csv(TRACKS)
    print(f"track rows: {len(tracks)}", flush=True)

    # Detection rate: of (baseline, init_tag, member) triples, how many produced
    # at least one track point within the Milton lifecycle window. The master
    # tracks CSV already filters to the window, so the presence of any row =
    # detection. n_total scales with the ensemble size of each baseline
    # (10 for the ML baselines, 50 for IFS-ENS).
    detect = tracks.groupby(["baseline", "init_tag", "member"]).size().rename("n_pts").reset_index()
    # IFS-ENS uses the stratified 10-member subset {0, 5, 10, ..., 45} to match
    # the rest of the paper's SwissClim eval pipeline; aggregate_tracks.py
    # filters the master CSV to that subset before this stats step runs.
    n_members_per_baseline = {
        "aifsens": 10,
        "aifs_perturbed": 10,
        "atlas": 10,
        "aurora_encoder": 10,
        "fcn3": 10,
        "graphcast_all": 10,
        "sfno_modes10": 10,
        "ifs_ens": 10,
        # Phase 5 (perturbed-IC) variants -- same 10 mbr each
        "aifs_perturbed_ic": 10,
        "aurora_encoder_ic": 10,
        "graphcast_all_ic": 10,
        "sfno_modes10_ic": 10,
    }
    detect_summary = detect.groupby("baseline").size().rename("n_detected").reset_index()
    detect_summary["n_total"] = detect_summary["baseline"].map(
        lambda b: 14 * n_members_per_baseline.get(b, 10)
    )
    detect_summary["detection_rate"] = detect_summary["n_detected"] / detect_summary["n_total"]

    # Verification-based stats (need an IBTrACS match)
    matched = df.dropna(subset=["pos_err_ibt_km"])
    psl = matched.dropna(subset=["psl_err_ibt"])
    era5_pos = df.dropna(subset=["pos_err_era5_km"])

    pos_err = (
        matched.groupby("baseline")["pos_err_ibt_km"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "mean_pos_err_km", "count": "n_pos"})
    )
    psl_err = (
        psl.groupby("baseline")["psl_err_ibt"]
        .agg(["mean", "median", "std"])
        .rename(columns={"mean": "mean_psl_err_hpa", "median": "median_psl_err_hpa"})
    )
    era5_err = (
        era5_pos.groupby("baseline")["pos_err_era5_km"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_pos_err_vs_era5_km", "count": "n_era5"})
    )

    summary = (
        detect_summary.set_index("baseline")
        .join(pos_err, how="outer")
        .join(psl_err[["mean_psl_err_hpa", "median_psl_err_hpa"]], how="outer")
        .join(era5_err, how="outer")
        .reset_index()
    )
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"-> {OUT_SUMMARY}", flush=True)

    matched = matched.copy()
    matched["lead_bin_h"] = pd.cut(
        matched["lead_h"], bins=LEAD_BINS, labels=LEAD_LABELS, right=True, include_lowest=True
    )
    by_lead = (
        matched.groupby(["baseline", "lead_bin_h"], observed=True)["pos_err_ibt_km"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "mean_pos_err_km", "count": "n"})
        .reset_index()
    )
    by_lead.to_csv(OUT_BY_LEAD, index=False)
    print(f"-> {OUT_BY_LEAD}", flush=True)

    print("\nPer-baseline summary:", flush=True)
    cols = [
        "baseline",
        "n_detected",
        "detection_rate",
        "mean_pos_err_km",
        "median",
        "std",
        "mean_psl_err_hpa",
        "mean_pos_err_vs_era5_km",
    ]
    with pd.option_context("display.float_format", lambda x: f"{x:.2f}"):
        print(summary[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
