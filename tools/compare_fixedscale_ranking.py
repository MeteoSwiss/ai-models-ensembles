"""Does a fixed climatological scaling change what the multivariate scores pick?

tools/submit_esvs_sigk_fixedscale.sh scores each production winner and its
closest ablation rivals twice: once with the per-init truth std (as published,
improper) and once with the fixed 1990-2019 climatological scale (proper).
This compares the two, per backbone family, and reports whether the winner
still beats its rival(s) on ES, VS and SIGK - i.e. whether the tie-break that
selected the production configuration survives a proper scaling.

Usage:
  python tools/compare_fixedscale_ranking.py [--dir <table_metrics_fixedscale>]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _env import SCRATCH

DIR = str(SCRATCH / "ai-models-ensembles/scratch/table_metrics_fixedscale")

# family -> (winner label, [rival labels]); labels match the run tags of the
# rescore job.
FAMILIES = {
    "aurora": ("win_aurora_enc", ["riv_aurora_enc_s044"]),
    "graphcast": ("win_graphcast_all", ["riv_graphcast_m2g", "riv_graphcast_g2m"]),
    "sfno": ("win_sfno_modes10", ["riv_sfno_enc_s054", "riv_sfno_enc_s035"]),
    "aifs": ("win_aifs_decoder", ["riv_aifs_all_s010"]),
}
SCORES = ["energy_score_mvar", "variogram_score_p05", "signature_kernel_score"]


def load(d: Path) -> dict:
    """(label, tag, lead, score) -> value."""
    out: dict = {}
    for f in sorted(d.glob("*.csv")):
        parts = f.stem.split("_")
        if parts[-2] not in ("fixed", "truthstd"):
            continue
        tag = parts[-2]
        with f.open() as fh:
            for row in csv.DictReader(fh):
                out[(row["model"], tag, int(row["lead_hours"]), row["score"])] = float(row["value"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--leads", nargs="+", type=int, default=[120, 240])
    args = ap.parse_args()

    vals = load(Path(args.dir))
    if not vals:
        print(f"no CSVs under {args.dir}")
        return

    flips = 0
    for lead in args.leads:
        print(f"\n=== lead {lead} h   (lower is better for all three scores)")
        for fam, (win, rivals) in FAMILIES.items():
            for score in SCORES:
                row = []
                verdict = []
                for tag in ("truthstd", "fixed"):
                    w = vals.get((win, tag, lead, score))
                    if w is None:
                        continue
                    beaten = []
                    for r in rivals:
                        rv = vals.get((r, tag, lead, score))
                        if rv is None:
                            continue
                        beaten.append(w < rv)
                        row.append(
                            f"{tag}: {win.split('_', 1)[1]}={w:.4g} vs {r.split('_', 1)[1]}={rv:.4g}"
                        )
                    if beaten:
                        verdict.append(all(beaten))
                if len(verdict) == 2 and verdict[0] != verdict[1]:
                    flips += 1
                    mark = "  <-- FLIPS"
                elif len(verdict) == 2:
                    mark = "  (winner ahead)" if verdict[0] else "  (winner behind, both ways)"
                else:
                    mark = "  (incomplete)"
                if row:
                    print(f"  {fam:10s} {score:24s}{mark}")
                    for r in row:
                        print(f"      {r}")
    print(f"\n{flips} winner/rival comparison(s) change direction under the fixed scale.")


if __name__ == "__main__":
    main()
