"""Aggregate per-baseline signature_kernel_score.py CSVs into the appendix table.

Reads <in-dir>/<baseline>.csv (one row each), sorts by the signature kernel
score (lower is better), and prints both a console table and LaTeX rows ready
to drop next to Tab. esvs (bold = per-column optimum). Pretty labels match the
ES/VS table rows exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PRETTY = {
    "aifsens": "AIFS-ENS",
    "atlas": "Atlas-ERA5",
    "aifs_perturbed": "aifs\\_perturbed",
    "fcn3": "FourCastNet 3",
    "graphcast_all": "graphcast\\_all",
    "aurora_encoder": "aurora\\_encoder",
    "sfno_modes10": "sfno\\_modes10",
}
# fixed display order = the Tab. esvs row order (ES ranking)
ORDER = list(PRETTY)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    rows = []
    for b in ORDER:
        f = Path(args.in_dir) / f"{b}.csv"
        if f.exists():
            rows.append(pd.read_csv(f).iloc[0])
        else:
            print(f"  MISSING {f}")
    if not rows:
        print("No baseline CSVs found.")
        return 1
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)

    best = df["value"].min()
    print(
        f"\nSignature Kernel Score (lower is better), {int(df['n_inits'].max())} inits, "
        f"{int(df['n_pixels'].max())} px/init, {int(df['n_members'].max())} members\n"
    )
    print(f"{'baseline':16s} {'SIGK':>12s}  {'n_inits':>7s}")
    for _, r in df.iterrows():
        mark = "  <- best" if abs(r["value"] - best) < 1e-12 else ""
        print(f"{r['model']:16s} {r['value']:>12.4f}  {int(r['n_inits']):>7d}{mark}")

    print("\nLaTeX rows (bold = optimum):")
    for _, r in df.iterrows():
        cell = (
            f"\\mathbf{{{r['value']:.3f}}}"
            if abs(r["value"] - best) < 1e-12
            else f"{r['value']:.3f}"
        )
        print(f"{PRETTY.get(r['model'], r['model']):16s} & ${cell}$ \\\\")
    print(f"\nWrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
