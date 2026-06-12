"""One-shot remediation for the spatial-mean SSR Fortin-factor bug (2026-06-13).

tools/spatial_mean_ssr.py applied (M+1)/(M-1) to the ddof=1 sample variance;
the correct Fortin et al. 2014 factor for ddof=1 is (M+1)/M. Both factors are
constant multipliers on spread^2 for fixed M, so every affected CSV can be
corrected EXACTLY post hoc: spread and ssr scale by sqrt((M-1)/M) (~0.9487 at
M=10), error is unchanged. This avoids re-reading the forecast zarrs entirely.

Exactness assumption: M was identical for every (init, var, lead, level) cell
aggregated into a row. True by construction here (all runs are 10-member
ensembles); the script aborts on any row with n_members != 10.

For each CSV: archives the original, then rewrites spread/ssr in place.
Archive location: <stem>_pre_fortinfix_2026-06-13.csv alongside, EXCEPT in
diagnostics/sfno_phase6_ssr where tools/plot_sfno_phase6_ssr.py globs *.csv,
so archives go to a pre_fortinfix_2026-06-13/ subdirectory instead.
Idempotent: files whose archive already exists are skipped. Stdlib-only.
"""

import csv
import glob
import math
import os
import shutil
import sys

BASE = "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles"
SUFFIX = "_pre_fortinfix_2026-06-13"
FACTOR = math.sqrt(9.0 / 10.0)


def targets() -> list[str]:
    csvs = sorted(glob.glob(BASE + "/baselines/*/spatial_mean_ssr/spatial_ssr.csv"))
    csvs += sorted(glob.glob(BASE + "/ablation/*/*/eval/*/spatial_mean_ssr/spatial_ssr.csv"))
    csvs += sorted(glob.glob(BASE + "/diagnostics/sfno_phase6_ssr/*.csv"))
    return [c for c in csvs if SUFFIX not in c]


def archive_path(path: str) -> str:
    d, name = os.path.split(path)
    if d.endswith("/diagnostics/sfno_phase6_ssr"):
        return os.path.join(d, "pre_fortinfix_2026-06-13", name)
    return path[: -len(".csv")] + SUFFIX + ".csv"


def main() -> int:
    done = skipped = 0
    for path in targets():
        archive = archive_path(path)
        if os.path.exists(archive):
            print(f"SKIP (already archived): {path}")
            skipped += 1
            continue
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
            fieldnames = list(rows[0].keys())
        bad = [r for r in rows if int(r["n_members"]) != 10]
        if bad:
            print(f"ABORT {path}: rows with n_members != 10: {bad}")
            return 1
        os.makedirs(os.path.dirname(archive), exist_ok=True)
        shutil.copy2(path, archive)
        for r in rows:
            for col in ("spread", "ssr"):
                # ssr is empty when error^2 was 0 (pandas writes NaN as '')
                if r[col] not in ("", "nan"):
                    r[col] = repr(float(r[col]) * FACTOR)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"FIXED {path} ({len(rows)} rows, x{FACTOR:.6f})")
        done += 1
    print(f"\n{done} rescaled, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
