#!/usr/bin/env python3
"""Replace only the LSD (120 h, 240 h) cells of figures/calibration_basis_table.tex
with the values recomputed from the pooled energy_spectra re-eval
(SwissClim member-0 align fix, 2026-09-17), re-marking the within-model
argmin in bold. Every other column is left byte-identical.

Usage (host venv, from tools/):
    python patch_calibration_lsd.py            # print old -> new per row
    python patch_calibration_lsd.py --write    # also rewrite the .tex
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "figures" / "calibration_basis_table.tex"
LEADS = (120, 240)
# column index of LSD@120 among the numeric cells (CRPSS x2, SSR x2, SSIM x2, LSD x2, ...)
LSD_COL = 6


def load_generator():
    spec = importlib.util.spec_from_file_location("act", HERE / "assemble_calibration_table.py")
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(m)
    return m


def main() -> None:
    write = "--write" in sys.argv
    g = load_generator()
    rows = g.ROWS
    new = {i: {L: g.lsd(r, L) for L in LEADS} for i, r in enumerate(rows)}
    bold = set()
    for model in {r["model"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["model"] == model]
        for L in LEADS:
            best = min(idx, key=lambda i: new[i][L])
            bold.add((best, L))

    lines = TEX.read_text().splitlines(keepends=True)
    # table body rows look like:  Phase 2b & 0.025 / encoder \,\winner & 0.499 & ...
    pat = re.compile(r"^(\s*)(Phase \S+) & (.+?) & (.*) \\\\\s*$")
    hdr = re.compile(r"\\multicolumn\{20\}\{@\{\}l\}\{\\textbf\{(\w+)\}\}")
    out, hit, model = [], 0, None
    for line in lines:
        h = hdr.search(line)
        if h:
            model = h.group(1)  # aurora / graphcast / sfno / aifs
        m = pat.match(line)
        if not m:
            out.append(line)
            continue
        indent, phase, cfg, rest = m.groups()
        cells = [c.strip() for c in rest.split("&")]
        # cfg strings repeat across models (e.g. "0.03 / all"), so key on the
        # model header the row sits under plus the phase
        i = next(
            (k for k, r in enumerate(rows) if r["phase"] == phase and r["model"].startswith(model)),
            None,
        )
        if i is None:
            sys.exit(f"no ROWS entry for tex row: {model} | {phase} | {cfg}")
        for j, L in enumerate(LEADS):
            s = f"{new[i][L]:.2f}"
            cell = f"\\textbf{{{s}}}" if (i, L) in bold else s
            print(f"{rows[i]['model']:22s} {phase:9s} LSD{L}: {cells[LSD_COL + j]:>16s} -> {cell}")
            cells[LSD_COL + j] = cell
        out.append(f"{indent}{phase} & {cfg} & " + " & ".join(cells) + " \\\\\n")
        hit += 1
    if hit != len(rows):
        sys.exit(f"patched {hit} rows, expected {len(rows)}")
    if write:
        TEX.write_text("".join(out))
        print(f"wrote {TEX}")


if __name__ == "__main__":
    main()
