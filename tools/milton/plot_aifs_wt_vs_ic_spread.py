"""Milton case study: AIFS weight-only vs weight+IC ensemble dispersion.

Two panels vs lead time, averaged over the 14 Milton initialisations:
(a) track-cone width = mean great-circle distance of member cyclone centres
    to the ensemble-mean centre; (b) MSL intensity spread = cross-member
std of central pressure. Both quantify the under-dispersion of weight-only
AIFS that IC perturbation restores.

Source: milton_master_tracks.csv (tools/milton/aggregate_tracks.py).
Output: figures/milton_F9_aifs_wt_vs_ic_spread.{pdf,png}
"""

from __future__ import annotations
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/
from model_colors import AIFS_IC_CONTRAST, color_for, marker_for, style_for

BASE = Path("/iopsstor/scratch/cscs/sadamov/milton_case_study")
MASTER = BASE / "milton_master_tracks.csv"
OUT = Path(
    "/users/sadamov/pyprojects/ai-models-ensembles/figures/milton_F9_aifs_wt_vs_ic_spread.pdf"
)

# Match the headline figure: weight-only purple, weight+IC pink (the one figure
# where the AIFS variants are deliberately drawn apart, see model_colors.py).
COLOUR = {
    "aifs_perturbed": color_for("aifs_perturbed"),
    "aifs_perturbed_ic": AIFS_IC_CONTRAST,
    "ifs_ens": color_for("ifs_ens"),
    "aifsens": color_for("aifsens"),
}
PRETTY = {
    "aifs_perturbed": "weight-only",
    "aifs_perturbed_ic": "weight + IC",
    "ifs_ens": "IFS-ENS",
    "aifsens": "AIFS-ENS",
}
BINS = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]


def haversine(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1)
    dl = math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def which_bin(h):
    for lo, hi in BINS:
        if lo <= h < hi:
            return f"{lo}-{hi}"
    return None


pos = defaultdict(list)  # (baseline, init_tag, valid_time) -> [(lat, lon, psl)]
with open(MASTER) as f:
    for r in csv.DictReader(f):
        if r["baseline"] not in COLOUR:
            continue
        pos[(r["baseline"], r["init_tag"], r["time"])].append(
            (float(r["lat"]), float(r["lon"]), float(r["psl_hpa"]))
        )

cone = defaultdict(lambda: defaultdict(list))  # baseline -> bin -> [spread_km]
psl = defaultdict(lambda: defaultdict(list))
for (b, init, t), members in pos.items():
    if len(members) < 2:
        continue
    h = (
        datetime.fromisoformat(t) - datetime.strptime(init, "%Y%m%d_%H%M")
    ).total_seconds() / 3600.0
    bn = which_bin(h)
    if bn is None:
        continue
    lats = [m[0] for m in members]
    lons = [m[1] for m in members]
    clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
    cone[b][bn].append(
        sum(haversine(la, lo, clat, clon) for la, lo in zip(lats, lons)) / len(members)
    )
    psl[b][bn].append(st.pstdev([m[2] for m in members]))

x = [(lo + hi) / 2 for lo, hi in BINS]
labels = [f"{lo}-{hi}" for lo, hi in BINS]


def series(d, b):
    return [sum(d[b][bn]) / len(d[b][bn]) if d[b].get(bn) else float("nan") for bn in labels]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
for b in ("aifs_perturbed", "aifs_perturbed_ic", "ifs_ens", "aifsens"):
    ax1.plot(
        x,
        series(cone, b),
        color=COLOUR[b],
        label=PRETTY[b],
        ms=4,
        lw=1.6,
        linestyle=style_for(b),
        marker=marker_for(b),
    )
    ax2.plot(
        x,
        series(psl, b),
        color=COLOUR[b],
        label=PRETTY[b],
        ms=4,
        lw=1.6,
        linestyle=style_for(b),
        marker=marker_for(b),
    )
ax1.set_ylabel("Track-cone width (km)")
ax2.set_ylabel("MSL spread (hPa)")
for ax in (ax1, ax2):
    ax.set_xlabel("Lead time (h)")
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144])
    ax.grid(True, lw=0.4, alpha=0.5)
ax1.set_title("(a) Track-cone width", fontsize=9)
ax2.set_title("(b) Intensity spread", fontsize=9)
ax1.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.savefig(str(OUT).replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
print(f"-> {OUT}")
for b in ("aifs_perturbed", "aifs_perturbed_ic", "ifs_ens", "aifsens"):
    print(f"{PRETTY[b]:22s} cone {series(cone, b)}")
    print(f"{'':22s} psl  {series(psl, b)}")
