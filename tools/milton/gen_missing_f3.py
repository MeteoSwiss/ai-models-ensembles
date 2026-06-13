"""Generate the 4 missing F3 cascading-detection figures for the Milton
case-study Sec 4.6 draft. AIFS-ENS and aurora_encoder already on disk."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures_milton import f3_cascading_detection

BASELINES = ["fcn3", "sfno_modes10", "graphcast_all"]

for b in BASELINES:
    t0 = time.time()
    print(f"=== {b} ===", flush=True)
    try:
        f3_cascading_detection(b)
        print(f"  done in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

print("\nAll 4 missing cascading figures attempted.", flush=True)
