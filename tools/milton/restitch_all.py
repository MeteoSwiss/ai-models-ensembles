"""Re-run StitchNodes on every existing detect.txt with the bumped maxgap.

The DetectNodes phase (which is the expensive one) is unchanged; only the
StitchNodes / extract / aggregate phases are repeated. Iterates over every
baseline x init x member dir under tracks/ that already has a detect.txt.

Per-init milton_tracks.csv files are rewritten in place. Run aggregate_tracks.py
afterwards to refresh the master CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from track_one_init import (  # noqa: E402
    extract_milton_track,
)

TRACKS_ROOT = Path("/iopsstor/scratch/cscs/sadamov/milton_case_study/tracks")


def main() -> None:
    inits = sorted({p.name for p in TRACKS_ROOT.glob("*/*") if p.is_dir()})
    print(f"baselines: {sorted(p.name for p in TRACKS_ROOT.iterdir() if p.is_dir())}", flush=True)
    print(f"inits: {len(inits)} unique tags", flush=True)

    total_re = 0
    for baseline_dir in sorted(TRACKS_ROOT.iterdir()):
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        for init_dir in sorted(baseline_dir.iterdir()):
            if not init_dir.is_dir():
                continue
            init_tag = init_dir.name
            detect_files = sorted(init_dir.glob("m*_detect.txt"))
            if not detect_files:
                continue
            tracks = []
            for det in detect_files:
                m = det.stem.replace("_detect", "")  # e.g. m00
                sti = init_dir / f"{m}_stitch.txt"
                try:
                    # Re-run StitchNodes only (DetectNodes already produced det).
                    # We use a tiny in-place hack: track_one_init.run_tracker
                    # re-runs DetectNodes first, which would clobber the detect.
                    # Instead inline-call StitchNodes with the patched maxgap
                    # from track_one_init.py.
                    import subprocess

                    cmd_s = [
                        "StitchNodes",
                        "--in",
                        str(det),
                        "--out",
                        str(sti),
                        "--in_fmt",
                        "lon,lat,psl,v10,vort850",
                        "--range",
                        "5.0",
                        "--mintime",
                        "36h",
                        "--maxgap",
                        "18h",
                        "--threshold",
                        "v10,>=,17.0,3",
                    ]
                    r = subprocess.run(cmd_s, capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  FAIL {baseline}/{init_tag}/{m}: {r.stderr[-400:]}", flush=True)
                        continue
                except Exception as e:
                    print(f"  EXC {baseline}/{init_tag}/{m}: {e}", flush=True)
                    continue
                t = extract_milton_track(sti)
                if not t.empty:
                    t["baseline"] = baseline
                    t["init_tag"] = init_tag
                    t["member"] = int(m.replace("m", ""))
                    tracks.append(t)
            if tracks:
                df = pd.concat(tracks, ignore_index=True)
                df.to_csv(init_dir / "milton_tracks.csv", index=False)
            total_re += len(detect_files)
        print(
            f"  re-stitched {baseline}: {sum(1 for d in baseline_dir.glob('*/m*_detect.txt'))} member-files",
            flush=True,
        )

    print(f"\nDone. Total re-stitched member-files: {total_re}", flush=True)


if __name__ == "__main__":
    main()
