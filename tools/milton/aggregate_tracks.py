"""Aggregate all milton_tracks.csv into one master DataFrame and compute
NHC-style verification per (baseline, init, member): position error and
intensity error vs both IBTrACS (ultimate truth) and ERA5 (resolution-matched
truth). Save the master + verification CSVs.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE = Path("/iopsstor/scratch/cscs/sadamov/milton_case_study")
TRACKS = BASE / "tracks"
OUT_MASTER = BASE / "milton_master_tracks.csv"
OUT_VERIF = BASE / "milton_verification.csv"


def haversine(lat1, lon1, lat2, lon2):
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(a)) * 6371


def load_master() -> pd.DataFrame:
    csvs = list(TRACKS.rglob("milton_tracks.csv"))
    print(f"found {len(csvs)} per-init track CSVs")
    if not csvs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(c, parse_dates=["time"]) for c in csvs], ignore_index=True)
    # Filter IFS-ENS to the stratified 10-member subset used by the rest of the
    # paper's SwissClim eval pipeline (IFS_ENS_MEMBERS in evaluate_baselines.sh).
    # Without this filter Milton's tracker would report on all 50 IFS-ENS
    # members while every other baseline reports on 10, producing apples-to-
    # oranges detection rates.
    stratified_ifs_ens = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45}
    before = len(df[df.baseline == "ifs_ens"])
    df = df[(df.baseline != "ifs_ens") | (df.member.isin(stratified_ifs_ens))].copy()
    after = len(df[df.baseline == "ifs_ens"])
    print(f"  filtered IFS-ENS rows {before} -> {after} (stratified-10 subset)")
    print(
        f"master: {len(df)} rows, {df['baseline'].nunique()} baselines, "
        f"{df.groupby(['baseline','init_tag']).ngroups} (baseline, init) cells, "
        f"{df.groupby(['baseline','init_tag','member']).ngroups} (baseline, init, member) tracks"
    )
    return df


def load_ibtracs() -> pd.DataFrame:
    ibt = xr.open_dataset(BASE / "milton_2024_ibtracs.nc")
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(ibt["time"].values),
            "lat": ibt["lat"].values,
            "lon": ibt["lon"].values % 360,
            "psl_hpa_truth": ibt["usa_pres"].values,
            "v10_truth_kt": ibt["usa_wind"].values,
        }
    )
    return df


def era5_track() -> pd.DataFrame:
    """ERA5 truth track from era5_stitch.txt (set by era5_control.py)."""
    import re

    p = BASE / "era5_stitch.txt"
    if not p.exists():
        return pd.DataFrame()
    text = p.read_text()
    blocks = re.split(r"(?=^start\t)", text, flags=re.M)
    for blk in blocks:
        if not blk.startswith("start"):
            continue
        rows = []
        for line in blk.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            try:
                lon = float(parts[3])
                lat = float(parts[4])
                psl = float(parts[5])
                v10 = float(parts[6])
                yr, mo, dy, hr = int(parts[8]), int(parts[9]), int(parts[10]), int(parts[11])
                rows.append(
                    {
                        "time": pd.Timestamp(f"{yr:04d}-{mo:02d}-{dy:02d}T{hr:02d}:00"),
                        "lat": lat,
                        "lon": lon,
                        "psl_hpa_era5": psl / 100,
                        "v10_era5": v10,
                    }
                )
            except Exception:
                continue
        in_milton = [
            r
            for r in rows
            if 15 <= r["lat"] <= 32
            and 258 <= r["lon"] <= 285
            and pd.Timestamp("2024-10-04") <= r["time"] <= pd.Timestamp("2024-10-12")
        ]
        if len(in_milton) >= 5:
            return pd.DataFrame(in_milton)
    return pd.DataFrame()


def verify(master: pd.DataFrame, ibtracs: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    """Per-row verification: NHC track position error + MSL bias against both truths."""
    rows = []
    for _, t in master.iterrows():
        # Nearest IBTrACS time within +-3 h
        dt_ibt = (ibtracs["time"] - t["time"]).abs()
        idx_i = dt_ibt.idxmin()
        ib = ibtracs.iloc[idx_i] if dt_ibt.iloc[idx_i] <= pd.Timedelta(hours=3) else None
        # Nearest ERA5
        if not era5.empty:
            dt_er = (era5["time"] - t["time"]).abs()
            idx_e = dt_er.idxmin()
            er = era5.iloc[idx_e] if dt_er.iloc[idx_e] <= pd.Timedelta(hours=3) else None
        else:
            er = None
        row = {
            "baseline": t["baseline"],
            "init_tag": t["init_tag"],
            "member": t["member"],
            "time": t["time"],
            "lead_h": int(
                (
                    t["time"]
                    - pd.Timestamp(
                        f"{t['init_tag'][0:4]}-{t['init_tag'][4:6]}-{t['init_tag'][6:8]}T{t['init_tag'][9:11]}:{t['init_tag'][11:13]}"
                    )
                ).total_seconds()
                // 3600
            ),
            "fcst_lat": t["lat"],
            "fcst_lon": t["lon"],
            "fcst_psl_hpa": t["psl_hpa"],
            "fcst_v10_ms": t["v10_ms"],
        }
        if ib is not None:
            row.update(
                {
                    "ibt_lat": ib["lat"],
                    "ibt_lon": ib["lon"],
                    "ibt_psl_hpa": ib["psl_hpa_truth"],
                    "ibt_v10_kt": ib["v10_truth_kt"],
                    "pos_err_ibt_km": haversine(t["lat"], t["lon"], ib["lat"], ib["lon"]),
                    "psl_err_ibt": t["psl_hpa"] - ib["psl_hpa_truth"],
                }
            )
        if er is not None:
            row.update(
                {
                    "era5_lat": er["lat"],
                    "era5_lon": er["lon"],
                    "era5_psl_hpa": er["psl_hpa_era5"],
                    "pos_err_era5_km": haversine(t["lat"], t["lon"], er["lat"], er["lon"]),
                    "psl_err_era5": t["psl_hpa"] - er["psl_hpa_era5"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    master = load_master()
    if master.empty:
        print("no tracks yet -- run after batch completes")
        return
    master.to_csv(OUT_MASTER, index=False)
    print(f"wrote master -> {OUT_MASTER}")

    ibt = load_ibtracs()
    era5 = era5_track()
    print(f"ibtracs rows: {len(ibt)}, era5 rows: {len(era5)}")

    v = verify(master, ibt, era5)
    v.to_csv(OUT_VERIF, index=False)
    print(f"wrote verification -> {OUT_VERIF}")

    # Quick summary: per baseline, fraction of (init, member) cells that produced a Milton track
    print()
    print("=" * 60)
    print("DETECTION RATE per baseline (any Milton track produced):")
    n_inits = 14
    n_members = 10
    n_cells = n_inits * n_members
    for b in sorted(master["baseline"].unique()):
        sub = master[master["baseline"] == b]
        cells = sub.groupby(["init_tag", "member"]).ngroups
        print(
            f"  {b:18s} {cells:3d} / {n_cells} ({100*cells/n_cells:5.1f}%) of cells produced a Milton track"
        )
    print()
    print("INTENSITY: median min-MSL per (baseline, init) across members:")
    p = master.groupby(["baseline", "init_tag", "member"])["psl_hpa"].min().reset_index()
    p_med = p.groupby(["baseline", "init_tag"])["psl_hpa"].median().unstack()
    print(p_med.to_string(float_format=lambda x: f"{x:6.1f}"))


if __name__ == "__main__":
    main()
