"""Track one (baseline, init) -- runs tracker on all 10 members of one forecast.zarr.

Usage:
    python track_one_init.py <baseline> <init_tag>
e.g. python track_one_init.py aifsens 20241004_0000
"""

from __future__ import annotations
import subprocess
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE = Path("/iopsstor/scratch/cscs/sadamov/milton_case_study")
TRACKS_ROOT = BASE / "tracks"
TRACKS_ROOT.mkdir(parents=True, exist_ok=True)

BASELINES = [
    "aurora_encoder",
    "graphcast_all",
    "sfno_modes10",
    "aifsens",
    "fcn3",
    "atlas",
    "ifs_ens",
]

# Milton bbox (Gulf of Mexico + Caribbean + SE US + western Atlantic)
LON_MIN, LON_MAX = 250, 305
LAT_MIN, LAT_MAX = 8, 42


def baseline_zarr_path(baseline: str, init_tag: str) -> Path:
    return Path(
        f"/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/{baseline}/{init_tag}/forecast.zarr"
    )


def make_member_nc(ds_mem: xr.Dataset, nc_out: Path, init_t: pd.Timestamp) -> None:
    sub = ds_mem.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
    lead_h = (sub["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    valid = init_t + pd.to_timedelta(lead_h, unit="h")
    out = xr.Dataset(
        coords={
            "time": ("time", valid),
            "latitude": sub["latitude"],
            "longitude": sub["longitude"],
        }
    )
    out["psl"] = (
        ("time", "latitude", "longitude"),
        sub["mean_sea_level_pressure"].values.astype("float32"),
    )
    z = sub["geopotential"]
    z500 = z.sel(level=500).values.astype("float32")
    z850 = z.sel(level=850).values.astype("float32")
    out["z_thick_500m850"] = (("time", "latitude", "longitude"), z500 - z850)
    u10 = sub["10m_u_component_of_wind"].values.astype("float32")
    v10 = sub["10m_v_component_of_wind"].values.astype("float32")
    out["v10"] = (("time", "latitude", "longitude"), np.sqrt(u10**2 + v10**2))
    u850 = sub["u_component_of_wind"].sel(level=850).values.astype("float64")
    v850 = sub["v_component_of_wind"].sel(level=850).values.astype("float64")
    lat_vals = sub["latitude"].values
    lon_vals = sub["longitude"].values
    R = 6.371e6
    dlat = np.deg2rad(np.gradient(lat_vals))
    dlon = np.deg2rad(np.gradient(lon_vals))
    cos_lat = np.cos(np.deg2rad(lat_vals))[:, None]
    dv_dx = np.gradient(v850, axis=2) / (R * cos_lat * dlon[None, None, :])
    du_dy = np.gradient(u850, axis=1) / (R * dlat[None, :, None])
    out["vort850"] = (("time", "latitude", "longitude"), (dv_dx - du_dy).astype("float32"))
    out["time"].encoding = {
        "units": f"hours since {init_t.strftime('%Y-%m-%d %H:%M:%S')}",
        "calendar": "standard",
    }
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(nc_out, encoding=encoding)


def run_tracker(nc_in: Path, detect_out: Path, stitch_out: Path) -> None:
    cmd_d = [
        "DetectNodes",
        "--in_data",
        str(nc_in),
        "--out",
        str(detect_out),
        "--searchbymin",
        "psl",
        "--closedcontourcmd",
        "psl,200.0,5.5,0",
        "--closedcontourcmd",
        "z_thick_500m850,-58.8,6.5,1.0",
        "--outputcmd",
        "psl,min,0;v10,max,2;vort850,max,2",
        "--mergedist",
        "6.0",
        "--latname",
        "latitude",
        "--lonname",
        "longitude",
        "--regional",
    ]
    r = subprocess.run(cmd_d, capture_output=True, text=True)
    if r.returncode != 0 or "EXCEPTION" in r.stdout:
        raise RuntimeError(f"DetectNodes failed: {r.stdout[-2000:]}")
    cmd_s = [
        "StitchNodes",
        "--in",
        str(detect_out),
        "--out",
        str(stitch_out),
        "--in_fmt",
        "lon,lat,psl,v10,vort850",
        "--range",
        "5.0",
        "--mintime",
        "36h",
        # --maxgap was 6h; bumped to 18h to bridge the WB2 IFS-ENS NaN gaps
        # (max observed contiguous gap is 18h; almost all gaps are 6 or 12h).
        # Applied uniformly across baselines: a no-op for ML baselines whose
        # detect.txt files have no missing leads. See [[ifs-ens-10m-wind-nan-bug]].
        "--maxgap",
        "18h",
        "--threshold",
        "v10,>=,17.0,3",
    ]
    r = subprocess.run(cmd_s, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"StitchNodes failed: {r.stderr[-2000:]}")


def extract_milton_track(stitch_path: Path) -> pd.DataFrame:
    if not stitch_path.exists():
        return pd.DataFrame()
    text = stitch_path.read_text()
    blocks = re.split(r"(?=^start\t)", text, flags=re.M)
    for blk in blocks:
        if not blk.startswith("start"):
            continue
        pts = []
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
                pts.append(
                    {
                        "time": pd.Timestamp(f"{yr:04d}-{mo:02d}-{dy:02d}T{hr:02d}:00"),
                        "lat": lat,
                        "lon": lon,
                        "psl_hpa": psl / 100,
                        "v10_ms": v10,
                    }
                )
            except Exception:
                continue
        in_milton = [
            p
            for p in pts
            if 15 <= p["lat"] <= 32
            and 258 <= p["lon"] <= 285
            and pd.Timestamp("2024-10-04") <= p["time"] <= pd.Timestamp("2024-10-12")
        ]
        if len(in_milton) >= 4:
            return pd.DataFrame(in_milton)
    return pd.DataFrame()


def main(baseline: str, init_tag: str):
    init_t = pd.Timestamp(
        f"{init_tag[0:4]}-{init_tag[4:6]}-{init_tag[6:8]}T{init_tag[9:11]}:{init_tag[11:13]}"
    )
    if baseline == "ifs_ens":
        # Consolidated multi-init zarr at /capstor/store/cscs/mch/s83/IFS/ifs_ens.zarr.
        # Reshard is in-progress so consolidated metadata is stale; open with
        # consolidated=False. Select the init slice + drop the init_time dim
        # so the rest of the pipeline sees a regular per-init dataset.
        print(f"{baseline} {init_tag} -> loading consolidated IFS-ENS zarr...")
        ds_all = xr.open_zarr("/capstor/store/cscs/mch/s83/IFS/ifs_ens.zarr", consolidated=False)
        init_np = np.datetime64(init_t.isoformat())
        ds = ds_all.sel(init_time=init_np)
    else:
        zarr = baseline_zarr_path(baseline, init_tag)
        if not zarr.exists():
            print(f"MISS {zarr}")
            return
        print(f"{baseline} {init_tag} -> loading zarr...")
        ds = xr.open_zarr(zarr, consolidated=True).isel(init_time=0)
    # IFS-ENS: track only the stratified 10-member subset matching the
    # SwissClim eval pipeline (IFS_ENS_MEMBERS in evaluate_baselines.sh).
    # The full 50-member tracking was an inconsistency relative to every
    # other paper number.
    if baseline == "ifs_ens" and ds.sizes["ensemble"] >= 50:
        stratified = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        ds = ds.isel(ensemble=stratified)
    n_members = ds.sizes["ensemble"]
    out_dir = TRACKS_ROOT / baseline / init_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tracks = []
    for m in range(n_members):
        ncf = out_dir / f"m{m:02d}_input.nc"
        det = out_dir / f"m{m:02d}_detect.txt"
        sti = out_dir / f"m{m:02d}_stitch.txt"
        if sti.exists():
            print(f"  m{m:02d}: reusing existing stitch")
        else:
            make_member_nc(ds.isel(ensemble=m), ncf, init_t)
            try:
                run_tracker(ncf, det, sti)
            except Exception as e:
                print(f"  m{m:02d}: FAILED {e}")
                continue
            ncf.unlink()  # remove the input nc to save space
        track = extract_milton_track(sti)
        if not track.empty:
            track["baseline"] = baseline
            track["init_tag"] = init_tag
            track["member"] = m
            all_tracks.append(track)
            print(
                f"  m{m:02d}: Milton track {len(track)} pts, min MSL {track['psl_hpa'].min():.1f} hPa"
            )
        else:
            print(f"  m{m:02d}: no Milton track")
    if all_tracks:
        combined = pd.concat(all_tracks, ignore_index=True)
        combined.to_csv(out_dir / "milton_tracks.csv", index=False)
        print(f"saved {len(combined)} rows -> milton_tracks.csv")
    print(f"done {baseline} {init_tag}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
