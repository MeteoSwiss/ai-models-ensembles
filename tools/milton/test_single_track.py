"""Single-init Milton tracker verification.

Picks AIFS-CRPS init 2024-10-04 00 UTC member 0 (5-day lead reaches peak
intensity 2024-10-07 18 UTC), runs TempestExtremes DetectNodes + StitchNodes
with the Walsh 2007 / Ullrich & Zarzycki 2017 parameter set, and compares to
IBTrACS truth.

This is the verification step before scaling to the 980-forecast SLURM batch.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE = Path("/iopsstor/scratch/cscs/sadamov/milton_case_study")
ZARR = Path(
    "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifsens/20241004_0000/forecast.zarr"
)
MEMBER = 0
NC_OUT = BASE / "test_input_aifsens_20241004_m0.nc"
DETECT_OUT = BASE / "test_detect_aifsens_20241004_m0.txt"
STITCH_OUT = BASE / "test_stitch_aifsens_20241004_m0.txt"


def make_input_nc() -> None:
    """Dump MSL + Z500 + Z850 + 10m wind + 850 vort for one member to a TE-ready NetCDF."""
    if NC_OUT.exists():
        print(f"reusing existing {NC_OUT}")
        return
    print(f"building {NC_OUT} from {ZARR.name}/member {MEMBER}...")
    ds = xr.open_zarr(ZARR, consolidated=True).isel(ensemble=MEMBER, init_time=0)
    # Resolve lead_time -> valid_time
    init_t = pd.Timestamp(ds["init_time"].values)
    lead_h = (ds["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    valid = init_t + pd.to_timedelta(lead_h, unit="h")

    # Build small subset (keep only fields we need)
    out = xr.Dataset(
        coords={
            "time": ("time", valid),
            "latitude": ds["latitude"],
            "longitude": ds["longitude"],
        }
    )
    out["psl"] = (
        ("time", "latitude", "longitude"),
        ds["mean_sea_level_pressure"].values.astype("float32"),
    )
    z = ds["geopotential"]
    z500 = z.sel(level=500).values.astype("float32")
    z850 = z.sel(level=850).values.astype("float32")
    out["z_thick_500m850"] = (("time", "latitude", "longitude"), (z500 - z850))
    u10 = ds["10m_u_component_of_wind"].values.astype("float32")
    v10 = ds["10m_v_component_of_wind"].values.astype("float32")
    out["v10"] = (("time", "latitude", "longitude"), np.sqrt(u10**2 + v10**2))
    # 850 hPa relative vorticity from u/v at 850 hPa
    u850 = ds["u_component_of_wind"].sel(level=850).values.astype("float64")
    v850 = ds["v_component_of_wind"].sel(level=850).values.astype("float64")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    R = 6.371e6
    dlat = np.deg2rad(np.gradient(lat))
    dlon = np.deg2rad(np.gradient(lon))
    cos_lat = np.cos(np.deg2rad(lat))[:, None]
    # central differences
    dv_dx = np.gradient(v850, axis=2) / (R * cos_lat * dlon[None, None, :])
    du_dy = np.gradient(u850, axis=1) / (R * dlat[None, :, None])
    vort850 = dv_dx - du_dy
    out["vort850"] = (("time", "latitude", "longitude"), vort850.astype("float32"))
    out.attrs["source"] = f"{ZARR.name}/member{MEMBER}"
    out.attrs["note"] = "Walsh 2007 / UZ17 TempestExtremes input"
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(NC_OUT, encoding=encoding)
    print(f"wrote {NC_OUT.stat().st_size/1e6:.1f} MB")


def run_detect() -> None:
    """Run DetectNodes with Walsh 2007 + UZ17 parameter set."""
    cmd = [
        "DetectNodes",
        "--in_data",
        str(NC_OUT),
        "--out",
        str(DETECT_OUT),
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
        "--timefilter",
        "6hr",
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("stdout (last 30 lines):")
    print("\n".join(r.stdout.splitlines()[-30:]))
    if r.returncode != 0:
        print("stderr:", r.stderr[-2000:])
        sys.exit(1)


def run_stitch() -> None:
    cmd = [
        "StitchNodes",
        "--in",
        str(DETECT_OUT),
        "--out",
        str(STITCH_OUT),
        "--in_fmt",
        "lon,lat,psl,v10,vort850",
        "--range",
        "5.0",
        "--mintime",
        "36h",
        "--maxgap",
        "6h",
        "--threshold",
        "v10,>=,17.0,3",
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("stdout (last 30 lines):")
    print("\n".join(r.stdout.splitlines()[-30:]))
    if r.returncode != 0:
        print("stderr:", r.stderr[-2000:])
        sys.exit(1)


if __name__ == "__main__":
    make_input_nc()
    run_detect()
    run_stitch()
    print()
    print("stitch output:")
    print(STITCH_OUT.read_text()[:3000])
