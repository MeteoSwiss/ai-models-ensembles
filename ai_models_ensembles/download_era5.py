import argparse
import ast
import os
from datetime import datetime, timedelta

import earthkit.data
import xarray as xr
from earthkit.data import settings

parser = argparse.ArgumentParser(description="Download ERA5 data.")
parser.add_argument("out_dir", type=str, help="The output directory")
parser.add_argument(
    "start_date", type=str, help="Start date in the format YYYYMMDDHHMM"
)
parser.add_argument(
    "end_date", type=str, help="End date in the format YYYYMMDDHHMM"
)
parser.add_argument(
    "interval",
    type=int,
    help="The time step in hours between each analysis time")
parser.add_argument("model_name", type=str, help="The ai-model name")

args = parser.parse_args()

settings.set("cache-policy", "user")
settings.set(
    "user-cache-directory",
    "/scratch/mch/sadamov/temp/earthkit-cache")

path = os.path.join(args.out_dir, args.start_date, args.model_name)
with open(path + "/fields.txt", "r") as f:
    lines = f.readlines()

grid = ast.literal_eval(lines[0].split(": ")[1].strip())
area = ast.literal_eval(lines[1].split(": ")[1].strip())
pressure_levels = ast.literal_eval(lines[3].split(": ")[1].strip())
pressure_level_params = ast.literal_eval(lines[4].split(": ")[1].strip())
single_level_params = ast.literal_eval(lines[6].split(": ")[1].strip())

start_date = datetime.strptime(args.start_date, "%Y%m%d%H%M")
end_date = datetime.strptime(args.end_date, "%Y%m%d%H%M")

if args.model_name == "graphcast":
    print("Downloading data for GraphCast model...", flush=True)
    start_date_prev = start_date - timedelta(hours=6)
else:
    print("Downloading data for FCN model...", flush=True)
analysis_times = []
current_date = start_date_prev
while current_date <= end_date:
    analysis_times.extend([current_date + timedelta(hours=h)
                           for h in range(0, 24, args.interval)])
    current_date += timedelta(days=1)

dates = sorted(list({t.strftime("%Y-%m-%d") for t in analysis_times}))
times = sorted(list({t.strftime("%H:%M") for t in set(analysis_times)}))
date_prev = int(start_date_prev.strftime("%Y%m%d"))
time_prev = int(start_date_prev.strftime("%H%M"))
date_now = int(start_date.strftime("%Y%m%d"))
time_now = int(start_date.strftime("%H%M"))

ds_single = earthkit.data.from_source(
    "cds",
    "reanalysis-era5-single-levels",
    variable=single_level_params,
    product_type="reanalysis",
    area=area,
    grid=grid,
    date=dates,
    time=times,
)

ds_pressure = earthkit.data.from_source(
    "cds",
    "reanalysis-era5-pressure-levels",
    variable=pressure_level_params,
    product_type="reanalysis",
    area=area,
    grid=grid,
    date=dates,
    time=times,
    levels=pressure_levels,
)

ds_single = ds_single.sel(stepRange=["11-12", "0"])

# TP times have to be shifted to match expected 12-hourly time steps
if args.model_name == "graphcast":
    ds_tp = ds_single.sel({"shortName": "tp"})
    ds_rest = ds_single.sel(
        {"shortName": ['lsm', '2t', 'msl', '10u', '10v', 'z']})
    ds_tp_prev = ds_tp.isel(date=1, time=1)
    ds_rest_prev = ds_rest.sel(date=date_prev, time=time_prev)
    ds_tp_now = ds_tp.isel(date=2, time=1)
    ds_rest_now = ds_rest.sel(date=date_now, time=time_now)
    ds_pressure_prev = ds_pressure.sel(date=date_prev, time=time_prev)
    ds_pressure_now = ds_pressure.sel(date=date_now, time=time_now)
    ds_combined = ds_tp_prev + ds_tp_now + ds_rest_prev + ds_rest_now + ds_pressure_prev + ds_pressure_now
else:
    ds_combined = ds_single + ds_pressure
    ds_combined = ds_combined.isel(date=0, time=0)

print("Saving initial conditions to grib...")
ds_combined.save(f"{path}/era5_init.grib")

if args.model_name == "graphcast":
    ds_rest = ds_rest.to_xarray().drop_vars(["z"])
    ds_rest['surface_z'] = ds_single.sel({"shortName": "z"}).to_xarray()['z']
    ds_combined_xr = xr.merge(
        [ds_tp.to_xarray(), ds_rest, ds_pressure.to_xarray()])
else:
    ds_combined_xr = xr.merge([ds_single.to_xarray(), ds_pressure.to_xarray()])

ds_combined_xr = ds_combined_xr.sel(time=slice(
    start_date, end_date + timedelta(hours=1)))

chunks = {"latitude": -1, "longitude": -1, "time": 1, "isobaricInhPa": -1}
print("Saving ground truth to zarr...")
ds_combined_xr.chunk(
    chunks=chunks).drop_vars(["valid_time"]).to_zarr(
    f"{path}/ground_truth.zarr",
    mode="w",
    consolidated=True)
