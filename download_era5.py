import argparse
import ast
import os
from datetime import datetime, timedelta

import earthkit.data
from earthkit.data import settings

parser = argparse.ArgumentParser(description="Download ERA5 data.")
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

settings.set("user-cache-directory", "/scratch/mch/sadamov/temp/earthkit-cache")

path = os.path.join(args.start_date, args.model_name)
with open(path + "/fields.txt", "r") as f:
    lines = f.readlines()

grid = ast.literal_eval(lines[0].split(": ")[1].strip())
area = ast.literal_eval(lines[1].split(": ")[1].strip())
pressure_levels = ast.literal_eval(lines[3].split(": ")[1].strip())
pressure_level_params = ast.literal_eval(lines[4].split(": ")[1].strip())
single_level_params = ast.literal_eval(lines[6].split(": ")[1].strip())

start_date = datetime.strptime(args.start_date, "%Y%m%d%H%M")
end_date = datetime.strptime(args.end_date, "%Y%m%d%H%M")

analysis_times = []
current_date = start_date
while current_date <= end_date:
    analysis_times.extend([current_date + timedelta(hours=h)
                          for h in range(0, 24, args.interval)])
    current_date += timedelta(days=1)

years = sorted(list({t.strftime("%Y") for t in analysis_times}))
months = sorted(list({t.strftime("%m") for t in analysis_times}))
days = sorted(list({t.strftime("%d") for t in analysis_times}))
times = sorted(list({t.strftime("%H:%M") for t in set(analysis_times)}))

ds_single = earthkit.data.from_source(
    "cds",
    "reanalysis-era5-single-levels",
    variable=single_level_params,
    product_type="reanalysis",
    area=area,
    grid=grid,
    year=years,
    month=months,
    day=days,
    time=times,
)
# ds_single.save(f"{args.start_date}/{args.model_name}/era5_single.grib")

ds_pressure = earthkit.data.from_source(
    "cds",
    "reanalysis-era5-pressure-levels",
    variable=pressure_level_params,
    product_type="reanalysis",
    area=area,
    grid=grid,
    year=years,
    month=months,
    day=days,
    time=times,
    levels=pressure_levels,
)
# ds_pressure.save(f"{args.start_date}/{args.model_name}/era5_pressure.grib")

ds_combined = ds_single + ds_pressure
ds_combined.isel(time=0).save(f"{args.start_date}/{args.model_name}/era5_init.grib")

chunks = {"latitude": -1, "longitude": -1, "time": 1, "isobaricInhPa": -1}
print("Saving ground truth to zarr...")
ds_combined = ds_combined.to_xarray().sel(
    time=slice(
        start_date,
        end_date)).chunk(
            chunks=chunks).to_zarr(
    f"{args.start_date}/{args.model_name}/ground_truth.zarr",
    mode="w",
    consolidated=True)
