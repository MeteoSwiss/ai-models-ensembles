import argparse
import ast
import os
from datetime import datetime

import earthkit.data
import xarray as xr
from earthkit.data import settings

parser = argparse.ArgumentParser(description="Download IFS data.")
parser.add_argument("out_dir", type=str, help="The output directory")
parser.add_argument(
    "date_time", type=str, help="Date and time in the format YYYYMMDDHHMM"
)
parser.add_argument(
    "interval", type=int, help="The time step in hours between each analysis time"
)
parser.add_argument("num_days", type=int, help="Number of days to download")
parser.add_argument("model_name", type=str, help="The ai-model name")

args = parser.parse_args()

settings.set("user-cache-directory", "/scratch/mch/sadamov/temp/earthkit-cache")

# Read parameters from fields.txt
path = os.path.join(args.out_dir, args.date_time, args.model_name)
with open(path + "/fields.txt", "r") as f:
    lines = f.readlines()

grid = ast.literal_eval(lines[0].split(": ")[1].strip())
area = ast.literal_eval(lines[1].split(": ")[1].strip())
pressure_levels = ast.literal_eval(lines[3].split(": ")[1].strip())
pressure_level_params = ast.literal_eval(lines[4].split(": ")[1].strip())
single_level_params = ast.literal_eval(lines[6].split(": ")[1].strip())

# Extract date components
date = datetime.strptime(args.date_time, "%Y%m%d%H%M")
year = date.strftime("%Y")
month = date.strftime("%m")
day = date.strftime("%d")
hour = date.strftime("%H")

# Build the request
request = {
    "area": area,
    "class": "od",
    "date": f"{year}-{month}-{day}",
    "expver": "1",
    "grid": grid,
    "levtype": "sfc",
    "param": single_level_params,
    "step": f"0/to/{args.num_days * 24}/by/{args.interval}",
    "stream": "enfo",
    "expect": "any",
    "time": hour,
    "type": "cf",
}

chunks = {
    "step": 1,
    "time": -1,
    "isobaricInhPa": 1,
    "latitude": -1,
    "longitude": -1,
}
chunks_surface = chunks.copy()
chunks_surface["surface"] = chunks_surface.pop("isobaricInhPa")

# Retrieve the single level data
ds_single = earthkit.data.from_source("mars", request, lazily=True)

ds_single = (
    ds_single.to_xarray(chunks=chunks_surface)
    .drop_vars("valid_time")
    .chunk(chunks_surface)
)

# Retrieve the pressure level data in chunks because of MARS size limits
request.update(
    {
        "levtype": "pl",
        "levelist": pressure_levels,
        "param": pressure_level_params,
    }
)
ds_pressure = earthkit.data.from_source("mars", request, lazily=True)

shortnames = list(set(ds_pressure.metadata("shortName")))
special_vars = ["r"]
normal_vars = [var for var in shortnames if var not in special_vars]

ds_normal = ds_pressure.sel(shortName=normal_vars).to_xarray(chunks=chunks)
ds_special = ds_pressure.sel(shortName=special_vars).to_xarray(chunks=chunks)
ds_combined = xr.merge([ds_normal, ds_special]).chunk(chunks).drop_vars("valid_time")

print("Writing to zarr")
ds_combined.to_zarr(
    f"{path}/ifs_control.zarr",
    consolidated=True,
    mode="w",
)

print("Adding Surface data")
ds_single.drop_vars(["z"] if "z" in ds_single.variables else []).to_zarr(
    f"{path}/ifs_control.zarr", consolidated=True, mode="a"
)
