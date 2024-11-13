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
    "interval", type=int,
    help="The time step in hours between each analysis time")
parser.add_argument("num_days", type=int, help="Number of days to download")
parser.add_argument("model_name", type=str, help="The ai-model name")

args = parser.parse_args()

settings.set(
    "user-cache-directory",
    "/scratch/mch/sadamov/temp/earthkit-cache")

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
    "number": "1/to/50/by/1",
    "param": single_level_params,
    "step": f"0/to/{args.num_days * 24}/by/{args.interval}",
    "stream": "enfo",
    "expect": "any",
    "time": hour,
    "type": "pf",
}

chunks = {
    "number": 1,
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
# ds_single.save(f"{args.date_time}/{args.model_name}/ifs_single.grib")

ds_single = ds_single.to_xarray(chunks=chunks_surface).drop_vars(
    "valid_time").chunk(chunks_surface)
# Split the "number" dimension into chunks
# TODO: This shouldn't be hardcoded to 50 members
number_chunks = [f"{i}/to/{i+9}/by/1" for i in range(1, 51, 10)]


# Retrieve the pressure level data in chunks because of MARS size limits
for i, number_chunk in enumerate(number_chunks):
    request.update(
        {
            "levtype": "pl",
            "levelist": pressure_levels,
            "param": pressure_level_params,
            "number": number_chunk,
        }
    )
    ds_pressure_chunk = earthkit.data.from_source("mars", request, lazily=True)

    shortnames = list(set(ds_pressure_chunk.metadata("shortName")))
    special_vars = ["r"]
    normal_vars = [var for var in shortnames if var not in special_vars]
    
    # Convert to xarray and chunk
    ds_normal = ds_pressure_chunk.sel(shortName=normal_vars).to_xarray(chunks=chunks)
    ds_special = ds_pressure_chunk.sel(shortName=special_vars).to_xarray(chunks=chunks)
    ds_combined = xr.merge([ds_normal, ds_special]).chunk(chunks).drop_vars("valid_time")

    # Write to Zarr with correct append_dim
    ds_combined.to_zarr(
        f"{path}/ifs_ens.zarr",
        consolidated=True,
        mode="w" if i == 0 else "a",
        append_dim="number" if i > 0 else None,
    )

print("Adding Surface data")
ds_single.drop_vars(
    ["z"] if "z" in ds_single.variables else []).to_zarr(
        f"{path}/ifs_ens.zarr",
        consolidated=True,
    mode="a")
