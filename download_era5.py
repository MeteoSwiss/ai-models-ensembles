import argparse
import ast
from datetime import datetime, timedelta

import earthkit.data

# Parse command line arguments
parser = argparse.ArgumentParser(description='Download ERA5 data.')
parser.add_argument(
    'DATE_TIME',
    type=str,
    help='Date and time in the format YYYYMMDDHHMM')
args = parser.parse_args()

# Read parameters from fields.txt
with open("fields.txt", "r") as f:
    lines = f.readlines()

grid = ast.literal_eval(lines[0].split(": ")[1].strip())
area = ast.literal_eval(lines[1].split(": ")[1].strip())
pressure_levels = ast.literal_eval(lines[3].split(": ")[1].strip())
pressure_level_params = ast.literal_eval(lines[4].split(": ")[1].strip())
single_level_params = ast.literal_eval(lines[6].split(": ")[1].strip())

# Convert the initial date and time to a datetime object
datetime_obj = datetime.strptime(args.DATE_TIME, '%Y%m%d%H%M')

# Initialize an empty list to store the datasets
datasets = []

# Repeat the retrieval for the next 240 hours (6 hourly intervals)
for i in range(41):
    # Convert the date and time to the YYYYMMDD and HHMM formats
    date = datetime_obj.strftime('%Y%m%d')
    time = datetime_obj.strftime('%H%M')

    # Retrieve data for the current date and time
    ds_single = earthkit.data.from_source(
        "cds",
        "reanalysis-era5-single-levels",
        variable=single_level_params,
        product_type="reanalysis",
        area=area,
        grid=grid,
        date=date,
        time=time)

    ds_pressure = earthkit.data.from_source(
        "cds",
        "reanalysis-era5-pressure-levels",
        variable=pressure_level_params,
        product_type="reanalysis",
        area=area,
        grid=grid,
        date=date,
        time=time,
        levels=pressure_levels)

    # Combine the single-level and pressure-level datasets
    if i == 0:
        ds_combined = ds_single + ds_pressure
        # Save to GRIB
        print("Saving initial conditions to GRIB...")
        ds_combined.save("era5_init.grib")
    elif i == 1:
        ds_combined = ds_single + ds_pressure
    else:
        ds_combined += ds_single + ds_pressure

    # Add 6 hours to the datetime
    datetime_obj += timedelta(hours=6)

# Save to GRIB
print("Saving to GRIB...")
ds_combined.save("era5.grib")
