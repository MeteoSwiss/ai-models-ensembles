import argparse
import contextlib
import os

import numcodecs
import xarray as xr

# Create the parser
parser = argparse.ArgumentParser(description="Evaluate the NeurWP Ensemble.")

parser.add_argument("path_out", type=str, help="The path to the output directory")
# Parse the arguments
args = parser.parse_args()

grib_files = []

# Walk through root_dir
for dir_name, subdir_list, file_list in os.walk(args.path_out):
    # Check if any file in file_list ends with .grib and does not contain era5_init
    for file_name in file_list:
        if file_name.endswith(".grib") and "era5_init" not in file_name:
            # Construct full file path
            full_path = os.path.join(dir_name, file_name)
            grib_files.append(full_path)

grib_files.sort()
datasets = []

# BUG: There is an annoying GRIB issue, where the three variables below are present
# on mulitple heightAboveGround levels. Should be fixed in model-output rather than
# Define the shortNames and their corresponding filter keys
shortNames = ["100v", "100u", "2t", "rest"]
filter_keys = [{"shortName": "100v"}, {"shortName": "100u"}, {"shortName": "2t"}, None]

# Write the combined dataset to an array within the member"s group
path_store = os.path.join(
    args.path_out,
    "forecast.zarr"
)

ds_list = []
datasets = []
chunks = {"latitude": -1, "longitude": -1, "step": 1, "member": 1}

for i, grib_file in enumerate(grib_files):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stderr(devnull):
            for j, (shortName, filter_key) in enumerate(zip(shortNames, filter_keys)):
                # Open the dataset with the appropriate filter key and chunk sizes
                if filter_key is not None:
                    ds_list.append(xr.open_dataset(
                        grib_file, engine="cfgrib", backend_kwargs={
                            "filter_by_keys": filter_key}))
                else:
                    ds_list.append(xr.open_dataset(grib_file))
                # Add member=i dimension to the dataset
                ds = xr.merge(ds_list)
                ds = ds.assign_coords(member=i)
                ds = ds.expand_dims({"member": 1})
                ds = ds.chunk(chunks=chunks)
                datasets.append(ds)
                ds_list = []
    if os.path.exists(path_store):
        ds.to_zarr(store=path_store, mode="a", append_dim="member", consolidated=True)
    else:
        ds.to_zarr(store=path_store, mode="w", consolidated=True,
                   encoding={var: {"compressor": numcodecs.Zlib(level=1)}
                             for var in ds.data_vars})
    print(f"Stored {i+1}/{len(grib_files)} datasets")
