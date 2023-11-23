import argparse
import contextlib
import os

import xarray as xr

# Create the parser
parser = argparse.ArgumentParser(description='Evaluate the NeurWP Ensemble.')

# Add the arguments
parser.add_argument('MODEL_NAME', metavar='MODEL_NAME', type=str, help='the model name')
parser.add_argument('DATE_TIME', metavar='DATE_TIME', type=str, help='the date time')

# Parse the arguments
args = parser.parse_args()

root_dir = os.path.join(args.MODEL_NAME, args.DATE_TIME)
grib_files = []

# Walk through root_dir
for dir_name, subdir_list, file_list in os.walk(root_dir):
    # Check if MODEL_NAME.grib is in files
    if args.MODEL_NAME + ".grib" in file_list:
        # Construct full file path
        full_path = os.path.join(dir_name, args.MODEL_NAME + ".grib")
        grib_files.append(full_path)

grib_files.sort()
datasets = []

# BUG: There is an annoying GRIB issue, where the three variables below are present
# on mulitple heightAboveGround levels. Should be fixed in model-output rather than
# Define the shortNames and their corresponding filter keys
shortNames = ['100v', '100u', '2t', 'rest']
filter_keys = [{'shortName': '100v'}, {'shortName': '100u'}, {'shortName': '2t'}, None]
datasets = []

for i, grib_file in enumerate(grib_files[:2]):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            for j, (shortName, filter_key) in enumerate(zip(shortNames, filter_keys)):
                # Open the dataset with the appropriate filter key and chunk sizes
                if filter_key is not None:
                    ds = xr.open_dataset(
                        grib_file, engine='cfgrib', backend_kwargs={
                            'filter_by_keys': filter_key})
                else:
                    ds = xr.open_dataset(
                        grib_file)
                # Add member=i dimension to the dataset
                ds = ds.expand_dims(member=i)
                datasets.append(ds)
    print(f"Loaded {i+1}/{len(grib_files)} datasets")

combined = xr.concat(datasets, dim='member')
print("Chunking...")
combined = combined.chunk()

# Write the combined dataset to an array within the member's group
path_store = os.path.join(
    args.MODEL_NAME,
    args.DATE_TIME,
    'forecast.zarr'
)
print("Writing to zarr with path: ")
combined.to_zarr(
    store=path_store,
    mode="w",
    consolidated=True)

# TODO: delayed execution with dask might be necessary
