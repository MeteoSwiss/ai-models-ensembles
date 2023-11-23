import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description='Evaluate the NeurWP Ensemble.')
parser.add_argument(
    'MODEL_NAME',
    metavar='MODEL_NAME',
    type=str,
    help='the model grib output')
parser.add_argument(
    'DATE_TIME',
    type=bool,
    help='print pressure levels')
args = parser.parse_args()

forecast = xr.open_dataset("forecast.zarr")

# Concatenate all datasets into one
ground_truth = xr.open_dataset("era5.grib")

# Calculate the ensemble spread
ensemble_spread = forecast.std(dim='member')

# Calculate the ensemble skill
ensemble_skill = np.mean(
    np.abs(
        forecast.mean(
            dim='member') -
        ground_truth),
    dim=[
        'latitude',
        'longitude'])

# Group the spread and skill by forecast step
grouped_spread = ensemble_spread.groupby('step')
grouped_skill = ensemble_skill.groupby('step')

# Calculate the spread and skill at each forecast step
spread_skill_ratio = grouped_spread.mean() / grouped_skill.mean()

# Create a line plot of the spread-skill ratio
plt.figure(figsize=(10, 6))
spread_skill_ratio.plot.line('o-')
plt.title('Spread-Skill Ratio Over Forecast Step')
plt.xlabel('Forecast Step')
plt.ylabel('Spread-Skill Ratio')
plt.grid(True)
plt.savefig('spread_skill_ratio.png')
