import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description="Evaluate the NeurWP Ensemble.")
parser.add_argument("path_out", type=str, help="The path to the output directory")

args = parser.parse_args()

forecast = xr.open_zarr(args.path_out + "/forecast.zarr", consolidated=True)
ground_truth = xr.open_zarr("ground_truth.zarr", consolidated=True)

ground_truth['step'] = ('time', np.arange(len(ground_truth['time'])))
ground_truth = ground_truth.swap_dims({'time': 'step'})
ground_truth["step"] = forecast["step"]

# Calculate the ensemble spread for each forecast step and variable
ensemble_spread = forecast.std(dim=["member", "latitude", "longitude", "isobaricInhPa"])

# Calculate the ensemble skill for each forecast step and variable
ensemble_skill = abs(forecast.mean(dim="member") - ground_truth).mean(
    ["latitude", "longitude", "isobaricInhPa"])

# Calculate the spread and skill at each forecast step
spread_skill_ratio = ensemble_spread / ensemble_skill

# Plot the spread and skill at each forecast step and variable
for variable in spread_skill_ratio.data_vars:
    data_array = spread_skill_ratio[variable]
    fig, ax = plt.subplots()
    data_array.plot(ax=ax)
    ax.set_title(f"Spread-skill ratio for {variable}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Spread-skill ratio")
    ax.set_ylim(bottom=0)
    plt.savefig(f"{args.path_out}/spread_skill_ratio_{variable}.png")
    plt.close(fig)
