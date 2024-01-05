import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description="Evaluate the NeurWP Ensemble.")
parser.add_argument("model_name", type=str, help="The ai-model name")
parser.add_argument(
    "date_time",
    type=str,
    help="Date and time in the format YYYYMMDDHHMM")
parser.add_argument("perturbation", type=float, help="The perturbation size")
args = parser.parse_args()

path_out = os.path.join(
    args.model_name,
    str(args.date_time),
    str(args.perturbation))

forecast = xr.open_zarr(path_out + "/forecast.zarr", consolidated=True)
forecast_unperturbed = xr.open_zarr(
    args.model_name +
    "/forecast.zarr",
    consolidated=True)
ground_truth = xr.open_zarr("ground_truth.zarr", consolidated=True)

ground_truth['step'] = ('time', np.arange(len(ground_truth['time'])))
ground_truth = ground_truth.swap_dims({'time': 'step'})
forecast["step"] = forecast.time + forecast.step
forecast_unperturbed["step"] = forecast["step"]
ground_truth["step"] = forecast["step"]

# Calculate the spatial mean
fc_mean = forecast.mean(dim=["latitude", "longitude", "isobaricInhPa"])
fc_mean_unperturbed = forecast_unperturbed.mean(
    dim=["latitude", "longitude", "isobaricInhPa"])
gt_mean = ground_truth.mean(dim=["latitude", "longitude", "isobaricInhPa"])

# Calculate the std across all members for each forecast step and variable
ensemble_spread = fc_mean.std(dim=["member"])

# Calculate the ensemble skill for each forecast step and variable
ensemble_skill = abs(fc_mean.mean(dim="member") - gt_mean)

# Calculate the spread and skill at each forecast step
spread_skill_ratio = ensemble_spread / ensemble_skill

# Plot the spread and skill at each forecast step and variable
for variable in spread_skill_ratio.data_vars:
    print("Creating plots for variable: ", variable)
    data_array = spread_skill_ratio[variable]
    fig, ax = plt.subplots(figsize=(8, 6))
    # Assuming 'step' is a DataArray or Dataset of numpy.datetime64 objects
    data_array.plot(ax=ax)

    # Create a secondary axis
    ax2 = ax.twinx()
    # Plot the ensemble skill on the secondary axis
    ensemble_spread[variable].plot(ax=ax2, color='red')

    ax.set_xlabel("")
    ax.set_ylabel("Spread-skill ratio")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Spread-Skill Ratio and Ensemble Spread for {variable}")
    ax2.set_ylabel("Ensemble Spread", color='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.set_title("")
    plt.savefig(f"{path_out}/spread_skill_ratio_{variable}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot gt_mean in black
    gt_mean[variable].plot(ax=ax, color='black', label='ERA5')

    # Plot each member of fc_mean in green
    for i, member in enumerate(fc_mean.member):
        if i == 0:  # Add label only for the first member
            fc_mean[variable].sel(
                member=member).plot(
                ax=ax,
                color='green',
                alpha=0.3,
                label='AI-Model')
        else:
            fc_mean[variable].sel(member=member).plot(ax=ax, color='green', alpha=0.05)
    fc_mean[variable].median(
        dim="member").plot(
        ax=ax,
        color='orange',
        label='AI-Model Median')
    fc_mean_unperturbed[variable].plot(
        ax=ax, color='purple', label='AI-Model Unperturbed')

    ax.set_ylabel(f"{variable}")
    ax.set_title(f"Forecast vs Ground-Truth Comparison: {variable}")
    ax.legend()

    plt.savefig(f"{path_out}/{variable}_fc_gt.png")
    plt.close(fig)
