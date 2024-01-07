import argparse
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description="Evaluate the NeurWP Ensemble.")
parser.add_argument(
    "date_time",
    type=str,
    help="Date and time in the format YYYYMMDDHHMM")
parser.add_argument("model_name", type=str, help="The ai-model name")
parser.add_argument("perturbation_init", type=float, help="The init perturbation size")
parser.add_argument(
    "perturbation_latent",
    type=float,
    help="The latent perturbation size")
args = parser.parse_args()


def plot_for_variable(
        variable, rmse, rmse_unperturbed, spread_skill_ratio, ensemble_spread, gt_mean,
        fc_mean, fc_mean_unperturbed, alpha_value, path_out):
    print("Creating plots for variable: ", variable)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each member's rmse in green
    for i, member in enumerate(rmse.member):
        rmse_member_mean = rmse[variable].sel(member=member)
        if i == 0:  # Add label only for the first member
            rmse_member_mean.plot(
                ax=ax,
                color='green',
                alpha=alpha_value,
                label='AI-Model Members')
        else:
            rmse_member_mean.plot(ax=ax, color='green', alpha=alpha_value)

    # Plot the unperturbed rmse in purple
    rmse_unperturbed[variable].plot(
        ax=ax, color='purple', label='AI-Model Unperturbed')

    ax.set_ylabel(f"{variable}")
    ax.set_title(f"Mean Absolute Global Forecast Error: {variable}")
    ax.legend()
    plt.savefig(f"{path_out}/rmse_{variable}.png")
    plt.close(fig)

    data_array = spread_skill_ratio[variable]
    fig, ax = plt.subplots(figsize=(8, 6))
    data_array.plot(ax=ax)

    # Create a secondary axis
    ax2 = ax.twinx()
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
    gt_mean[variable].plot(ax=ax, color='black', label='ERA5')
    # Plot each member of fc_mean in green
    for i, member in enumerate(fc_mean.member):
        if i == 0:  # Add label only for the first member
            fc_mean[variable].sel(
                member=member).plot(
                ax=ax,
                color='green',
                alpha=alpha_value,
                label='AI-Model Members')
        else:
            fc_mean[variable].sel(
                member=member).plot(
                ax=ax,
                color='green',
                alpha=alpha_value)
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
    plt.savefig(f"{path_out}/timeseries_fc_gt_{variable}.png")
    plt.close(fig)


if __name__ == "__main__":
    path_out = os.path.join(
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}")

    forecast = xr.open_zarr(path_out + "/forecast.zarr", consolidated=True)
    forecast_unperturbed = xr.open_zarr(
        f"{args.date_time}/{args.model_name}/forecast.zarr", consolidated=True)
    ground_truth = xr.open_zarr(
        args.date_time +
        "/ground_truth.zarr",
        consolidated=True)

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

    # Calculate the squared differences at each spatial point
    squared_diff = (forecast - ground_truth) ** 2
    squared_diff_unperturbed = (forecast_unperturbed - ground_truth) ** 2

    # Calculate the spatial mean of the squared differences
    mean_squared_diff = squared_diff.mean(
        dim=["latitude", "longitude", "isobaricInhPa"])
    mean_squared_diff_unperturbed = squared_diff_unperturbed.mean(
        dim=["latitude", "longitude", "isobaricInhPa"])

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_diff)
    rmse_unperturbed = np.sqrt(mean_squared_diff_unperturbed)

    ensemble_spread = rmse.std(dim=["member"])
    ensemble_skill = abs(rmse.mean(dim="member"))
    spread_skill_ratio = ensemble_spread / ensemble_skill

    alpha_value = 1 / rmse.member.size**(2 / 3)
    variables = list(spread_skill_ratio.data_vars)
    plot_args = (
        rmse,
        rmse_unperturbed,
        spread_skill_ratio,
        ensemble_spread,
        gt_mean,
        fc_mean,
        fc_mean_unperturbed,
        alpha_value,
        path_out)
    with multiprocessing.Pool() as pool:
        pool.starmap(
            plot_for_variable, [
                (variable,) + plot_args for variable in variables])
