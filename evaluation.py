import argparse
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import rankdata

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
parser.add_argument("members", type=int, help="The number of ensemble members")
parser.add_argument(
    "--plot_vertical_levels",
    action="store_true",
    help="Plot vertical levels")
args = parser.parse_args()


def plot_rank_histogram(variable, forecast, ground_truth,
                        path_out, color_palette, level=None):
    print(f"Creating rank histogram for variable: {variable}, level: {level}")

    # Select the level if provided
    forecast_var = forecast[variable].sel(
        isobaricInhPa=level) if level else forecast[variable]
    ground_truth_var = ground_truth[variable].sel(
        isobaricInhPa=level) if level else ground_truth[variable]

    # Convert to numpy arrays
    forecast_np = forecast_var.values
    # Add an axis for 'member'
    ground_truth_np = ground_truth_var.values[np.newaxis, ...]

    # Stack ground truth as a new member at the end
    combined_np = np.concatenate([forecast_np, ground_truth_np], axis=0)

    # Compute ranks along the 'member' axis using 'ordinal' method to avoid half ranks
    ranks_np = np.apply_along_axis(
        lambda x: rankdata(x, method='ordinal'),
        axis=0, arr=combined_np)

    # Get the rank of the ground truth
    gt_rank_np = ranks_np[-1, ...]  # Last member is the ground truth

    # Count the occurrence of each rank
    unique, counts = np.unique(gt_rank_np, return_counts=True)
    rank_counts = dict(zip(unique, counts))

    # Plot the rank histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        list(rank_counts.keys()),
        list(rank_counts.values()),
        color=color_palette[4],
        edgecolor=color_palette[5])
    ax.set_title(
        f"Rank Histogram for {variable}{' at level ' + str(level) if level else ''}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    plt.savefig(
        os.path.join(
            path_out,
            f"rank_histogram_{variable}{'_' + str(level) if level else ''}.png"))
    plt.close(fig)


def calculate_and_plot_energy_spectra(
        variable, forecast, forecast_unperturbed, ground_truth, alpha_value, path_out,
        color_palette, level=None):
    print(
        f"Calculating and plotting energy spectra for variable: {variable}, level: {level}")

    # Helper function to calculate PSD and radial frequency bins
    def calculate_psd(field):
        fft_result = np.fft.fft2(field.values)
        fft_shifted = np.fft.fftshift(fft_result)
        psd = np.abs(fft_shifted)**2
        return psd

    # Select the level if provided
    if level is not None:
        forecast = forecast.sel(isobaricInhPa=level)
        forecast_unperturbed = forecast_unperturbed.sel(isobaricInhPa=level)
        ground_truth = ground_truth.sel(isobaricInhPa=level)

    # Calculate the frequency bins for plotting
    freq_x = np.fft.fftfreq(
        forecast.sizes['longitude'],
        d=1.0)  # Assuming unit grid spacing
    freq_y = np.fft.fftfreq(forecast.sizes['latitude'], d=1.0)
    freq_x = np.fft.fftshift(freq_x)
    freq_y = np.fft.fftshift(freq_y)

    # Calculate radial frequency bins
    freq_r = np.sqrt(freq_x[None, :]**2 + freq_y[:, None]**2)

    # Plot the energy spectra
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot for each ensemble member
    for i, member in enumerate(forecast.member.values):
        member_field = forecast[variable].sel(member=member).mean(dim='step')
        psd_member = calculate_psd(member_field)
        radial_psd_member, radial_bins = np.histogram(
            freq_r, bins=30, weights=psd_member)
        if i == 0:  # Only label the first member
            ax.loglog(radial_bins[:-1], radial_psd_member, color=color_palette[1],
                      label='AI-Model Members', alpha=alpha_value)
        else:
            ax.loglog(radial_bins[:-1], radial_psd_member,
                      alpha=alpha_value, color=color_palette[1])

    # Plot unperturbed and ground truth
    mean_field_perturbed = forecast[variable].median(dim=['member']).mean(dim=['step'])
    mean_field_unperturbed = forecast_unperturbed[variable].mean(dim=['member', 'step'])
    mean_field_ground_truth = ground_truth[variable].mean(dim='step')
    psd_perturbed = calculate_psd(mean_field_perturbed)
    psd_unperturbed = calculate_psd(mean_field_unperturbed)
    psd_ground_truth = calculate_psd(mean_field_ground_truth)
    radial_psd_perturbed, _ = np.histogram(freq_r, bins=30, weights=psd_perturbed)
    radial_psd_unperturbed, _ = np.histogram(freq_r, bins=30, weights=psd_unperturbed)
    radial_psd_ground_truth, _ = np.histogram(freq_r, bins=30, weights=psd_ground_truth)
    ax.loglog(radial_bins[:-1], radial_psd_perturbed, color=color_palette[2],
              label='AI-Model Median')
    ax.loglog(radial_bins[:-1],
              radial_psd_unperturbed,
              label='AI-Model Unperturbed',
              linestyle='--',
              color=color_palette[3])
    ax.loglog(radial_bins[:-1], radial_psd_ground_truth, color=color_palette[0],
              label='Ground Truth: ERA5', linestyle=':')

    # Rest of the plotting code remains the same...
    # Retrieve the current x-axis major and minor tick locations
    major_xticks = ax.get_xticks(minor=False)
    minor_xticks = ax.get_xticks(minor=True)
    # Set the tick positions using FixedLocator
    ax.xaxis.set_major_locator(FixedLocator(major_xticks))
    ax.xaxis.set_minor_locator(FixedLocator(minor_xticks))
    # Set the labels for the x-axis major and minor ticks
    ax.set_xticklabels(['{:.0e}'.format(tick) for tick in major_xticks], minor=False)
    ax.set_xticklabels(['{:.0e}'.format(tick) for tick in minor_xticks], minor=True)
    for label in ax.get_xticklabels(which='both'):
        label.set_rotation(45)

    ax.set_title(
        f"Energy Spectra for {variable}{' at level ' + str(level) if level is not None else ''}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power Spectral Density")
    ax.legend()
    plt.savefig(
        os.path.join(
            path_out,
            f"energy_spectra_comparison_{variable}{'_' + str(level) if level is not None else ''}.png"))
    plt.close(fig)


def plot_rmse(variable, rmse, rmse_median, rmse_unperturbed, alpha_value,
              path_out, color_palette, level=None):
    print(f"Creating RMSE plots for variable: {variable}, level: {level}")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Select the level if provided
    if level is not None:
        rmse = rmse.sel(isobaricInhPa=level)
        rmse_unperturbed = rmse_unperturbed.sel(isobaricInhPa=level)

    for i, member in enumerate(rmse.member):
        rmse_member_mean = rmse[variable].sel(member=member)
        if i == 0:  # Add label only for the first member
            rmse_member_mean.plot(
                ax=ax,
                color=color_palette[1],
                alpha=alpha_value,
                label='AI-Model Members')
        else:
            rmse_member_mean.plot(ax=ax, color=color_palette[1], alpha=alpha_value)

    rmse_median[variable].plot(
        ax=ax,
        color=color_palette[2],
        label='AI-Model Median')

    rmse_unperturbed[variable].plot(
        ax=ax,
        color=color_palette[3],
        label='AI-Model Unperturbed')

    ax.set_ylabel(f"RMSE ({variable})")
    ax.set_title(
        f"Root Mean Square Error: {variable}{' at level ' + str(level) if level is not None else ''}")
    ax.legend()
    plt.savefig(
        os.path.join(
            path_out,
            f"rmse_{variable}{'_' + str(level) if level is not None else ''}.png"))
    plt.close(fig)


def plot_spread_skill_ratio(
        variable, spread_skill_ratio, ensemble_spread, path_out, color_palette,
        level=None):
    print(f"Creating spread-skill ratio plots for variable: {variable}, level: {level}")
    if level is not None:
        data_array = spread_skill_ratio[variable].sel(isobaricInhPa=level)
        ensemble_spread_data = ensemble_spread[variable].sel(isobaricInhPa=level)
    else:
        data_array = spread_skill_ratio[variable]
        ensemble_spread_data = ensemble_spread[variable]

    fig, ax = plt.subplots(figsize=(8, 6))
    data_array.plot(ax=ax, color=color_palette[5])

    ax2 = ax.twinx()
    ensemble_spread_data.plot(ax=ax2, color=color_palette[4])

    ax.set_xlabel("")
    ax.set_ylabel("Spread-skill ratio")
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Spread-Skill Ratio and Ensemble Spread for {variable}{' at level ' + str(level) if level is not None else ''}")
    ax2.set_ylabel("Ensemble Spread", color=color_palette[4])
    ax2.tick_params(axis='y', colors=color_palette[4])
    ax2.set_title("")
    plt.savefig(
        os.path.join(
            path_out,
            f"spread_skill_ratio_{variable}{'_' + str(level) if level is not None else ''}.png"))
    plt.close(fig)


def plot_timeseries_fc_gt(
        variable, gt_mean, fc_mean, fc_mean_unperturbed, alpha_value, path_out,
        color_palette, level=None):
    print(f"Creating timeseries plots for variable: {variable}, level: {level}")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Select the level if provided
    gt_mean_var = gt_mean[variable].sel(
        isobaricInhPa=level) if level is not None else gt_mean[variable]
    fc_mean_var = fc_mean[variable].sel(
        isobaricInhPa=level) if level is not None else fc_mean[variable]
    fc_mean_unperturbed_var = fc_mean_unperturbed[variable].sel(
        isobaricInhPa=level) if level is not None else fc_mean_unperturbed[variable]

    gt_mean_var.plot(ax=ax, color=color_palette[0], label='Ground Truth: ERA5')

    for i, member in enumerate(fc_mean.member):
        fc_mean_member = fc_mean_var.sel(member=member)
        if i == 0:  # Add label only for the first member
            fc_mean_member.plot(
                ax=ax,
                color=color_palette[1],
                alpha=alpha_value,
                label='AI-Model Members')
        else:
            fc_mean_member.plot(ax=ax, color=color_palette[1], alpha=alpha_value)

    fc_mean_var.median(
        dim="member").plot(
        ax=ax,
        color=color_palette[2],
        label='AI-Model Median')
    fc_mean_unperturbed_var.plot(
        ax=ax, color=color_palette[3],
        label='AI-Model Unperturbed')

    # Collect all values for the density plot at the latest time step
    fc_values_last_step = fc_mean_var.isel(step=-1).values

    # Create an inset axis for the density plot
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", size=1.2, pad=0.1)

    # Plot the density distribution rotated (horizontal)
    sns.kdeplot(y=fc_values_last_step, ax=ax2, color=color_palette[1], bw_adjust=0.5)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('Density')

    ax.set_ylabel(f"{variable}{' at level ' + str(level) if level is not None else ''}")
    ax.set_title(
        f"Forecast vs Ground-Truth Comparison: {variable}{' at level ' + str(level) if level is not None else ''}")
    ax.legend()
    plt.savefig(
        os.path.join(
            path_out,
            f"timeseries_fc_gt_{variable}{'_' + str(level) if level is not None else ''}.png"))
    plt.close(fig)


if __name__ == "__main__":
    path_out = os.path.join(
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}")
    forecast = xr.open_zarr(
        path_out + "/forecast.zarr",
        consolidated=True
    )
    forecast_unperturbed = xr.open_zarr(
        f"{args.date_time}/{args.model_name}/forecast.zarr",
        consolidated=True
    )
    ground_truth = xr.open_zarr(
        args.date_time + "/ground_truth.zarr",
        consolidated=True
    )

    indices = np.random.permutation(forecast.member.size)
    selected_indices = indices[:args.members]
    forecast = forecast.isel(member=selected_indices)

    ground_truth['step'] = ('time', np.arange(len(ground_truth['time'])))
    ground_truth = ground_truth.swap_dims({'time': 'step'})
    forecast["step"] = forecast.time + forecast.step
    forecast_unperturbed["step"] = forecast["step"]
    ground_truth["step"] = forecast["step"]

    # Calculate the spatial mean
    fc_mean = forecast.mean(dim=["latitude", "longitude"])
    fc_mean_unperturbed = forecast_unperturbed.mean(
        dim=["latitude", "longitude"])
    gt_mean = ground_truth.mean(dim=["latitude", "longitude"])

    # Calculate the squared differences at each spatial point
    squared_diff = (forecast - ground_truth) ** 2
    squared_diff_median = (forecast.median(dim="member") - ground_truth) ** 2
    squared_diff_unperturbed = (forecast_unperturbed - ground_truth) ** 2

    # Calculate the RMSE for each grid point across the ensemble members
    rmse_grid = np.sqrt(squared_diff.mean(dim="member"))
    rmse_grid_unperturbed = np.sqrt(squared_diff_unperturbed.mean(dim="member"))
    rmse_ensemble = np.sqrt(
        squared_diff.mean(
            dim=[
                "latitude",
                "longitude"]))
    rmse_median = np.sqrt(
        squared_diff_median.mean(
            dim=[
                "latitude",
                "longitude"]))
    rmse_ensemble_unperturbed = np.sqrt(
        squared_diff_unperturbed.mean(
            dim=[
                "latitude",
                "longitude"]))

    # Calculate the ensemble spread for each grid point across the ensemble members
    ensemble_spread_grid = forecast.std(dim="member")

    # Calculate the spread/skill ratio for each grid point
    spread_skill_ratio_grid = (ensemble_spread_grid / rmse_grid)

    # Calculate the mean spread/skill ratio across all grid points
    mean_spread_skill_ratio = spread_skill_ratio_grid.mean(
        dim=["latitude", "longitude"])

    # Calculate the mean ensemble spread across all grid points (for use in
    # plotting functions)
    mean_ensemble_spread = ensemble_spread_grid.mean(
        dim=["latitude", "longitude"])

    alpha_value = 1 / forecast.member.size**(5 / 8)

    # Define a custom colorblind-friendly palette
    color_palette = sns.color_palette([
        "#f75b78",
        "#6495ed",
        "#0e2d75",
        "#f9c740",
        "#45b7aa",
        "#353434",
    ])

    variables = list(forecast.data_vars)

    # Prepare arguments for each plot function, considering vertical levels
    rank_histogram_args = []
    energy_spectra_args = []
    rmse_args = []
    spread_skill_ratio_args = []
    timeseries_fc_gt_args = []

    vars_3d = [var for var in variables if 'isobaricInhPa' in forecast[var].dims]
    vars_2d = [var for var in variables if 'isobaricInhPa' not in forecast[var].dims]

    for variable in vars_3d:
        if args.plot_vertical_levels:
            # Variable has vertical levels, loop over each level
            for level in forecast[variable].coords['isobaricInhPa'].values:
                rank_histogram_args.append(
                    (variable, forecast, ground_truth, path_out, color_palette, level))
                energy_spectra_args.append(
                    (variable,
                     forecast,
                     forecast_unperturbed,
                     ground_truth,
                     alpha_value,
                     path_out,
                     color_palette,
                     level))
                rmse_args.append(
                    (variable,
                     rmse_ensemble,
                     rmse_median,
                     rmse_ensemble_unperturbed,
                     alpha_value,
                     path_out,
                     color_palette,
                     level))
                spread_skill_ratio_args.append(
                    (variable,
                     mean_spread_skill_ratio,
                     mean_ensemble_spread,
                     path_out,
                     color_palette,
                     level))
                timeseries_fc_gt_args.append(
                    (variable,
                     gt_mean,
                     fc_mean,
                     fc_mean_unperturbed,
                     alpha_value,
                     path_out,
                     color_palette,
                     level))
    for variable in vars_2d:
        # Variable does not have vertical levels, no need to loop
        rank_histogram_args.append(
            (variable, forecast, ground_truth, path_out, color_palette))
        energy_spectra_args.append(
            (variable,
             forecast,
             forecast_unperturbed,
             ground_truth,
             alpha_value,
             path_out,
             color_palette))
        rmse_args.append(
            (variable,
                rmse_ensemble,
                rmse_median,
                rmse_ensemble_unperturbed,
                alpha_value,
                path_out,
                color_palette))
        spread_skill_ratio_args.append(
            (variable,
             mean_spread_skill_ratio,
             mean_ensemble_spread,
             path_out,
             color_palette))
        timeseries_fc_gt_args.append(
            (variable,
             gt_mean,
             fc_mean,
             fc_mean_unperturbed,
             alpha_value,
             path_out,
             color_palette))
    # Use multiprocessing to execute all plot functions in parallel
    with multiprocessing.Pool(processes=1) as pool:
        pool.starmap(plot_rank_histogram, rank_histogram_args)
    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(plot_rmse, rmse_args)
        pool.starmap(plot_spread_skill_ratio, spread_skill_ratio_args)
        pool.starmap(calculate_and_plot_energy_spectra, energy_spectra_args)
        pool.starmap(plot_timeseries_fc_gt, timeseries_fc_gt_args)
