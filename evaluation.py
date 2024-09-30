import argparse
import os

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("Agg")
matplotlib.rcParams.update({'font.size': 15})
plt.ioff()

parser = argparse.ArgumentParser(description="Evaluate the NeurWP Ensemble.")
parser.add_argument(
    "date_time", type=str, help="Date and time in the format YYYYMMDDHHMM"
)
parser.add_argument("model_name", type=str, help="The ai-model name")
parser.add_argument(
    "perturbation_init",
    type=float,
    help="The init perturbation size")
parser.add_argument(
    "perturbation_latent", type=float, help="The latent perturbation size"
)
parser.add_argument(
    "members",
    type=int,
    help="The number of ensemble members")
parser.add_argument(
    "crop_region",
    type=str,
    help="The region to crop the data to")
parser.add_argument(
    "debug",
    type=bool,
    help="Whether to run in debug mode",
    default=False)

args = parser.parse_args()

config = {
    "color_palette": sns.color_palette(
        ["#f75b78", "#6495ed", "#0e2d75", "#f9c740", "#45b7aa", "#353434"]
    ),
    "sample_size": 100000,
    "selected_vars": ["t2m"]
}


def load_and_prepare_data(path_in, crop_region, debug_mode=False):
    """
    Load the data and prepare it for evaluation.

    Args:
        path_in (str): The path to the input data.
        crop_region (str): The region to crop the data to. Can be "europe" or "global".

    Returns:
        dict: A dictionary containing
            ground truth (xr.Dataset): The ground truth data.
            forecast (xr.Dataset): The forecast data.
            forecast_unperturbed (xr.Dataset): The unperturbed forecast data.
            forecast_ifs (xr.Dataset): The IFS forecast data.
            forecast_unperturbed_ifs (xr.Dataset): The unperturbed IFS forecast data.
    """

    ground_truth = xr.open_zarr(
        f"{path_in}/ground_truth.zarr",
        consolidated=True).isel(
        number=0,
        surface=0)
    forecast = xr.open_zarr(
        f"{path_in}/init_{args.perturbation_init}_latent_{args.perturbation_latent}/forecast.zarr",
        consolidated=True,)
    forecast_unperturbed = xr.open_zarr(
        f"{path_in}/forecast.zarr", consolidated=True)
    forecast_ifs = xr.open_zarr(f"{path_in}/ifs_ens.zarr", consolidated=True)

    if crop_region == "europe":
        lat_min, lat_max = 25, 80
        lon_min, lon_max = 340, 50

        # Use modulo arithmetic to ensure longitudes are in 0-360 range
        lon_min = lon_min % 360
        lon_max = lon_max % 360

        # Create a list of longitudes that wraps around 0/360
        lats = list(range(lat_min, lat_max + 1))
        lons = list(range(lon_min, 360)) + list(range(0, lon_max + 1))

        # Crop all datasets to the European lat-lon box
        ground_truth = ground_truth.sel(latitude=lats, longitude=lons)
        forecast = forecast.sel(latitude=lats, longitude=lons)
        forecast_unperturbed = forecast_unperturbed.sel(
            latitude=lats, longitude=lons)
        forecast_ifs = forecast_ifs.sel(latitude=lats, longitude=lons)
    else:
        lat_min, lat_max = -90, 90
        lon_min, lon_max = 0, 360

    forecast_ifs = (
        forecast_ifs.rename_dims({"number": "member"})
        .rename({"number": "member"})
        .set_index(member="member")
        .reindex()
        .isel(surface=0)
    )
    ground_truth = ground_truth.isel(step=0)
    forecast_ifs["member"] = forecast_ifs["member"] - 1
    forecast_unperturbed = xr.concat(
        [ground_truth, forecast_unperturbed], "step")
    forecast_unperturbed = forecast_unperturbed.isel(time=0).sel(
        member=0,
    )
    forecast_unperturbed_ifs = forecast_ifs.isel(time=0).sel(
        member=0,
    )
    forecast_ifs = forecast_ifs.isel(member=slice(1, None))
    forecast = xr.concat(
        [ground_truth, forecast],
        "step",
    )

    indices = np.random.permutation(forecast.member.size)
    indices_ifs = np.random.permutation(forecast_ifs.member.size)
    selected_indices = indices[: args.members]
    selected_indices_ifs = indices_ifs[: args.members]
    forecast = forecast.isel(member=selected_indices, time=0)
    forecast_ifs = forecast_ifs.isel(member=selected_indices_ifs, time=0)

    ground_truth["step"] = ("time", np.arange(len(ground_truth["time"])))
    ground_truth = ground_truth.swap_dims({"time": "step"})
    forecast["step"] = forecast.time + forecast.step
    forecast_ifs["step"] = forecast["step"]
    forecast_unperturbed["step"] = forecast["step"]
    forecast_unperturbed_ifs["step"] = forecast["step"]
    ground_truth["step"] = forecast["step"]

    forecast = forecast.transpose(..., "member")
    forecast_ifs = forecast_ifs.transpose(..., "member")

    full_levels = forecast_ifs.isobaricInhPa.where(
        ~forecast_ifs.t.isnull()
        .any(dim=["latitude", "longitude", "step", "member"])
        .compute(),
        drop=True,
    ).values
    forecast = forecast.sel(isobaricInhPa=full_levels)
    forecast_ifs = forecast_ifs.sel(isobaricInhPa=full_levels)
    forecast_unperturbed = forecast_unperturbed.sel(isobaricInhPa=full_levels)
    forecast_unperturbed_ifs = forecast_unperturbed_ifs.sel(
        isobaricInhPa=full_levels)
    ground_truth = ground_truth.sel(isobaricInhPa=full_levels)

    ground_truth = ground_truth[config["selected_vars"]]
    forecast = forecast[config["selected_vars"]]
    forecast_unperturbed = forecast_unperturbed[config["selected_vars"]]
    forecast_ifs = forecast_ifs[config["selected_vars"]]
    forecast_unperturbed_ifs = forecast_unperturbed_ifs[config["selected_vars"]]

    if debug_mode:
        max_members = 10
        max_time_steps = 5

        forecast = forecast.isel(member=slice(0, max_members))
        forecast_ifs = forecast_ifs.isel(member=slice(0, max_members))

        ground_truth = ground_truth.isel(step=slice(0, max_time_steps))
        forecast = forecast.isel(step=slice(0, max_time_steps))
        forecast_unperturbed = forecast_unperturbed.isel(
            step=slice(
                0,
                max_time_steps))
        forecast_ifs = forecast_ifs.isel(step=slice(0, max_time_steps))

    for ds_name in [
        "ground_truth",
        "forecast",
        "forecast_unperturbed",
        "forecast_ifs",
        "forecast_unperturbed_ifs",
    ]:
        ds = locals()[ds_name]
        print(f"{ds_name}: {ds.sizes}")

    return {
        "ground_truth": ground_truth,
        "forecast": forecast,
        "forecast_unperturbed": forecast_unperturbed,
        "forecast_ifs": forecast_ifs,
        "forecast_unperturbed_ifs": forecast_unperturbed_ifs,
    }


def calculate_stats(ground_truth, forecast,
                    forecast_unperturbed, crop_region):
    """
    Calculate the statistics for the evaluation.

    Args:
        ground_truth (xr.Dataset): The ground truth data.
        forecast (xr.Dataset): The forecast data.
        forecast_unperturbed (xr.Dataset): The unperturbed forecast data.

    Returns:
        dict: A dictionary containing the statistics.
    """
    fc_mean = forecast.mean(dim=["latitude", "longitude"])
    fc_mean_unperturbed = forecast_unperturbed.mean(
        dim=["latitude", "longitude"])
    gt_mean = ground_truth.mean(dim=["latitude", "longitude"])
    squared_diff = (forecast - ground_truth) ** 2
    squared_diff_mean = (forecast.mean(dim="member") - ground_truth) ** 2
    rmse_grid = np.sqrt(squared_diff.mean(dim="member")).drop_isel(step=0)
    rmse = np.sqrt(
        squared_diff.mean(
            dim=[
                "latitude",
                "longitude"])).drop_isel(
        step=0)
    rmse_mean = np.sqrt(
        squared_diff_mean.mean(
            dim=[
                "latitude",
                "longitude"])).drop_isel(
        step=0)

    squared_diff_unperturbed = (forecast_unperturbed - ground_truth) ** 2
    rmse_unperturbed = np.sqrt(
        squared_diff_unperturbed.mean(
            dim=[
                "latitude",
                "longitude"])).drop_isel(
        step=0)

    if crop_region == "europe":
        lat_min, lat_max = 25, 80

    ensemble_spread_grid = forecast.std(dim="member").drop_isel(step=0)
    spread_skill_ratio_grid = ensemble_spread_grid / rmse_grid
    spread_skill_ratio = spread_skill_ratio_grid.mean(
        dim=["latitude", "longitude"])
    ensemble_spread = ensemble_spread_grid.mean(dim=["latitude", "longitude"])

    def calculate_energy_spectra(data, lat_band=(30, 60)):
        lat_slice = slice(lat_band[0], lat_band[1])
        data_lat_band = data.sel(latitude=lat_slice)

        # Select the last time step
        data_last_step = data_lat_band.isel(step=-1)

        energy_spectra_members = []

        # Calculate energy spectra for each member
        for member in data_last_step.member:
            data_member = data_last_step.sel(member=member)

            # Calculate mean over latitude band
            data_mean = data_member.mean(dim='latitude')

            # Calculate FFT along longitude
            fft = np.fft.fft(data_mean.values, axis=-1)

            # Calculate power spectrum
            power_spectrum = np.abs(fft) ** 2

            # Calculate wavelengths
            n = data_mean.longitude.size
            wavelengths = (360 / np.fft.fftfreq(n)
                           [1:n // 2]) * 111  # Convert to km

            energy_spectra_members.append(xr.Dataset({
                'power': ('wavelength', power_spectrum[1:n // 2]),
                'wavelength': ('wavelength', wavelengths)
            }))

        # Concatenate energy spectra of all members along a new dimension
        energy_spectra_concat = xr.concat(energy_spectra_members, dim='member')

        return energy_spectra_concat

    energy_spectra = {}
    for var in forecast.data_vars:
        energy_spectra[var] = {
            'forecast': calculate_energy_spectra(
                forecast[var], lat_band=(
                    lat_min, lat_max)),
            'unperturbed': calculate_energy_spectra(
                forecast_unperturbed[var], lat_band=(
                    lat_min, lat_max)),
            'ground_truth': calculate_energy_spectra(
                ground_truth[var], lat_band=(
                    lat_min, lat_max)).squeeze()  # Remove singleton 'member' dimension
        }

    return {
        "fc_mean": fc_mean,
        "fc_mean_unperturbed": fc_mean_unperturbed,
        "gt_mean": gt_mean,
        "rmse": rmse,
        "rmse_mean": rmse_mean,
        "rmse_unperturbed": rmse_unperturbed,
        "spread_skill_ratio": spread_skill_ratio,
        "ensemble_spread": ensemble_spread,
        "energy_spectra": energy_spectra,
    }


def calculate_y_lims(vars_3d, vars_2d, forecast,
                     forecast_ifs, default_stats, ifs_stats):
    """
    Calculate the y-limits for the plots.

    Args:
        vars_3d (list): The 3D variables.
        vars_2d (list): The 2D variables.
        forecast (xr.Dataset): The forecast data.
        forecast_ifs (xr.Dataset): The IFS forecast data.
        default_stats (dict): The default statistics.
        ifs_stats (dict): The IFS statistics.

    Returns:
        dict: A dictionary containing the y-limits.
    """
    y_lims_rmse = {}
    y_lims_spread_skill_ratio = {}
    y_lims_timeseries = {}
    y_lims_energy_spectra = {}

    for variable in vars_3d + vars_2d:
        is_3d = variable in vars_3d
        levels = forecast.isobaricInhPa.values if is_3d else [None]

        for level in levels:
            sel_kwargs = {'isobaricInhPa': level} if is_3d else {}

            def get_stat(stat_dict, stat_name):
                return stat_dict[stat_name][variable].sel(
                    **sel_kwargs) if is_3d else stat_dict[stat_name][variable]

            rmse_min = min(
                get_stat(
                    default_stats, 'rmse').min().values, get_stat(
                    ifs_stats, 'rmse').min().values)
            rmse_max = max(
                get_stat(
                    default_stats, 'rmse').max().values, get_stat(
                    ifs_stats, 'rmse').max().values)
            spread_skill_ratio_min = min(
                get_stat(
                    default_stats, 'spread_skill_ratio').min().values, get_stat(
                    ifs_stats, 'spread_skill_ratio').min().values)
            spread_skill_ratio_max = max(
                get_stat(
                    default_stats, 'spread_skill_ratio').max().values, get_stat(
                    ifs_stats, 'spread_skill_ratio').max().values)
            ensemble_spread_min = min(
                get_stat(
                    default_stats, 'ensemble_spread').min().values, get_stat(
                    ifs_stats, 'ensemble_spread').min().values)
            ensemble_spread_max = max(
                get_stat(
                    default_stats, 'ensemble_spread').max().values, get_stat(
                    ifs_stats, 'ensemble_spread').max().values)
            timeseries_min = min(
                get_stat(
                    default_stats, 'fc_mean').min().values, get_stat(
                    ifs_stats, 'fc_mean').min().values)
            timeseries_max = max(
                get_stat(
                    default_stats, 'fc_mean').max().values, get_stat(
                    ifs_stats, 'fc_mean').max().values)

            y_lims_rmse[(variable, level)] = (rmse_min, rmse_max)
            y_lims_spread_skill_ratio[(variable, level)] = ((
                spread_skill_ratio_min, spread_skill_ratio_max),
                (ensemble_spread_min, ensemble_spread_max))
            y_lims_timeseries[(variable, level)] = (
                timeseries_min, timeseries_max)

        energy_spectra_min = min(
            default_stats["energy_spectra"][variable]
            ["forecast"].power.min().values,
            ifs_stats["energy_spectra"][variable]
            ["forecast"].power.min().values)
        energy_spectra_max = max(
            default_stats["energy_spectra"][variable]
            ["forecast"].power.max().values,
            ifs_stats["energy_spectra"][variable]
            ["forecast"].power.max().values)
        energy_level = None if variable in vars_2d else forecast.isobaricInhPa.values[0]
        y_lims_energy_spectra[(variable, energy_level)] = (
            energy_spectra_min, energy_spectra_max)

    print("Y-limits calculated")

    return {
        "y_lims_rmse": y_lims_rmse,
        "y_lims_spread_skill_ratio": y_lims_spread_skill_ratio,
        "y_lims_timeseries": y_lims_timeseries,
        "y_lims_energy_spectra": y_lims_energy_spectra,
    }


def prepare_plot_args(
    data,
    stats,
    vars_3d,
    vars_2d,
    y_lims_rmse,
    y_lims_spread_skill_ratio,
    y_lims_timeseries,
    y_lims_energy_spectra,
    use_ifs=False,
    path_out=".",
    model_name="",
):
    """
    Prepare the arguments for the plotting functions.

    Args:
        data (dict): A dictionary containing the data.
        stats (dict): A dictionary containing the statistics.
        vars_3d (list): The 3D variables.
        vars_2d (list): The 2D variables.
        y_lims_rmse (dict): The y-limits for the RMSE plots.
        y_lims_spread_skill_ratio (dict): The y-limits for the spread-skill ratio plots.
        y_lims_timeseries (dict): The y-limits for the timeseries plots.
        y_lims_energy_spectra (dict): The y-limits for the energy spectra plots.
        use_ifs (bool): Whether to use the IFS data.
        path_out (str): The path to the output directory.
        model_name (str): The name of the model.

    Returns:
        dict: A dictionary containing the plotting arguments.
    """
    forecast_key = "forecast_ifs" if use_ifs else "forecast"
    rank_histogram_args = []
    energy_spectra_args = []
    rmse_args = []
    spread_skill_ratio_args = []
    timeseries_fc_gt_args = []

    variables = [
        (vars_3d, data[forecast_key][var].coords["isobaricInhPa"].values, True)
        for var in vars_3d
    ] + [(vars_2d, [None], False)]

    for variable, levels, is_3d in variables:
        for var in variable:
            for level in levels:
                rank_histogram_args.append(
                    (
                        var,
                        data[forecast_key],
                        data["ground_truth"],
                        path_out,
                        config["color_palette"],
                        model_name,
                        level,
                    )
                )
                energy_spectra_args.append(
                    (
                        var,
                        stats["energy_spectra"][var]["forecast"],
                        stats["energy_spectra"][var]["unperturbed"],
                        stats["energy_spectra"][var]["ground_truth"],
                        alpha_value,
                        path_out,
                        config["color_palette"],
                        model_name,
                        level,
                        y_lims_energy_spectra[(var, level)]
                        if is_3d
                        else y_lims_energy_spectra[(var, None)],
                    )
                )
                rmse_args.append(
                    (
                        var,
                        stats["rmse"],
                        stats["rmse_mean"],
                        stats["rmse_unperturbed"],
                        alpha_value,
                        path_out,
                        config["color_palette"],
                        model_name,
                        level,
                        y_lims_rmse[(var, level)],
                    )
                )
                spread_skill_ratio_args.append(
                    (
                        var,
                        stats["spread_skill_ratio"],
                        stats["ensemble_spread"],
                        path_out,
                        config["color_palette"],
                        model_name,
                        level,
                        y_lims_spread_skill_ratio[(var, level)][0],
                        y_lims_spread_skill_ratio[(var, level)][1],
                    )
                )
                timeseries_fc_gt_args.append(
                    (
                        var,
                        stats["gt_mean"],
                        stats["fc_mean"],
                        stats["fc_mean_unperturbed"],
                        data["ground_truth"],
                        alpha_value,
                        path_out,
                        config["color_palette"],
                        model_name,
                        level,
                        y_lims_timeseries[(var, level)],
                    )
                )

    print("Plotting arguments prepared")

    return {
        "rank_histogram": rank_histogram_args,
        "energy_spectra": energy_spectra_args,
        "rmse": rmse_args,
        "spread_skill_ratio": spread_skill_ratio_args,
        "timeseries_fc_gt": timeseries_fc_gt_args,
    }


def plot_rank_histogram(
        variable, forecast, ground_truth, path_out, color_palette, model_name,
        level=None):
    """
    Plot the rank histogram.

    Args:
        variable (str): The variable to plot.
        forecast (xr.Dataset): The forecast data.
        ground_truth (xr.Dataset): The ground truth data.
        path_out (str): The path to the output directory.
        color_palette (list): The color palette.
        model_name (str): The name of the model.
        level (int): The level to plot.
    """
    print(f"Creating rank histogram for variable: {variable}, level: {level}")

    forecast_var = (
        forecast[variable].sel(isobaricInhPa=level)
        if level else forecast[variable])
    ground_truth_var = (
        ground_truth[variable].sel(isobaricInhPa=level)
        if level
        else ground_truth[variable]
    )

    ground_truth_var = (
        ground_truth_var.expand_dims("member")
        .drop_vars(["number", "time"])
        .assign_coords(member=[0])
    )

    combined = xr.concat([forecast_var, ground_truth_var], dim="member")

    print(combined.member.values, flush=True)

    # Create a random subsample
    combined_stacked = combined.stack(z=("step", "latitude", "longitude"))
    sample_size = min(config["sample_size"], combined_stacked.z.size)
    indices = np.sort(
        np.random.choice(
            combined_stacked.z.size,
            size=sample_size,
            replace=False))
    combined_sample = combined_stacked.isel(z=indices)

    ranks = combined_sample.chunk(dict(member=-1)).rank("member")

    # Get the rank of the ground truth
    gt_rank = ranks.sel(member=0)
    unique_ranks, rank_counts = np.unique(gt_rank.values, return_counts=True)
    rank_counts = dict(zip(unique_ranks, rank_counts))

    # ---------------------------------------------------
    values = combined_sample.values.flatten()
    ground_truth_value = values[-1]  # Assuming ground truth is the last member

    plt.figure(figsize=(10, 6))
    plt.hist(values[:-1], bins=50, alpha=0.7, label='Ensemble Members')
    plt.axvline(
        ground_truth_value,
        color='r',
        linestyle='dashed',
        linewidth=2,
        label='Ground Truth')
    plt.title(f'Distribution of Values for {variable}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'value_distribution_{variable}.png')
    plt.close()

    # Plot the rank histogram
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar(
        list(rank_counts.keys()),
        list(rank_counts.values()),
        color=color_palette[4],
        edgecolor=color_palette[5],
    )
    ax.set_title(
        f"Rank Histogram for {variable}{' at level ' + str(level) if level else ''}"
    )
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    plt.savefig(
        os.path.join(
            path_out,
            f"rank_histogram_{variable}{'_' + str(level) if level else ''}.png",
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_energy_spectra(
    variable,
    energy_spectra_forecast,
    energy_spectra_unperturbed,
    energy_spectra_ground_truth,
    alpha_value,
    path_out,
    color_palette,
    model_name,
    level=None,
    y_lims=None,
):
    """
    Plot the energy spectra for each member and the mean of all members.
    """
    print(f"Plotting energy spectra for variable: {variable}")
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot each member's energy spectra
    for member in energy_spectra_forecast.member:
        ax.loglog(
            energy_spectra_forecast.wavelength,
            energy_spectra_forecast.sel(member=member).power,
            color=color_palette[1],
            alpha=alpha_value,
            label=f"{model_name} Member" if member == 0 else None
        )

    # Calculate and plot the mean energy spectra
    mean_energy_spectra = energy_spectra_forecast.mean(dim="member")
    ax.loglog(
        mean_energy_spectra.wavelength,
        mean_energy_spectra.power,
        color=color_palette[2],
        label=f"{model_name} Mean"
    )

    # Plot the unperturbed and ground truth energy spectra
    ax.loglog(
        energy_spectra_unperturbed.wavelength,
        energy_spectra_unperturbed.power,
        color=color_palette[3],
        label=f"{model_name} Unperturbed",
        linestyle="--"
    )
    ax.loglog(
        energy_spectra_ground_truth.wavelength,
        energy_spectra_ground_truth.power,
        color=color_palette[0],
        label="Ground Truth: ERA5",
        linestyle=":"
    )

    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power")
    ax.set_title(f"Energy Spectra for {variable}")
    ax.legend()

    if y_lims is not None:
        ax.set_ylim(y_lims)

    plt.savefig(
        os.path.join(path_out, f"energy_spectra_comparison_{variable}.png"),
        dpi=300
    )
    plt.close(fig)


def plot_rmse(
    variable,
    rmse,
    rmse_mean,
    rmse_unperturbed,
    alpha_value,
    path_out,
    color_palette,
    model_name,
    level=None,
    y_lims=None,
):
    """
    Plot the RMSE.

    Args:
        variable (str): The variable to plot.
        rmse (xr.Dataset): The RMSE data.
        rmse_mean (xr.Dataset): The mean RMSE data.
        rmse_unperturbed (xr.Dataset): The unperturbed RMSE data.
        alpha_value (float): The alpha value for the plot.
        path_out (str): The path to the output directory.
        color_palette (list): The color palette.
        model_name (str): The name of the model.
        level (int): The level to plot.
        y_lims (tuple): The y-limits for the plot.
    """
    print(f"Creating RMSE plots for variable: {variable}, level: {level}")
    fig, ax = plt.subplots(figsize=(12, 9))

    # Select the level if provided
    if level is not None:
        rmse = rmse.sel(isobaricInhPa=level)
        rmse_unperturbed = rmse_unperturbed.sel(isobaricInhPa=level)

    for i, member in enumerate(rmse.member):
        rmse_member = rmse[variable].sel(member=member)
        if i == 0:  # Add label only for the first member
            rmse_member.plot(
                ax=ax,
                color=color_palette[1],
                alpha=alpha_value,
                label=f"{model_name} Members",
            )
        else:
            rmse_member.plot(ax=ax, color=color_palette[1], alpha=alpha_value)

    rmse_mean[variable].plot(
        ax=ax, color=color_palette[2], label=f"{model_name} mean"
    )

    rmse_unperturbed[variable].plot(
        ax=ax, color=color_palette[3], label=f"{model_name} Unperturbed"
    )

    if y_lims is not None:
        (ymin, ymax) = y_lims
        ax.set_ylim(ymin.item() * 0.9, ymax.item() * 1.1)

    ax.set_ylabel(f"RMSE ({variable})")
    ax.set_title(
        f"Root Mean Square Error: {variable}{' at level ' + str(level) if level is not None else ''}"
    )
    ax.legend()
    plt.savefig(
        os.path.join(
            path_out,
            f"rmse_{variable}{'_' + str(level) if level is not None else ''}.png",
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_spread_skill_ratio(
    variable,
    spread_skill_ratio,
    ensemble_spread,
    path_out,
    color_palette,
    model_name,
    level=None,
    y_lims1=None,
    y_lims2=None,
):
    """
    Plot the spread-skill ratio.

    Args:
        variable (str): The variable to plot.
        spread_skill_ratio (xr.Dataset): The spread-skill ratio data.
        ensemble_spread (xr.Dataset): The ensemble spread data.
        path_out (str): The path to the output directory.
        color_palette (list): The color palette.
        model_name (str): The name of the model.
        level (int): The level to plot.
        y_lims1 (tuple): The y-limits for the spread-skill ratio plot.
        y_lims2 (tuple): The y-limits for the ensemble spread plot.
    """
    print(
        f"Creating spread-skill ratio plots for variable: {variable}, level: {level}")
    if level is not None:
        data_array = spread_skill_ratio[variable].sel(isobaricInhPa=level)
        ensemble_spread_data = ensemble_spread[variable].sel(
            isobaricInhPa=level)
    else:
        data_array = spread_skill_ratio[variable]
        ensemble_spread_data = ensemble_spread[variable]

    fig, ax = plt.subplots(figsize=(12, 9))
    data_array.plot(ax=ax, color=color_palette[5])

    ax2 = ax.twinx()
    ensemble_spread_data.plot(ax=ax2, color=color_palette[4])

    if y_lims1 is not None:
        (ymin, ymax) = y_lims1
        ax.set_ylim(0, ymax.item() * 1.1)

    ax.set_xlabel("")
    ax.set_ylabel("Spread-skill ratio")
    ax.set_title(
        f"{model_name} for {variable}{' at level ' + str(level) if level is not None else ''}"
    )

    if y_lims2 is not None:
        (ymin, ymax) = y_lims2
        ax2.set_ylim(ymin.item() * 0.9, ymax.item() * 1.1)

    ax2.set_xlabel("")
    ax2.set_ylabel("Ensemble Spread", color=color_palette[4])
    ax2.tick_params(axis="y", colors=color_palette[4])
    ax2.set_title("")
    plt.savefig(
        os.path.join(
            path_out,
            f"spread_skill_ratio_{variable}{'_' + str(level) if level is not None else ''}.png",
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_timeseries_fc_gt(
    variable,
    gt_mean,
    fc_mean,
    fc_mean_unperturbed,
    ground_truth,
    alpha_value,
    path_out,
    color_palette,
    model_name,
    level=None,
    y_lims=None,
):
    print(
        f"Creating timeseries plots for variable: {variable}, level: {level}")
    fig, ax = plt.subplots(figsize=(12, 9))

    # Select the level if provided
    gt_mean_var = (
        gt_mean[variable].sel(isobaricInhPa=level)
        if level is not None
        else gt_mean[variable]
    )
    fc_mean_var = (
        fc_mean[variable].sel(isobaricInhPa=level)
        if level is not None
        else fc_mean[variable]
    )
    fc_mean_unperturbed_var = (
        fc_mean_unperturbed[variable].sel(isobaricInhPa=level)
        if level is not None
        else fc_mean_unperturbed[variable]
    )

    gt_mean_var.plot(ax=ax, color=color_palette[0], label="Ground Truth: ERA5")

    for i, member in enumerate(fc_mean.member):
        fc_mean_member = fc_mean_var.sel(member=member)
        if i == 0:  # Add label only for the first member
            fc_mean_member.plot(
                ax=ax,
                color=color_palette[1],
                alpha=alpha_value,
                label=f"{model_name} Members",
            )
        else:
            fc_mean_member.plot(
                ax=ax, color=color_palette[1],
                alpha=alpha_value)

    fc_mean_var.mean(dim="member").plot(
        ax=ax, color=color_palette[2], label=f"{model_name} mean"
    )
    fc_mean_unperturbed_var.plot(
        ax=ax, color=color_palette[3], label=f"{model_name} Unperturbed"
    )

    # Collect all values for the density plot at the latest time step
    fc_values_last_step = fc_mean_var.isel(step=-1).values

    if y_lims is not None:
        (ymin, ymax) = y_lims
        extent = ymax - ymin
        ax.set_ylim(
            ymin.item() -
            0.1 *
            extent.item(),
            ymax.item() +
            0.1 *
            extent.item())

    # Create an inset axis for the density plot
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", size=1.2, pad=0.1)

    # Plot the density distribution rotated (horizontal)
    density = sns.kdeplot(
        y=fc_values_last_step,
        ax=ax2,
        color=color_palette[1],
        bw_adjust=0.5
    )

    # Normalize the density values
    xy = density.get_lines()[0].get_xydata()
    xy[:, 1] /= np.trapz(xy[:, 1], xy[:, 0])
    density.get_lines()[0].set_ydata(xy[:, 1])

    ax2.set_ylim(ax.get_ylim())
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Density")

    # Add the map plot as an inset
    ax_inset = inset_axes(
        ax,
        width="15%",
        height="15%",
        loc='upper right',
        borderpad=2)
    _plot_map(variable, ground_truth, level=level, ax=ax_inset)

    ax.set_xlabel("")
    ax.set_ylabel(
        f"{variable}{' at level ' + str(level) if level is not None else ''}")
    ax.set_title(
        f"Comparison of Forecast vs Ground-Truth: {variable}{' at level ' + str(level) if level is not None else ''}"
    )
    ax.legend()
    plt.savefig(
        os.path.join(
            path_out,
            f"timeseries_fc_gt_{variable}{'_' + str(level) if level is not None else ''}.png",
        ),
        dpi=300,
    )
    plt.close(fig)


def _plot_map(variable, ground_truth, level=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={
                               'projection': ccrs.PlateCarree()})

    lat = ground_truth.latitude.values
    lon = ground_truth.longitude.values

    im = ax.pcolormesh(
        lon, lat,
        ground_truth[variable].isel(time=0, step=0).values,
        cmap="plasma", transform=ccrs.PlateCarree()
    )

    ax.set_title("Ground Truth t2m at surface, initial time")
    ax.coastlines()
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Temperature (K)')

    if ax is None:
        plt.tight_layout()
        plt.savefig("ground_truth_t2m_map.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    path_in = os.path.join(str(args.date_time), args.model_name)
    assert (
        args.members <= 50
    ), "The number of ensemble members must be less than or equal to 50 to plot IFS ENS"
    model_name = args.model_name.title()
    path_out = os.path.join(
        path_in,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}",
        args.crop_region,
        f"png_{args.model_name}")
    path_out_ifs = os.path.join(
        path_in,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}",
        args.crop_region,
        "png_ifs")
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(path_out_ifs, exist_ok=True)

    data = load_and_prepare_data(
        path_in,
        args.crop_region,
        debug_mode=args.debug)

    print("data loaded", flush=True)
    default_stats = calculate_stats(
        data["ground_truth"],
        data["forecast"],
        data["forecast_unperturbed"],
        args.crop_region)
    print("stats calculated", flush=True)
    ifs_stats = calculate_stats(
        data["ground_truth"],
        data["forecast_ifs"],
        data["forecast_unperturbed_ifs"],
        args.crop_region)
    print("ifs stats calculated", flush=True)

    alpha_value = 1 / data["forecast"].member.size ** (5 / 8)
    variables = list(data["forecast"].data_vars)
    vars_3d = [
        var for var in variables
        if "isobaricInhPa" in data["forecast"][var].dims]
    vars_2d = [
        var for var in variables
        if "isobaricInhPa" not in data["forecast"][var].dims]

    y_lims = calculate_y_lims(
        vars_3d,
        vars_2d,
        data["forecast"],
        data["forecast_ifs"],
        default_stats,
        ifs_stats,
    )

    default_plot_args = prepare_plot_args(
        data,
        default_stats,
        vars_3d,
        vars_2d,
        **y_lims,
        use_ifs=False,
        path_out=path_out,
        model_name=model_name,
    )

    ifs_plot_args = prepare_plot_args(
        data,
        ifs_stats,
        vars_3d,
        vars_2d,
        **y_lims,
        use_ifs=True,
        path_out=path_out_ifs,
        model_name="IFS ENS",
    )

    for plot_args in [default_plot_args, ifs_plot_args]:
        for args in plot_args["rank_histogram"]:
            plot_rank_histogram(*args)
        for args in plot_args["rmse"]:
            plot_rmse(*args)
        for args in plot_args["spread_skill_ratio"]:
            plot_spread_skill_ratio(*args)
        for args in plot_args["energy_spectra"]:
            plot_energy_spectra(*args)
        for args in plot_args["timeseries_fc_gt"]:
            plot_timeseries_fc_gt(*args)
