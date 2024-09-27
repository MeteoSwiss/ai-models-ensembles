import argparse
import os

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

args = parser.parse_args()

config = {
    "color_palette": sns.color_palette(
        ["#f75b78", "#6495ed", "#0e2d75", "#f9c740", "#45b7aa", "#353434"]
    ),
    "sample_size": 100000,
    "radial_bins": 30,
    "coarsen": 10,
    "selected_vars": ["t2m", "u10", "v10", "msl"]
}


def load_and_prepare_data(path_in, crop_region):
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

    if crop_region == "europe":
        lat_min, lat_max = 35, 70
        lon_min, lon_max = -10, 40

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

        # Create the plot
        print(ground_truth["t2m"].isel(time=0))

        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={
            'projection': ccrs.PlateCarree()})

        lat = ground_truth.latitude.values
        lon = ground_truth.longitude.values

        im = ax.pcolormesh(
            lon, lat,
            ground_truth["t2m"].isel(time=0, step=0).values,
            cmap="plasma", transform=ccrs.PlateCarree()
        )

        ax.set_title("Ground Truth t2m at surface, initial time")
        ax.coastlines()
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Temperature (K)')

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig("ground_truth_t2m_map.png",
                    dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

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


def calculate_stats(ground_truth, forecast, forecast_unperturbed):
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
    squared_diff_median = (forecast.median(dim="member") - ground_truth) ** 2
    rmse_grid = np.sqrt(squared_diff.mean(dim="member")).drop_isel(step=0)
    rmse = np.sqrt(
        squared_diff.mean(
            dim=[
                "latitude",
                "longitude"])).drop_isel(
        step=0)
    rmse_median = np.sqrt(
        squared_diff_median.mean(dim=["latitude", "longitude"])
    ).drop_isel(step=0)

    squared_diff_unperturbed = (forecast_unperturbed - ground_truth) ** 2
    rmse_unperturbed = np.sqrt(
        squared_diff_unperturbed.mean(dim=["latitude", "longitude"])
    ).drop_isel(step=0)

    ensemble_spread_grid = forecast.std(dim="member").drop_isel(step=0)
    spread_skill_ratio_grid = ensemble_spread_grid / rmse_grid
    spread_skill_ratio = spread_skill_ratio_grid.mean(
        dim=["latitude", "longitude"])
    ensemble_spread = ensemble_spread_grid.mean(dim=["latitude", "longitude"])

    def calculate_psd(field):
        fft_result = np.fft.fft2(field)
        fft_shifted = np.fft.fftshift(fft_result)
        psd = np.abs(fft_shifted) ** 2
        return psd

    freq_x = np.fft.fftfreq(forecast.sizes["longitude"], d=1.0)
    freq_y = np.fft.fftfreq(forecast.sizes["latitude"], d=1.0)
    freq_x = np.fft.fftshift(freq_x)
    freq_y = np.fft.fftshift(freq_y)
    coarsened_freq_x = freq_x[:: config["coarsen"]]
    coarsened_freq_y = freq_y[:: config["coarsen"]]
    freq_r = np.sqrt(coarsened_freq_x[None, :]
                     ** 2 + coarsened_freq_y[:, None] ** 2)

    def calculate_radial_psd(data, freq_r, bins=config["radial_bins"]):
        radial_psd = {}
        if "isobaricInhPa" in data.dims:
            for level in data.isobaricInhPa.values:
                radial_psd[level] = {}
                for member in data.member.values:
                    field = (
                        data.sel(isobaricInhPa=level, member=member)
                        .coarsen(
                            latitude=config["coarsen"],
                            longitude=config["coarsen"],
                            boundary="pad",
                        )
                        .mean()
                        .values
                    )
                    weights = calculate_psd(field)
                    radial_psd[level][member] = np.histogram(
                        freq_r, bins=bins, weights=weights
                    )[0]
        else:
            for member in data.member.values:
                field = (
                    data.sel(member=member)
                    .coarsen(
                        latitude=config["coarsen"],
                        longitude=config["coarsen"],
                        boundary="pad",
                    )
                    .mean()
                    .values
                )
                weights = calculate_psd(field)
                radial_psd[member] = np.histogram(
                    freq_r, bins=bins, weights=weights)[0]
        return radial_psd

    radial_psd_forecast = {}
    for var in forecast.data_vars:
        radial_psd_forecast[var] = calculate_radial_psd(
            forecast[var].mean(dim=["step"]), freq_r
        )

    radial_psd_unperturbed = {}
    for var in forecast_unperturbed.data_vars:
        radial_psd_unperturbed_var = calculate_radial_psd(
            forecast_unperturbed[var].mean(dim=["step"]).expand_dims(
                {"member": 1}),
            freq_r,)
        radial_psd_unperturbed[var] = radial_psd_unperturbed_var[0]

    radial_psd_ground_truth = {}
    for var in ground_truth.data_vars:
        radial_psd_ground_truth_var = calculate_radial_psd(
            ground_truth[var].mean(dim=["step"]).expand_dims({"member": 1}), freq_r
        )
        radial_psd_ground_truth[var] = radial_psd_ground_truth_var[0]

    return {
        "fc_mean": fc_mean,
        "fc_mean_unperturbed": fc_mean_unperturbed,
        "gt_mean": gt_mean,
        "rmse": rmse,
        "rmse_median": rmse_median,
        "rmse_unperturbed": rmse_unperturbed,
        "spread_skill_ratio": spread_skill_ratio,
        "ensemble_spread": ensemble_spread,
        "radial_psd_forecast": radial_psd_forecast,
        "radial_psd_unperturbed": radial_psd_unperturbed,
        "radial_psd_ground_truth": radial_psd_ground_truth,
    }


def calculate_y_lims(
    vars_3d, vars_2d, forecast, forecast_ifs, default_stats, ifs_stats
):
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

    for variable in vars_3d:
        for level in forecast.isobaricInhPa.values:
            rmse_min = min(
                default_stats["rmse"][variable].sel(
                    isobaricInhPa=level).min().values, ifs_stats["rmse"][variable].sel(
                    isobaricInhPa=level).min().values, )
            spread_skill_ratio_min = min(
                default_stats["spread_skill_ratio"][variable]
                .sel(isobaricInhPa=level)
                .min()
                .values,
                ifs_stats["spread_skill_ratio"][variable]
                .sel(isobaricInhPa=level)
                .min()
                .values,
            )
            ensemble_spread_min = min(
                default_stats["ensemble_spread"][variable]
                .sel(isobaricInhPa=level)
                .min()
                .values,
                ifs_stats["ensemble_spread"][variable]
                .sel(isobaricInhPa=level)
                .min()
                .values,
            )
            timeseries_min = min(
                default_stats["fc_mean"][variable] .sel(
                    isobaricInhPa=level) .min() .values, ifs_stats["fc_mean"][variable].sel(
                    isobaricInhPa=level).min().values, )
            rmse_max = max(
                default_stats["rmse"][variable].sel(
                    isobaricInhPa=level).max().values, ifs_stats["rmse"][variable].sel(
                    isobaricInhPa=level).max().values, )
            spread_skill_ratio_max = max(
                default_stats["spread_skill_ratio"][variable]
                .sel(isobaricInhPa=level)
                .max()
                .values,
                ifs_stats["spread_skill_ratio"][variable]
                .sel(isobaricInhPa=level)
                .max()
                .values,
            )
            ensemble_spread_max = max(
                default_stats["ensemble_spread"][variable]
                .sel(isobaricInhPa=level)
                .max()
                .values,
                ifs_stats["ensemble_spread"][variable]
                .sel(isobaricInhPa=level)
                .max()
                .values,
            )
            timeseries_max = max(
                default_stats["fc_mean"][variable] .sel(
                    isobaricInhPa=level) .max() .values, ifs_stats["fc_mean"][variable].sel(
                    isobaricInhPa=level).max().values, )
            energy_spectra_min = min(
                min(
                    min(arr[1:])
                    for arr in default_stats["radial_psd_forecast"][variable][
                        level
                    ].values()
                ),
                min(
                    min(arr[1:])
                    for arr in ifs_stats["radial_psd_forecast"][variable][
                        level
                    ].values()
                ),
            )
            energy_spectra_max = max(
                max(
                    max(arr[1:])
                    for arr in default_stats["radial_psd_forecast"][variable][
                        level
                    ].values()
                ),
                max(
                    max(arr[1:])
                    for arr in ifs_stats["radial_psd_forecast"][variable][
                        level
                    ].values()
                ),
            )
            y_lims_rmse[(variable, level)] = (rmse_min, rmse_max)
            y_lims_spread_skill_ratio[(variable, level)] = ((
                spread_skill_ratio_min,
                spread_skill_ratio_max,
            ), (ensemble_spread_min, ensemble_spread_max))
            y_lims_timeseries[(variable, level)] = (
                timeseries_min, timeseries_max)
            y_lims_energy_spectra[(variable, level)] = (
                energy_spectra_min,
                energy_spectra_max,
            )

    for variable in vars_2d:
        rmse_min = min(
            default_stats["rmse"][variable].min().values,
            ifs_stats["rmse"][variable].min().values,
        )
        spread_skill_ratio_min = min(
            default_stats["spread_skill_ratio"][variable].min().values,
            ifs_stats["spread_skill_ratio"][variable].min().values,
        )
        ensemble_spread_min = min(
            default_stats["ensemble_spread"][variable].min().values,
            ifs_stats["ensemble_spread"][variable].min().values,
        )
        timeseries_min = min(
            default_stats["fc_mean"][variable].min().values,
            ifs_stats["fc_mean"][variable].min().values,
        )
        rmse_max = max(
            default_stats["rmse"][variable].max().values,
            ifs_stats["rmse"][variable].max().values,
        )
        spread_skill_ratio_max = max(
            default_stats["spread_skill_ratio"][variable].max().values,
            ifs_stats["spread_skill_ratio"][variable].max().values,
        )
        ensemble_spread_max = max(
            default_stats["ensemble_spread"][variable].max().values,
            ifs_stats["ensemble_spread"][variable].max().values,
        )
        timeseries_max = max(
            default_stats["fc_mean"][variable].max().values,
            ifs_stats["fc_mean"][variable].max().values,
        )
        energy_spectra_min = min(
            min(
                min(arr[1:])
                for arr in default_stats
                ["radial_psd_forecast"]
                [variable].values()),
            min(
                min(arr[1:])
                for arr in ifs_stats
                ["radial_psd_forecast"]
                [variable].values()),)
        energy_spectra_max = max(
            max(
                max(arr[1:])
                for arr in default_stats
                ["radial_psd_forecast"]
                [variable].values()),
            max(
                max(arr[1:])
                for arr in ifs_stats
                ["radial_psd_forecast"]
                [variable].values()),)
        level = None
        y_lims_rmse[(variable, level)] = (rmse_min, rmse_max)
        y_lims_spread_skill_ratio[(variable, level)] = (
            (spread_skill_ratio_min, spread_skill_ratio_max),
            (ensemble_spread_min, ensemble_spread_max),
        )
        y_lims_timeseries[(variable, level)] = (
            timeseries_min,
            timeseries_max,
        )
        y_lims_energy_spectra[(variable, level)] = (
            energy_spectra_min,
            energy_spectra_max,
        )

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
                        stats["radial_psd_forecast"][var][level]
                        if is_3d
                        else stats["radial_psd_forecast"][var],
                        stats["radial_psd_unperturbed"][var][level]
                        if is_3d
                        else stats["radial_psd_unperturbed"][var],
                        stats["radial_psd_ground_truth"][var][level]
                        if is_3d
                        else stats["radial_psd_ground_truth"][var],
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
                        stats["rmse_median"],
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

    # Create a random subsample
    sample_size = config["sample_size"]
    combined_stacked = combined.stack(z=("step", "latitude", "longitude"))
    indices = np.sort(
        np.random.choice(
            combined_stacked.z.size,
            size=sample_size,
            replace=False))
    combined_sample = combined_stacked.isel(z=indices)

    ranks = combined_sample.chunk(dict(member=-1)).rank("member")

    # Get the rank of the ground truth
    gt_rank = ranks.isel(member=-1)
    unique_ranks, rank_counts = np.unique(gt_rank.values, return_counts=True)
    rank_counts = dict(zip(unique_ranks, rank_counts))

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
    radial_psd_forecast,
    radial_psd_unperturbed,
    radial_psd_ground_truth,
    alpha_value,
    path_out,
    color_palette,
    model_name,
    level=None,
    y_lims=None,
):
    """
    Plot the energy spectra.
    
    Args:
        variable (str): The variable to plot.
        radial_psd_forecast (dict): The radial PSD forecast data.
        radial_psd_unperturbed (dict): The radial PSD unperturbed data.
        radial_psd_ground_truth (dict): The radial PSD ground truth data.
        alpha_value (float): The alpha value for the plot.
        path_out (str): The path to the output directory.
        color_palette (list): The color palette.
        model_name (str): The name of the model.
        level (int): The level to plot.
        y_lims (tuple): The y-limits for the plot.
    """
    print(f"Plotting energy spectra for variable: {variable}, level: {level}")
    freq_x = np.fft.fftfreq(
        next(
            iter(
                radial_psd_forecast.values())).size,
        d=1.0)
    freq_x = np.fft.fftshift(freq_x)
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, (member, psd) in enumerate(
        radial_psd_forecast[level].items()
        if level is not None
        else radial_psd_forecast.items()
    ):
        ax.loglog(
            freq_x,
            psd,
            color=color_palette[1],
            alpha=alpha_value,
            label=f"{model_name} Members" if i == 0 else "",
        )

    # Calculate and plot the median PSD
    median_psd = np.median(
        list(radial_psd_forecast[level].values())
        if level is not None
        else list(radial_psd_forecast.values()),
        axis=0,
    )
    ax.loglog(
        freq_x,
        median_psd,
        color=color_palette[2],
        label=f"{model_name} Median",
        linestyle="--",
    )

    ax.loglog(
        freq_x, radial_psd_unperturbed[level]
        if level is not None else radial_psd_unperturbed,
        label=f"{model_name} Unperturbed", linestyle="--",
        color=color_palette[3],)
    ax.loglog(
        freq_x,
        radial_psd_ground_truth[level]
        if level is not None
        else radial_psd_ground_truth,
        color=color_palette[0],
        label="Ground Truth: ERA5",
        linestyle=":",
    )

    major_xticks = ax.get_xticks(minor=False)
    minor_xticks = ax.get_xticks(minor=True)
    ax.xaxis.set_major_locator(FixedLocator(major_xticks))
    ax.xaxis.set_minor_locator(FixedLocator(minor_xticks))
    ax.set_xticklabels(["{:.0e}".format(tick)
                       for tick in major_xticks], minor=False)
    ax.set_xticklabels(["{:.0e}".format(tick)
                       for tick in minor_xticks], minor=True)
    for label in ax.get_xticklabels(which="both"):
        label.set_rotation(45)
    if y_lims is not None:
        (ymin, ymax) = y_lims
        ax.set_ylim(ymin.item(), ymax.item() * 0.1)

    ax.set_title(f"Energy Spectra for {variable}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power Spectral Density")
    ax.legend()
    plt.savefig(
        os.path.join(
            path_out,
            f"energy_spectra_comparison_{variable}.png"),
        dpi=300)
    plt.close(fig)


def plot_rmse(
    variable,
    rmse,
    rmse_median,
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
        rmse_median (xr.Dataset): The median RMSE data.
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

    rmse_median[variable].plot(
        ax=ax, color=color_palette[2], label=f"{model_name} Median"
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
    alpha_value,
    path_out,
    color_palette,
    model_name,
    level=None,
    y_lims=None,
):
    """
    Plot the timeseries forecast vs ground truth.
    
    Args:
        variable (str): The variable to plot.
        gt_mean (xr.Dataset): The ground truth data.
        fc_mean (xr.Dataset): The forecast data.
        fc_mean_unperturbed (xr.Dataset): The unperturbed forecast data.
        alpha_value (float): The alpha value for the plot.
        path_out (str): The path to the output directory.
        color_palette (list): The color palette.
        model_name (str): The name of the model.
        level (int): The level to plot.
        y_lims (tuple): The y-limits for the plot.
    """
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

    fc_mean_var.median(dim="member").plot(
        ax=ax, color=color_palette[2], label=f"{model_name} Median"
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
            ymin.item() - 0.1 * extent.item(),
            ymax.item() + 0.1 * extent.item())

    # Create an inset axis for the density plot
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", size=1.2, pad=0.1)

    # Plot the density distribution rotated (horizontal)
    sns.kdeplot(
        y=fc_values_last_step,
        ax=ax2,
        color=color_palette[1],
        bw_adjust=0.5)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Density")

    ax.set_xlabel("")
    ax.set_ylabel(
        f"{variable}{' at level ' + str(level) if level is not None else ''}")
    ax.set_title(
        f"Forecast vs Ground-Truth Comparison: {variable}{' at level ' + str(level) if level is not None else ''}"
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

    data = load_and_prepare_data(path_in, args.crop_region)

    print("data loaded", flush=True)
    default_stats = calculate_stats(
        data["ground_truth"], data["forecast"], data["forecast_unperturbed"]
    )
    print("stats calculated", flush=True)
    ifs_stats = calculate_stats(
        data["ground_truth"],
        data["forecast_ifs"],
        data["forecast_unperturbed_ifs"])
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
