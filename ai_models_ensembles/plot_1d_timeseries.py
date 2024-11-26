import os

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde

from .preprocess_data import (
    calculate_stats,
    calculate_y_lims,
    load_and_prepare_data,
    parse_args,
)

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 15})
plt.ioff()


def _set_xticks(ax, data):
    ax.set_xticks(np.arange(0, (len(data.step) + 1) * 6, 24))
    ax.set_xticklabels(np.arange(0, (len(data.step) + 1) * 6, 24))


def prepare_plot_args(
    data,
    stats,
    vars_3d,
    vars_2d,
    config,
    y_lims_rmse,
    y_lims_spread_skill_ratio,
    y_lims_timeseries,
    y_lims_energy_spectra,
    use_ifs=False,
    path_out=".",
    model_name="",
    region="",
    date_time="",
):
    """
    Prepare the arguments for the plotting functions.

    Args:
        data (dict): A dictionary containing the data.
        stats (dict): A dictionary containing the statistics.
        vars_3d (list): The 3D variables.
        vars_2d (list): The 2D variables.
        config (dict): The configuration dictionary.
        y_lims_rmse (dict): The y-limits for the RMSE plots.
        y_lims_spread_skill_ratio (dict): The y-limits for the spread-skill ratio plots.
        y_lims_timeseries (dict): The y-limits for the timeseries plots.
        y_lims_energy_spectra (dict): The y-limits for the energy spectra plots.
        use_ifs (bool): Whether to use the IFS data.
        path_out (str): The path to the output directory.
        model_name (str): The name of the model.
        region (str): The cropped region
        date_time (str): The init date and time

    Returns:
        dict: A dictionary pf dictionaries containing the plotting arguments.
    """
    forecast_key = "forecast_ifs" if use_ifs else "forecast"
    error_map_args = []
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
                error_map_args.append(
                    {
                        "errors_mean": stats["rmse_mean"],
                        "errors_unperturbed": stats["rmse_unperturbed"],
                        "errors_members": stats["rmse"],
                        "path_out": path_out,
                        "level": level,
                    }
                )
                rank_histogram_args.append(
                    {
                        "variable": var,
                        "forecast": data[forecast_key],
                        "ground_truth": data["ground_truth"],
                        "path_out": path_out,
                        "color_palette": config["color_palette"],
                        "model_name": model_name,
                        "level": level,
                        "region": region,
                        "date_time": date_time,
                        "sample_size": config["sample_size"],
                    }
                )
                energy_spectra_args.append(
                    {
                        "variable": var,
                        "energy_spectra_forecast": stats["energy_spectra_forecast"],
                        "energy_spectra_unperturbed": stats[
                            "energy_spectra_unperturbed"
                        ],
                        "energy_spectra_ground_truth": stats[
                            "energy_spectra_ground_truth"
                        ],
                        "alpha_value": alpha_value,
                        "path_out": path_out,
                        "color_palette": config["color_palette"],
                        "model_name": model_name,
                        "level": level,
                        "y_lims": y_lims_energy_spectra[(var, level)]
                        if is_3d
                        else y_lims_energy_spectra[(var, None)],
                        "region": region,
                        "date_time": date_time,
                    }
                )
                rmse_args.append(
                    {
                        "variable": var,
                        "rmse": stats["rmse"],
                        "rmse_mean": stats["rmse_mean"],
                        "rmse_unperturbed": stats["rmse_unperturbed"],
                        "alpha_value": alpha_value,
                        "path_out": path_out,
                        "color_palette": config["color_palette"],
                        "model_name": model_name,
                        "level": level,
                        "y_lims": y_lims_rmse[(var, level)],
                        "region": region,
                        "date_time": date_time,
                    }
                )
                spread_skill_ratio_args.append(
                    {
                        "variable": var,
                        "sr_spread_skill_ratio": stats["sr_spread_skill_ratio"],
                        "sr_ensemble_spread": stats["sr_ensemble_spread"],
                        "path_out": path_out,
                        "color_palette": config["color_palette"],
                        "model_name": model_name,
                        "level": level,
                        "y_lims1": y_lims_spread_skill_ratio[(var, level)][0],
                        "y_lims2": y_lims_spread_skill_ratio[(var, level)][1],
                        "region": region,
                        "date_time": date_time,
                    }
                )
                timeseries_fc_gt_args.append(
                    {
                        "variable": var,
                        "gt_mean": stats["ts_gt_mean"],
                        "fc_mean": stats["ts_fc_mean"],
                        "fc_mean_unperturbed": stats["ts_fc_mean_unperturbed"],
                        "ground_truth": data["ground_truth"],
                        "alpha_value": alpha_value,
                        "path_out": path_out,
                        "color_palette": config["color_palette"],
                        "model_name": model_name,
                        "level": level,
                        "y_lims": y_lims_timeseries[(var, level)],
                        "region": region,
                        "date_time": date_time,
                    }
                )

    print("Plotting arguments prepared")

    return {
        "error_map": error_map_args,
        "rank_histogram": rank_histogram_args,
        "energy_spectra": energy_spectra_args,
        "rmse": rmse_args,
        "spread_skill_ratio": spread_skill_ratio_args,
        "timeseries_fc_gt": timeseries_fc_gt_args,
    }


def plot_error_map(errors_mean, errors_unperturbed, errors_members, path_out, level):
    """
    Plot a heatmap of errors of all forecast variables

    Args:
        errors: xarray.DataArray with dimensions (variable, step)
        errors_mean: xarray.DataArray with dimensions (variable, step)
        errors_unperturbed: xarray.DataArray with dimensions (variable, step)
        path_out: str, path to the output directory
        level: int, the level to plot

    """

    print("Creating scorecards")

    errors_mean = errors_mean.expand_dims("member").assign_coords(member=[9998])
    errors_unperturbed = errors_unperturbed.expand_dims("member").assign_coords(
        member=[9999]
    )
    errors_members = errors_members.sortby("member", ascending=True)
    errors_comb = xr.combine_by_coords(
        [errors_members, errors_mean, errors_unperturbed]
    )

    # TODO this should not be hardcoded and match the animation selection
    for mem_i in errors_comb.member.values[0:2]:
        if "isobaricInhPa" in errors_comb.dims:
            errors = errors_comb.sel(member=mem_i, isobaricInhPa=level)
        else:
            errors = errors_comb.sel(member=mem_i)
        member = (
            "mean"
            if mem_i == 9998
            else "unperturbed"
            if mem_i == 9999
            else f"member {mem_i}"
        )
        # Normalize all errors to [0,1] for color map
        max_errors = errors.max()
        min_errors = errors.min()
        errors_norm = (errors - min_errors) / (max_errors - min_errors)

        errors_norm = errors_norm.to_array()

        # Convert the DataArray to a NumPy array
        errors_norm_np = errors_norm.values
        errors_np = errors.to_array().values
        d_f, pred_steps = errors_np.shape

        fig, ax = plt.subplots(figsize=(15, 10))

        # Heatmap is projected across normalized values of ALL variables
        ax.imshow(
            errors_norm_np,
            cmap="OrRd",
            vmin=0,
            vmax=1.0,
            interpolation="none",
            aspect="auto",
        )

        # Add error values to the heatmap cells, rotated 90 degrees
        for (j, i), error in np.ndenumerate(errors_np):
            # Numbers > 9999 will be too large to fit
            formatted_error = f"{error:.2f}" if error < 9999 else f"{error:.2E}"
            ax.text(
                i,
                j,
                formatted_error,
                ha="center",
                va="center",
                rotation=90,
                usetex=False,
            )

        ax.set_xlabel("Forecast Step (Lead Time in Hours)")
        ax.set_xticks(np.arange(pred_steps))
        ax.set_xticklabels(np.arange(0, pred_steps * 6, 6), rotation=45)
        ax.set_yticks(np.arange(d_f))
        ax.set_yticklabels(errors_norm.coords["variable"].values)
        ax.set_ylabel("Variable")

        ax.set_title(f"Scorecard for {member.title()}")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                path_out,
                "scorecards",
                f"scorecard_member_{member}.png",
            ),
            dpi=300,
        )
        plt.close(fig)


def plot_rank_histogram(
    variable,
    forecast,
    ground_truth,
    path_out,
    color_palette,
    model_name,
    level=None,
    region="",
    date_time="",
    sample_size=10,
):
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
        region (str): The region name.
        date_time (str): The date and time string.
        sample_size (int): The sample size for the rank histogram.

    """
    print(f"Creating rank histogram for variable: {variable}, level: {level}")

    forecast_var = (
        forecast[variable].sel(isobaricInhPa=level).drop_isel(step=0)
        if level
        else forecast[variable]
    ).drop_isel(step=0)
    ground_truth_var = (
        ground_truth[variable].sel(isobaricInhPa=level).drop_isel(step=0)
        if level
        else ground_truth[variable].drop_isel(step=0)
    )

    ground_truth_var = (
        ground_truth_var.expand_dims("member")
        .drop_vars(["number", "time"])
        .assign_coords(member=[9999])
    )

    combined = xr.concat([forecast_var, ground_truth_var], dim="member")

    # Create a random subsample
    combined_stacked = combined.stack(z=("step", "latitude", "longitude"))
    sample_size = min(sample_size, combined_stacked.z.size)
    indices = np.sort(
        np.random.choice(combined_stacked.z.size, size=sample_size, replace=False)
    )
    combined_sample = combined_stacked.isel(z=indices)

    ranks = combined_sample.chunk(dict(member=-1)).rank("member")

    # Get the rank of the ground truth
    gt_rank = ranks.sel(member=9999)
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
        f"Rank Histogram for {variable}{' at level ' + str(level) if level else ''}\nRegion: {region}, Init Date: {date_time}, Model: {model_name}"
    )
    ax.set_xlabel("Rank of Ground Truth After 10-Days Forecast")
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
    region="",
    date_time="",
):
    """
    Plot the energy spectra for each member and the mean of all members using wavenumber on a log-log scale.
    """
    print(f"Plotting energy spectra for variable: {variable} on level {level}")
    fig, ax = plt.subplots(figsize=(12, 9))

    if level is not None:
        energy_spectra_forecast_lev = energy_spectra_forecast[variable].sel(
            isobaricInhPa=level
        )
        energy_spectra_unperturbed_lev = energy_spectra_unperturbed[variable].sel(
            isobaricInhPa=level
        )
        energy_spectra_ground_truth_lev = energy_spectra_ground_truth[variable].sel(
            isobaricInhPa=level
        )
    else:
        energy_spectra_forecast_lev = energy_spectra_forecast[variable]
        energy_spectra_unperturbed_lev = energy_spectra_unperturbed[variable]
        energy_spectra_ground_truth_lev = energy_spectra_ground_truth[variable]

    # Plot each member's energy spectra
    for member in energy_spectra_forecast_lev.member.values:
        ax.loglog(
            energy_spectra_forecast_lev.sel(member=member).wavenumber,
            energy_spectra_forecast_lev.sel(member=member).values,
            color=color_palette[1],
            alpha=alpha_value / 2,  # Plot becomes very busy
            label=f"{model_name} Member" if member == 0 else None,
        )

    # Calculate and plot the mean energy spectra
    mean_energy_spectra = energy_spectra_forecast_lev.mean(dim="member")
    ax.loglog(
        mean_energy_spectra.wavenumber,
        mean_energy_spectra.values,
        color=color_palette[2],
        label=f"{model_name} Mean",
    )

    # Plot the unperturbed and ground truth energy spectra
    ax.loglog(
        energy_spectra_unperturbed_lev.wavenumber,
        energy_spectra_unperturbed_lev.values,
        color=color_palette[3],
        label=f"{model_name} Unperturbed",
        linestyle="--",
    )
    ax.loglog(
        energy_spectra_ground_truth_lev.wavenumber,
        energy_spectra_ground_truth_lev.values,
        color=color_palette[0],
        label="Ground Truth: ERA5",
        linestyle=":",
    )

    ax.set_xlabel("Wavenumber (num wave cycles around the globe)")
    ax.set_ylabel("Energy Density")
    ax.set_title(
        f"Energy Spectra for {variable} on level {level}\nRegion: {region},"
        f"Init Date: {date_time}, Model: {model_name}"
    )
    ax.legend()

    # Set x and y axis to log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Add grid lines
    ax.grid(True, which="both", ls="-", alpha=0.2)
    k_min, k_max = ax.get_xlim()
    _, y_max = ax.get_ylim()

    # k ^ (-3) slope: This slope is associated with the theory of two -
    # dimensional turbulence and is often referred to as the enstrophy cascade.
    # It was proposed by Robert Kraichnan in the 1960s. In the atmosphere, this
    # slope is typically observed at larger scales(low wavenumbers) and is
    # associated with quasi - two - dimensional, rotational dynamics dominated
    # by the Earth's rotation and stratification.
    # k ^ (-5 / 3) slope: This slope is associated with the theory of three -
    # dimensional turbulence and is often referred to as the energy cascade or
    # Kolmogorov spectrum. It was first proposed by Andrey Kolmogorov in the
    # 1940s. In the atmosphere, this slope is typically observed at smaller
    # scales(high wavenumbers) where the flow behaves more like three -
    # dimensional isotropic turbulence.
    k_range = np.logspace(np.log10(k_min + 1e-6), np.log10(k_max), 10)

    ax.loglog(k_range, y_max * k_range ** (-3), "k--", alpha=0.5, label="k⁻³")
    ax.loglog(k_range, y_max * k_range ** (-5 / 3), "k:", alpha=0.5, label="k⁻⁵ᐟ³")

    # Update legend
    ax.legend(loc="lower left")

    if y_lims is not None:
        ax.set_ylim(y_lims)

    plt.savefig(
        os.path.join(path_out, f"energy_spectra_comparison_{variable}_{level}.png"),
        dpi=300,
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
    region="",
    date_time="",
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
        rmse_mean = rmse_mean.sel(isobaricInhPa=level)
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

    rmse_mean[variable].plot(ax=ax, color=color_palette[2], label=f"{model_name} Mean")

    rmse_unperturbed[variable].plot(
        ax=ax, color=color_palette[3], label=f"{model_name} Unperturbed"
    )

    _set_xticks(ax, rmse)
    ax.set_xlabel("Forecast Lead-Time (hours)")

    if y_lims is not None:
        (ymin, ymax) = y_lims
        ax.set_ylim(ymin.item() * 0.9, ymax.item() * 1.1)

    ax.set_ylabel("RMSE")
    ax.set_title(
        f"Root Mean Square Error: {variable}{' at level ' + str(level) if level is not None else ''}\nRegion: {region}, Init Date: {date_time}, Model: {model_name}"
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
    sr_spread_skill_ratio,
    sr_ensemble_spread,
    sr_spread_skill_ratio_ifs,
    sr_ensemble_spread_ifs,
    path_out,
    color_palette,
    model_names,
    level=None,
    y_lims1=None,
    y_lims2=None,
    region="",
    date_time="",
):
    """
    Plot the spread-skill ratio for both models on the same plot.

    Args:
        variable (str): The variable to plot.
        spread_skill_ratio (xr.DataArray): The spread-skill ratio data for the default model.
        ensemble_spread (xr.DataArray): The ensemble spread data for the default model.
        spread_skill_ratio_ifs (xr.DataArray): The spread-skill ratio data for the IFS ENS model.
        ensemble_spread_ifs (xr.DataArray): The ensemble spread data for the IFS ENS model.
        path_out (str): The path to the output directory.
        color_palette (list): The color palette.
        model_names (list): List containing names of the models, e.g. [DefaultModel, IFSEns]
        level (int): The level to plot.
        y_lims1 (tuple): The y-limits for the spread-skill ratio plot.
        y_lims2 (tuple): The y-limits for the ensemble spread plot.
        region (str): The region name.
        date_time (str): The date and time string.
    """
    print(
        f"Creating combined spread-skill ratio plots for variable: {variable}, level: {level}"
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    if "isobaricInhPa" in sr_spread_skill_ratio[variable].dims:
        spread_skill_ratio = sr_spread_skill_ratio.sel(isobaricInhPa=level)
        ensemble_spread = sr_ensemble_spread.sel(isobaricInhPa=level)
        spread_skill_ratio_ifs = sr_spread_skill_ratio_ifs.sel(isobaricInhPa=level)
        ensemble_spread_ifs = sr_ensemble_spread_ifs.sel(isobaricInhPa=level)
    else:
        spread_skill_ratio = sr_spread_skill_ratio
        ensemble_spread = sr_ensemble_spread
        spread_skill_ratio_ifs = sr_spread_skill_ratio_ifs
        ensemble_spread_ifs = sr_ensemble_spread_ifs

    # Plot spread-skill ratio for both models
    spread_skill_ratio[variable].plot(
        ax=ax, color=color_palette[5], label=f"{model_names[0]} SSR"
    )
    spread_skill_ratio_ifs[variable].plot(
        ax=ax, color=color_palette[0], label=f"{model_names[1]} SSR"
    )

    ax2 = ax.twinx()
    ensemble_spread[variable].plot(
        ax=ax2,
        color=color_palette[4],
        linestyle="--",
        label=f"{model_names[0]} Ensemble Spread",
    )
    ensemble_spread_ifs[variable].plot(
        ax=ax2,
        color=color_palette[1],
        linestyle="--",
        label=f"{model_names[1]} Ensemble Spread",
    )

    if y_lims1 is not None:
        (ymin, ymax) = y_lims1
        ax.set_ylim(0, ymax.item() * 1.1)

    ax.set_xlabel("Forecast Lead-Time (hours)")
    ax.set_ylabel("Spread-skill ratio")
    ax.set_title(
        f"Spread-Skill Ratio Comparison for {variable}{' at level ' + str(level) if level is not None else ''}\nRegion: {region}, Init Date: {date_time}"
    )

    if y_lims2 is not None:
        (ymin, ymax) = y_lims2
        ax2.set_ylim(ymin.item() * 0.9, ymax.item() * 1.1)

    ax2.set_ylabel("Ensemble Spread")
    ax2.tick_params(axis="y")
    ax2.set_title("")

    _set_xticks(ax, spread_skill_ratio)

    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

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
    region="",
    date_time="",
):
    def _plot_map(ground_truth_var, ax):
        lat = ground_truth.latitude.values
        lon = ground_truth.longitude.values

        ax.pcolormesh(
            lon,
            lat,
            ground_truth_var.isel(step=0).values,
            cmap="plasma",
            transform=ccrs.PlateCarree(),
        )

        ax.coastlines()
        ax.set_xticks([])
        ax.set_yticks([])

    print(f"Creating timeseries plots for variable: {variable}, level: {level}")
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
    ground_truth_var = (
        ground_truth[variable].sel(isobaricInhPa=level)
        if level is not None
        else ground_truth[variable]
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
            fc_mean_member.plot(ax=ax, color=color_palette[1], alpha=alpha_value)

    fc_mean_var.mean(dim="member").plot(
        ax=ax, color=color_palette[2], label=f"{model_name} Mean"
    )
    fc_mean_unperturbed_var.plot(
        ax=ax, color=color_palette[3], label=f"{model_name} Unperturbed"
    )

    if y_lims is not None:
        (ymin, ymax) = y_lims
        extent = ymax - ymin
        ax.set_ylim(
            ymin.item() - 0.1 * extent.item(), ymax.item() + 0.1 * extent.item()
        )

    # Arbitrary number to make sure the density curves look decent
    if fc_mean.member.size >= 20:
        # Collect all values for the density plot at the latest time step
        fc_values_last_step = fc_mean_var.isel(step=-1).values

        # Create an inset axis for the density plot
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("right", size=1.2, pad=0.1)

        # Calculate the KDE
        kde = gaussian_kde(fc_values_last_step, bw_method=0.5)
        y_lims = ax.get_ylim()

        # Generate y-values for the KDE
        y_values = np.linspace(y_lims[0], y_lims[1], 100)
        y_values_norm = (y_values - y_lims[0]) / (y_lims[1] - y_lims[0]) * 10

        # Calculate the densities
        densities = kde(y_values)

        # Normalize the densities so that the area under the curve is 1
        densities /= np.trapz(densities, y_values_norm)

        # Plot the densities
        ax2.plot(densities, y_values, color=color_palette[1])

        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Density")

    ax_inset = inset_axes(
        ax,
        width="20%",
        height="15%",
        loc="upper left",
        axes_class=GeoAxes,
        axes_kwargs=dict(projection=ccrs.PlateCarree()),
    )
    _plot_map(ground_truth_var, ax=ax_inset)

    ax.set_xlabel("Forecast Lead-Time (hours)")
    _set_xticks(ax, fc_mean_var)
    ax.set_ylabel(
        f"{variable} [{ground_truth_var.attrs['units']}] {' at level ' + str(level) if level is not None else ''}"
    )
    ax.set_title(
        f"Comparison of Forecast vs Ground-Truth: {variable}{' at level ' + str(level) if level is not None else ''}\nRegion: {region}, Init Date: {date_time}, Model: {model_name}"
    )
    ax.legend(loc="lower left")
    plt.savefig(
        os.path.join(
            path_out,
            f"timeseries_fc_gt_{variable}{'_' + str(level) if level is not None else ''}.png",
        ),
        dpi=300,
    )
    plt.close(fig)


if __name__ == "__main__":
    args, config = parse_args()
    path_in = os.path.join(args.out_dir, str(args.date_time), args.model_name)

    data = load_and_prepare_data(
        path_in,
        config["selected_vars"],
        args.crop_region,
        args.model_name,
        args.perturbation_init,
        args.perturbation_latent,
        args.layer,
        args.members,
        debug_mode=args.debug,
    )

    assert (
        args.members <= 50
    ), "The number of ensemble members must be less than or equal to 50 to plot IFS ENS"
    path_out = os.path.join(
        path_in,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}_layer_{args.layer}",
        args.crop_region,
        f"png_{args.model_name}",
    )
    path_out_ifs = os.path.join(
        path_in,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}_layer_{args.layer}",
        args.crop_region,
        "png_ifs",
    )
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(path_out_ifs, exist_ok=True)
    os.makedirs(os.path.join(path_out, "scorecards"), exist_ok=True)
    os.makedirs(os.path.join(path_out_ifs, "scorecards"), exist_ok=True)

    print("data loaded", flush=True)
    default_stats = calculate_stats(
        data["ground_truth"],
        data["forecast"],
        data["forecast_unperturbed"],
        args.crop_region,
    )
    print("stats calculated", flush=True)
    ifs_stats = calculate_stats(
        data["ground_truth"],
        data["forecast_ifs"],
        data["forecast_ifs_unperturbed"],
        args.crop_region,
    )
    print("ifs stats calculated", flush=True)

    alpha_value = 1 / data["forecast"].member.size ** (5 / 8)
    variables = list(data["forecast"].data_vars)
    vars_3d = [
        var for var in variables if "isobaricInhPa" in data["forecast"][var].dims
    ]
    vars_2d = [
        var for var in variables if "isobaricInhPa" not in data["forecast"][var].dims
    ]

    y_lims = calculate_y_lims(
        vars_3d,
        vars_2d,
        data["forecast"],
        data["forecast_ifs"],
        default_stats,
        ifs_stats,
    )
    print("y_lims calculated", flush=True)

    default_plot_args = prepare_plot_args(
        data,
        default_stats,
        vars_3d,
        vars_2d,
        config,
        **y_lims,
        use_ifs=False,
        path_out=path_out,
        model_name=args.model_name.title(),
        region=args.crop_region.title(),
        date_time=args.date_time,
    )

    ifs_plot_args = prepare_plot_args(
        data,
        ifs_stats,
        vars_3d,
        vars_2d,
        config,
        **y_lims,
        use_ifs=True,
        path_out=path_out_ifs,
        model_name="IFS ENS",
        region=args.crop_region.title(),
        date_time=args.date_time,
    )

    # Individual plots evaluated vs. ERA5
    for plot_args in [default_plot_args, ifs_plot_args]:
        for args_i in plot_args["energy_spectra"]:
            plot_energy_spectra(**args_i)
        for args_i in plot_args["rank_histogram"]:
            plot_rank_histogram(**args_i)
        for args_i in plot_args["rmse"]:
            plot_rmse(**args_i)
        for args_i in plot_args["timeseries_fc_gt"]:
            plot_timeseries_fc_gt(**args_i)
        plot_error_map(**plot_args["error_map"][0])

    # Combined plots using both IFS and MODEL in combination
    for args_dict in default_plot_args["spread_skill_ratio"]:
        args_dict["sr_spread_skill_ratio_ifs"] = ifs_stats["sr_spread_skill_ratio"]
        args_dict["sr_ensemble_spread_ifs"] = ifs_stats["sr_ensemble_spread"]
        args_dict["model_names"] = [args_dict.pop("model_name"), "IFS ENS"]

        plot_spread_skill_ratio(**args_dict)
