import os

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .preprocess_data import calculate_stats, load_and_prepare_data, parse_args


def create_plot(ax, data, var, level, step, title_prefix, lat, lon, vmin, vmax):
    """
    Create a plot for a given variable, level, and time step.

    Args:
        ax: Matplotlib axis object
        data: xarray DataArray
        var: Variable to plot
        level: Level to plot
        step: Time step to plot
        title_prefix: Prefix for the title
        lat: Latitude values
        lon: Longitude values
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale

    Returns:
        im: Matplotlib QuadMesh object
    """
    plot_data = data.isel(step=step).values

    im = ax.pcolormesh(
        lon,
        lat,
        plot_data,
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        animated=True,
    )

    ax.set_title(f"{title_prefix} {var} at {level}, {(step + 1) * 6} hours")
    ax.coastlines()
    ax.set_xticks([])
    ax.set_yticks([])

    return im


def plot_variable(forecast, ground_truth, var, level, lat, lon):
    """
    Plot a variable for a given forecast and ground truth data.

    Args:
        forecast: Forecast xarray DataArray
        ground_truth: Ground truth xarray DataArray
        var: Variable to plot
        level: Level to plot
        lat: Latitude values
        lon: Longitude values

    Returns:
        fig: Matplotlib Figure object
        updatefig: Update function for the animation
    """
    # Calculate global vmin and vmax
    vmin = np.nanmin([forecast.values.min(), ground_truth.values.min()])
    vmax = np.nanmax([forecast.values.max(), ground_truth.values.max()])

    fig, axes = plt.subplots(
        2, figsize=(10, 15), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    image1 = create_plot(
        axes[0], forecast, var, level, 0, "Forecast", lat, lon, vmin, vmax
    )
    image2 = create_plot(
        axes[1], ground_truth, var, level, 0, "Ground Truth", lat, lon, vmin, vmax
    )

    # Add colorbars
    fig.colorbar(image1, ax=axes[0], orientation="horizontal", pad=0.05)
    fig.colorbar(image2, ax=axes[1], orientation="horizontal", pad=0.05)

    updatefig = create_update_function(
        forecast, ground_truth, var, level, image1, image2, axes, lat, lon
    )
    return fig, updatefig


def create_plot_metric(
    ax, metric_data, var, level, step, title_prefix, lat, lon, vmin, vmax
):
    """
    Create a plot for a given metric, level, and time step.

    Args:
        ax: Matplotlib axis object
        metric_data: xarray DataArray
        var: Variable to plot
        level: Level to plot
        step: Time step to plot
        title_prefix: Prefix for the title
        lat: Latitude values
        lon: Longitude values
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale

    Returns:
        im: Matplotlib QuadMesh object
    """
    plot_data = metric_data.isel(step=step).values

    if title_prefix == "Error":
        cmap = "bwr"
        cmap_center = 0
        divnorm = TwoSlopeNorm(vmin=vmin, vcenter=cmap_center, vmax=vmax)
    elif title_prefix == "CRPS":
        cmap = "viridis"
    else:
        cmap = "plasma"

    im = ax.pcolormesh(
        lon,
        lat,
        plot_data,
        cmap=cmap,
        vmin=vmin if title_prefix != "Error" else None,
        vmax=vmax if title_prefix != "Error" else None,
        norm=divnorm if title_prefix == "Error" else None,
        transform=ccrs.PlateCarree(),
        animated=True,
    )

    ax.set_title(f"{title_prefix} of {var} at {level}, {(step + 1) * 6} hours")
    ax.coastlines()
    ax.set_xticks([])
    ax.set_yticks([])

    return im


def plot_metric(metric_data, var, level, lat, lon, metric_name):
    """
    Plot a metric for a given dataset.

    Args:
        metric_data: Metric xarray DataArray
        var: Variable to plot
        level: Level to plot
        lat: Latitude values
        lon: Longitude values
        metric_name: Name of the metric

    Returns:
        fig: Matplotlib Figure object
        updatefig: Update function for the animation
    """
    # Calculate global vmin and vmax
    if metric_name == "Error":
        vmax = np.nanmax(np.abs(metric_data.values))
        vmin = -vmax
    else:
        vmin = np.nanmin(metric_data.values)
        vmax = np.nanmax(metric_data.values)

    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    image = create_plot_metric(
        ax, metric_data, var, level, 0, metric_name, lat, lon, vmin, vmax
    )

    # Add colorbar
    fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.05)

    updatefig = create_update_function_metric(
        metric_data, var, level, image, ax, lat, lon, metric_name
    )
    return fig, updatefig


def create_update_function(
    forecast, ground_truth, var, level, image1, image2, axes, lat, lon
):
    """
    Create an update function for the animation.

    Args:
        forecast: Forecast data
        ground_truth: Ground truth data
        var: Variable to plot
        level: Level to plot
        image1: Matplotlib image object for forecast
        image2: Matplotlib image object for ground truth
        axes: Matplotlib axes object
        lat: Latitude values
        lon: Longitude values

        Returns:
        updatefig: Update function for the animation
    """

    def updatefig(i):
        for image, data, ax, title_prefix in zip(
            [image1, image2],
            [forecast, ground_truth],
            axes,
            ["Forecast", "Ground Truth"],
        ):
            plot_data = data.isel(step=i).values
            image.set_array(plot_data.ravel())
            ax.set_title(f"{title_prefix} {var} at {level}, {(i+1)*6} hours")
        return (
            image1,
            image2,
        )

    return updatefig


def create_and_save_animation(
    path, data, var, level, fig, updatefig, metric_name="comparison"
):
    """
    Create and save an animation for a given dataset.

    Args:
        path: Path to save the animation
        data: xarray dataset
        var: Variable to plot
        level: Level to plot
        fig: Matplotlib figure object
        updatefig: Update function for the animation
        metric_name: Name of the metric to plot
    """
    ani = animation.FuncAnimation(
        fig, updatefig, frames=data.step.size, interval=200, blit=True
    )
    ani.save(f"{path}/{metric_name}_{var}_{level}.gif", writer="imagemagick")
    plt.close()


def create_update_function_metric(
    metric_data, var, level, image, ax, lat, lon, metric_name
):
    """
    Create an update function for the animation.

    Args:
        metric_data: Metric data
        var: Variable to plot
        level: Level to plot
        image: Matplotlib image object
        ax: Matplotlib axes object
        lat: Latitude values
        lon: Longitude values
        metric_name: Name of the metric

    Returns:
        updatefig: Update function for the animation
    """

    def updatefig(i):
        plot_data = metric_data.isel(step=i).values
        image.set_array(plot_data.ravel())
        ax.set_title(f"{metric_name} of {var} at {level}, {(i+1)*6} hours")
        return (image,)

    return updatefig


def plot_static_steps(path_gif, data, var, level, lat, lon, metric_name):
    """
    Create static plots for a given dataset with a shared colorbar.

    Args:
        path_gif: Path to save the static plots
        data: xarray dataset
        var: Variable to plot
        level: Level to plot
        lat: Latitude values
        lon: Longitude values
        metric_name: Name of the metric
    """
    # Create a figure with 2x2 subplots and a colorbar
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 15), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    steps = [0, 8, 16, 24, 32, 39]

    # Determine the common color range
    vmin = np.min([data.isel(step=s).values.min() for s in steps])
    vmax = np.max([data.isel(step=s).values.max() for s in steps])

    # Set color map limits based on the metric
    if metric_name == "Error":
        cmap = "bwr"
        cmap_center = 0
        divnorm = TwoSlopeNorm(vmin=vmin, vcenter=cmap_center, vmax=vmax)
    elif metric_name == "CRPS":
        cmap = "viridis"
    else:
        cmap = "plasma"

    for ax, step in zip(axes.flatten(), steps):
        plot_data = data.isel(step=step).values
        im = ax.pcolormesh(
            lon,
            lat,
            plot_data,
            cmap=cmap,
            norm=divnorm if metric_name == "Error" else None,
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(f"{(step)*6} hours")  # Only show the hour as subtitle
        ax.coastlines()
        ax.set_xticks([])
        ax.set_yticks([])

    # Add a shared colorbar
    cb_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # Position of the colorbar
    fig.colorbar(
        im,
        cax=cb_ax,
        orientation="horizontal",
        label=f"{var} at {level}",
        pad=0.1,
        aspect=50,
    )
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.03, hspace=0.03)
    # Add main title
    fig.suptitle(f"{var} at {level}: {metric_name.title()}", fontsize=16)

    # plt.tight_layout(
    #     rect=[0, 0.05, 1, 0.95]
    # )  # Adjust layout to accommodate colorbar and main title
    plt.savefig(f"{path_gif}/{metric_name}_{var}_6fig.png")
    plt.close(fig)


def process_member(
    member, forecast, ground_truth, stats, path_forecast, lat, lon, args
):
    """
    Process a single ensemble member.

    Args:
        member: Ensemble member to process
        forecast: Forecast data
        ground_truth: Ground truth data
        stats: Statistics data
        path_forecast: Path to save the forecast data
        lat: Latitude values
        lon: Longitude values
        args: Parsed command line arguments
    """
    path_gif = f"{path_forecast}/{args.crop_region}/{member}/animations"
    os.makedirs(path_gif, exist_ok=True)
    variables = forecast.data_vars
    for var in variables:
        print(
            "Creating animations and static plots for member",
            member,
            "and variable:",
            var,
        )
        if "isobaricInhPa" in forecast[var].dims:
            for level in forecast.isobaricInhPa.values:
                # Existing forecast and ground truth animation
                forecast_var = forecast[var].sel(member=member, isobaricInhPa=level)
                ground_truth_var = ground_truth[var].sel(isobaricInhPa=level)
                fig, updatefig = plot_variable(
                    forecast_var, ground_truth_var, var, level, lat, lon
                )
                create_and_save_animation(
                    path_gif,
                    forecast_var,
                    var,
                    level,
                    fig,
                    updatefig,
                    metric_name="Forecast_vs_GroundTruth",
                )

                # Create static plot
                plot_static_steps(
                    path_gif, forecast_var, var, level, lat, lon, metric_name="Forecast"
                )

                # Retrieve error data from stats
                error_data = stats["diff"][var].sel(member=member, isobaricInhPa=level)

                # 1. Error
                fig, updatefig = plot_metric(
                    error_data, var, level, lat, lon, metric_name="Error"
                )
                create_and_save_animation(
                    path_gif,
                    error_data,
                    var,
                    level,
                    fig,
                    updatefig,
                    metric_name="Error",
                )

                # Create static plot for error
                plot_static_steps(
                    path_gif, error_data, var, level, lat, lon, metric_name="Error"
                )

                # Retrieve RMSE data (root of squared error)
                rmse_data = np.sqrt(error_data**2)
                fig, updatefig = plot_metric(
                    rmse_data, var, level, lat, lon, metric_name="RMSE"
                )
                create_and_save_animation(
                    path_gif, rmse_data, var, level, fig, updatefig, metric_name="RMSE"
                )

                # Create static plot for RMSE
                plot_static_steps(
                    path_gif, rmse_data, var, level, lat, lon, metric_name="RMSE"
                )

        else:
            level = "surface"
            # Existing forecast and ground truth animation
            forecast_var = forecast[var].sel(member=member)
            ground_truth_var = ground_truth[var]
            fig, updatefig = plot_variable(
                forecast_var, ground_truth_var, var, level, lat, lon
            )
            create_and_save_animation(
                path_gif,
                forecast_var,
                var,
                level,
                fig,
                updatefig,
                metric_name="Forecast_vs_GroundTruth",
            )

            # Create static plot
            plot_static_steps(
                path_gif, forecast_var, var, level, lat, lon, metric_name="Forecast"
            )

            # Retrieve error data
            error_data = stats["diff"][var].sel(member=member)
            # 1. Error
            fig, updatefig = plot_metric(
                error_data, var, level, lat, lon, metric_name="Error"
            )
            create_and_save_animation(
                path_gif, error_data, var, level, fig, updatefig, metric_name="Error"
            )

            # Create static plot for error
            plot_static_steps(
                path_gif, error_data, var, level, lat, lon, metric_name="Error"
            )

            # Retrieve RMSE data
            rmse_data = np.sqrt(error_data**2)
            fig, updatefig = plot_metric(
                rmse_data, var, level, lat, lon, metric_name="RMSE"
            )
            create_and_save_animation(
                path_gif, rmse_data, var, level, fig, updatefig, metric_name="RMSE"
            )

            # Create static plot for RMSE
            plot_static_steps(
                path_gif, rmse_data, var, level, lat, lon, metric_name="RMSE"
            )


def process_ensemble_metrics(
    forecast, ground_truth, stats, path_forecast, lat, lon, args
):
    """
    Process ensemble metrics.

    Args:
        forecast: Forecast data
        ground_truth: Ground truth data
        stats: Statistics data
        path_forecast: Path to save the forecast data
        lat: Latitude values
        lon: Longitude values
        args: Parsed command line arguments
    """
    path_gif = f"{path_forecast}/{args.crop_region}/ensemble/animations"
    os.makedirs(path_gif, exist_ok=True)
    variables = forecast.data_vars
    for var in variables:
        print(
            "Creating ensemble metrics animations and static plots for variable:", var
        )
        if "isobaricInhPa" in forecast[var].dims:
            for level in forecast.isobaricInhPa.values:
                # 3. CRPS between ensemble members and ground_truth
                crps_data = stats["crps"][var].sel(isobaricInhPa=level)

                # Plot and save the animations
                fig, updatefig = plot_metric(
                    crps_data, var, level, lat, lon, metric_name="CRPS"
                )
                create_and_save_animation(
                    path_gif, crps_data, var, level, fig, updatefig, metric_name="CRPS"
                )

                # Create static plot for CRPS
                plot_static_steps(
                    path_gif, crps_data, var, level, lat, lon, metric_name="CRPS"
                )

                # 4. Standard deviations across ensemble members
                ensemble_std = stats["ensemble_spread_grid"][var].sel(
                    isobaricInhPa=level
                )

                # Plot and save the animations
                fig, updatefig = plot_metric(
                    ensemble_std, var, level, lat, lon, metric_name="Ensemble Std Dev"
                )
                create_and_save_animation(
                    path_gif,
                    ensemble_std,
                    var,
                    level,
                    fig,
                    updatefig,
                    metric_name="Ensemble_Std_Dev",
                )

                # Create static plot for Ensemble Std Dev
                plot_static_steps(
                    path_gif,
                    ensemble_std,
                    var,
                    level,
                    lat,
                    lon,
                    metric_name="Ensemble_Std_Dev",
                )
        else:
            level = "surface"
            # 3. CRPS
            crps_data = stats["crps"][var]
            # Plot and save the animations
            fig, updatefig = plot_metric(
                crps_data, var, level, lat, lon, metric_name="CRPS"
            )
            create_and_save_animation(
                path_gif, crps_data, var, level, fig, updatefig, metric_name="CRPS"
            )

            # Create static plot for CRPS
            plot_static_steps(
                path_gif, crps_data, var, level, lat, lon, metric_name="CRPS"
            )

            # 4. Ensemble standard deviation
            ensemble_std = stats["ensemble_spread_grid"][var]
            # Plot and save the animations
            fig, updatefig = plot_metric(
                ensemble_std, var, level, lat, lon, metric_name="Ensemble Std Dev"
            )
            create_and_save_animation(
                path_gif,
                ensemble_std,
                var,
                level,
                fig,
                updatefig,
                metric_name="Ensemble_Std_Dev",
            )

            # Create static plot for Ensemble Std Dev
            plot_static_steps(
                path_gif,
                ensemble_std,
                var,
                level,
                lat,
                lon,
                metric_name="Ensemble_Std_Dev",
            )


def main():
    args, config = parse_args()
    path_in = os.path.join(args.out_dir, str(args.date_time), args.model_name)
    path_forecast = os.path.join(
        args.out_dir,
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}_layer_{args.layer}",
    )

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

    lat = data["ground_truth"].latitude.values
    lon = data["ground_truth"].longitude.values

    # Compute the statistics
    stats = calculate_stats(
        data["ground_truth"],
        data["forecast"],
        data["forecast_unperturbed"],
        args.crop_region,
    )

    # TODO: make this an user input
    members_to_plot = [0, 1]

    for member in members_to_plot:
        process_member(
            member,
            data["forecast"],
            data["ground_truth"],
            stats,
            path_forecast,
            lat,
            lon,
            args,
        )

    # Process ensemble metrics
    process_ensemble_metrics(
        data["forecast"], data["ground_truth"], stats, path_forecast, lat, lon, args
    )


if __name__ == "__main__":
    main()
