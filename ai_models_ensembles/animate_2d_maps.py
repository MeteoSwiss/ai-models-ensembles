from pathlib import Path
from typing import Any, Dict

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from ai_models_ensembles.utils import (
    build_output_filename,
    ensure_dir,
    save_dataset,
    save_npz,
)


def _save_payload(
    metric: str,
    variable: str,
    level,
    member_token: str,
    data: Dict[str, xr.DataArray],
    base_dir: Path,
    qualifier: str,
) -> None:
    data_dir = ensure_dir(base_dir / "data" / metric)
    ds = xr.Dataset({key: value for key, value in data.items()})
    ds.attrs.update({"variable": variable, "level": str(level), "member": member_token})
    filename = build_output_filename(
        metric=metric,
        variable=variable,
        level=level,
        qualifier=qualifier,
        ensemble=member_token,
        ext="nc",
    )
    save_dataset(ds, data_dir, filename)


def create_plot(
    ax,
    data,
    var: str,
    level,
    step: int,
    title_prefix: str,
    lat,
    lon,
    vmin: float,
    vmax: float,
):
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


def plot_variable(forecast, ground_truth, var: str, level, lat, lon):
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

    fig, axes = plt.subplots(2, figsize=(10, 15), subplot_kw={"projection": ccrs.PlateCarree()})
    image1 = create_plot(axes[0], forecast, var, level, 0, "Forecast", lat, lon, vmin, vmax)
    image2 = create_plot(axes[1], ground_truth, var, level, 0, "Ground Truth", lat, lon, vmin, vmax)

    # Add colorbars
    fig.colorbar(image1, ax=axes[0], orientation="horizontal", pad=0.05)
    fig.colorbar(image2, ax=axes[1], orientation="horizontal", pad=0.05)

    updatefig = create_update_function(
        forecast, ground_truth, var, level, image1, image2, axes, lat, lon
    )
    return fig, updatefig


def create_plot_metric(
    ax,
    metric_data,
    var: str,
    level,
    step: int,
    title_prefix: str,
    lat,
    lon,
    vmin: float,
    vmax: float,
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


def plot_metric(metric_data, var: str, level, lat, lon, metric_name: str):
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

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    image = create_plot_metric(ax, metric_data, var, level, 0, metric_name, lat, lon, vmin, vmax)

    # Add colorbar
    fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.05)

    updatefig = create_update_function_metric(
        metric_data, var, level, image, ax, lat, lon, metric_name
    )
    return fig, updatefig


def plot_crps_mean_map(
    crps_data: xr.DataArray,
    var: str,
    level,
    lat,
    lon,
    out_png_dir: Path,
    artifact_root: Path,
    ensemble_token: str = "ensemble",
) -> None:
    """
    Render a mean-over-time CRPS map and save PNG plus NPZ artifact for intercomparison.

    - PNG saved to out_png_dir as crps_map_<var>_<level>.png
    - NPZ saved under artifact_root/data/crps_map with latitude, longitude, crps arrays
    """
    # Reduce across time-like dims except for lat/lon
    reduce_dims = [
        d for d in ["step", "time", "init_time", "lead_time", "member"] if d in crps_data.dims
    ]
    if reduce_dims:
        crps_map = crps_data.mean(dim=reduce_dims, skipna=True)
    else:
        crps_map = crps_data

    # Ensure Y is ascending for pcolormesh
    lat_name = next((n for n in crps_map.dims if n in ("latitude", "lat", "y")), None)
    lon_name = next((n for n in crps_map.dims if n in ("longitude", "lon", "x")), None)
    if lat_name is None or lon_name is None:
        # Fall back to provided lat/lon arrays
        lat_vals = lat
        lon_vals = lon
        Z = crps_map.values
    else:
        _lat_vals = crps_map[lat_name].values
        if _lat_vals[0] > _lat_vals[-1]:
            crps_map = crps_map.sortby(lat_name)
        lat_vals = crps_map[lat_name].values
        lon_vals = crps_map[lon_name].values
        Z = crps_map.values

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines()
    vmax = float(np.nanmax(Z)) if np.isfinite(Z).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.pcolormesh(lon_vals, lat_vals, Z, cmap="viridis", shading="auto", vmin=0.0, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cb.set_label("CRPS")
    ax.set_title(f"CRPS map (mean) — {var} at {level}")
    out_png_dir = ensure_dir(out_png_dir)
    out_png = out_png_dir / build_output_filename(
        metric="crps_map",
        variable=var,
        level=level,
        qualifier=None,
        ensemble=ensemble_token,
        ext="png",
    )
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)

    # Save NPZ artifact
    data_dir = ensure_dir(Path(artifact_root) / "data" / "crps_map")
    out_npz = build_output_filename(
        metric="crps_map",
        variable=var,
        level=level,
        qualifier=None,
        ensemble=ensemble_token,
        ext="npz",
    )
    save_npz(
        {
            "crps": np.asarray(Z),
            "latitude": np.asarray(lat_vals),
            "longitude": np.asarray(lon_vals),
            "variable": np.array([var]),
        },
        data_dir,
        out_npz,
    )


def create_update_function(forecast, ground_truth, var: str, level, image1, image2, axes, lat, lon):
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
            ax.set_title(f"{title_prefix} {var} at {level}, {(i + 1) * 6} hours")
        return (
            image1,
            image2,
        )

    return updatefig


def create_and_save_animation(
    path: str, data, var: str, level, fig, updatefig, metric_name: str = "comparison"
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
    ani = animation.FuncAnimation(fig, updatefig, frames=data.step.size, interval=200, blit=True)
    dest = Path(path)
    ani.save(dest / f"{metric_name}_{var}_{level}.gif", writer="imagemagick")
    plt.close()


def create_update_function_metric(
    metric_data, var: str, level, image, ax, lat, lon, metric_name: str
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
        ax.set_title(f"{metric_name} of {var} at {level}, {(i + 1) * 6} hours")
        return (image,)

    return updatefig


def plot_static_steps(path_gif: Path, data, var: str, level, lat, lon, metric_name: str):
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
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), subplot_kw={"projection": ccrs.PlateCarree()})
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
        ax.set_title(f"{(step) * 6} hours")  # Only show the hour as subtitle
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
    path_gif = ensure_dir(Path(path_gif))
    plt.savefig(path_gif / f"{metric_name}_{var}_6fig.png")
    plt.close(fig)


def process_member(
    member: int,
    forecast,
    ground_truth,
    stats: Dict[str, Any],
    path_forecast: str,
    lat,
    lon,
    args: Any,
) -> None:
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
    path_base = Path(path_forecast) / args.crop_region / str(member)
    path_gif = ensure_dir(path_base / "animations")
    artifact_root = ensure_dir(
        Path(path_forecast) / args.crop_region / f"artifacts_{args.model_name}"
    )
    member_artifacts = ensure_dir(artifact_root / f"member_{member:02}")
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
                _save_payload(
                    "animation",
                    var,
                    level,
                    f"member{member:02}",
                    {
                        "forecast": forecast_var,
                        "ground_truth": ground_truth_var,
                    },
                    member_artifacts,
                    qualifier="forecast_vs_ground_truth",
                )
                fig, updatefig = plot_variable(forecast_var, ground_truth_var, var, level, lat, lon)
                create_and_save_animation(
                    str(path_gif),
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
                _save_payload(
                    "error",
                    var,
                    level,
                    f"member{member:02}",
                    {"error": error_data},
                    member_artifacts,
                    qualifier="forecast_minus_truth",
                )

                # 1. Error
                fig, updatefig = plot_metric(error_data, var, level, lat, lon, metric_name="Error")
                create_and_save_animation(
                    str(path_gif),
                    error_data,
                    var,
                    level,
                    fig,
                    updatefig,
                    metric_name="Error",
                )

                # Create static plot for error
                plot_static_steps(path_gif, error_data, var, level, lat, lon, metric_name="Error")

                # Retrieve RMSE data (root of squared error)
                rmse_data = np.sqrt(error_data**2)
                _save_payload(
                    "rmse",
                    var,
                    level,
                    f"member{member:02}",
                    {"rmse": rmse_data},
                    member_artifacts,
                    qualifier="member_rmse",
                )
                fig, updatefig = plot_metric(rmse_data, var, level, lat, lon, metric_name="RMSE")
                create_and_save_animation(
                    str(path_gif),
                    rmse_data,
                    var,
                    level,
                    fig,
                    updatefig,
                    metric_name="RMSE",
                )

                # Create static plot for RMSE
                plot_static_steps(path_gif, rmse_data, var, level, lat, lon, metric_name="RMSE")

        else:
            level = "surface"
            # Existing forecast and ground truth animation
            forecast_var = forecast[var].sel(member=member)
            ground_truth_var = ground_truth[var]
            _save_payload(
                "animation",
                var,
                level,
                f"member{member:02}",
                {
                    "forecast": forecast_var,
                    "ground_truth": ground_truth_var,
                },
                member_artifacts,
                qualifier="forecast_vs_ground_truth",
            )
            fig, updatefig = plot_variable(forecast_var, ground_truth_var, var, level, lat, lon)
            create_and_save_animation(
                str(path_gif),
                forecast_var,
                var,
                level,
                fig,
                updatefig,
                metric_name="Forecast_vs_GroundTruth",
            )

            # Create static plot
            plot_static_steps(path_gif, forecast_var, var, level, lat, lon, metric_name="Forecast")

            # Retrieve error data
            error_data = stats["diff"][var].sel(member=member)
            _save_payload(
                "error",
                var,
                level,
                f"member{member:02}",
                {"error": error_data},
                member_artifacts,
                qualifier="forecast_minus_truth",
            )
            # 1. Error
            fig, updatefig = plot_metric(error_data, var, level, lat, lon, metric_name="Error")
            create_and_save_animation(
                str(path_gif),
                error_data,
                var,
                level,
                fig,
                updatefig,
                metric_name="Error",
            )

            # Create static plot for error
            plot_static_steps(path_gif, error_data, var, level, lat, lon, metric_name="Error")

            # Retrieve RMSE data
            rmse_data = np.sqrt(error_data**2)
            _save_payload(
                "rmse",
                var,
                level,
                f"member{member:02}",
                {"rmse": rmse_data},
                member_artifacts,
                qualifier="member_rmse",
            )
            fig, updatefig = plot_metric(rmse_data, var, level, lat, lon, metric_name="RMSE")
            create_and_save_animation(
                str(path_gif), rmse_data, var, level, fig, updatefig, metric_name="RMSE"
            )

            # Create static plot for RMSE
            plot_static_steps(path_gif, rmse_data, var, level, lat, lon, metric_name="RMSE")


def process_ensemble_metrics(
    forecast,
    ground_truth,
    stats: Dict[str, Any],
    path_forecast: str,
    lat,
    lon,
    args: Any,
) -> None:
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
    path_base = Path(path_forecast) / args.crop_region / "ensemble"
    path_gif = ensure_dir(path_base / "animations")
    artifact_root = ensure_dir(
        Path(path_forecast) / args.crop_region / f"artifacts_{args.model_name}"
    )
    ensemble_artifacts = ensure_dir(artifact_root / "ensemble")
    variables = forecast.data_vars
    for var in variables:
        print("Creating ensemble metrics animations and static plots for variable:", var)
        if "isobaricInhPa" in forecast[var].dims:
            for level in forecast.isobaricInhPa.values:
                # 3. CRPS between ensemble members and ground_truth
                crps_data = stats["crps"][var].sel(isobaricInhPa=level)
                _save_payload(
                    "crps",
                    var,
                    level,
                    "ensemble",
                    {"crps": crps_data},
                    ensemble_artifacts,
                    qualifier="ensemble_vs_truth",
                )

                # Plot and save the animations
                fig, updatefig = plot_metric(crps_data, var, level, lat, lon, metric_name="CRPS")
                create_and_save_animation(
                    str(path_gif),
                    crps_data,
                    var,
                    level,
                    fig,
                    updatefig,
                    metric_name="CRPS",
                )

                # Create static plot for CRPS
                plot_static_steps(path_gif, crps_data, var, level, lat, lon, metric_name="CRPS")

                # CRPS mean map (PNG + NPZ)
                plot_crps_mean_map(
                    crps_data,
                    var,
                    level,
                    lat,
                    lon,
                    path_gif,
                    ensemble_artifacts,
                    ensemble_token="ensemble",
                )

                # 4. Standard deviations across ensemble members
                ensemble_std = stats["ensemble_spread_grid"][var].sel(isobaricInhPa=level)
                _save_payload(
                    "ensemble_spread",
                    var,
                    level,
                    "ensemble",
                    {"ensemble_spread": ensemble_std},
                    ensemble_artifacts,
                    qualifier="ensemble_std",
                )

                # Plot and save the animations
                fig, updatefig = plot_metric(
                    ensemble_std, var, level, lat, lon, metric_name="Ensemble Std Dev"
                )
                create_and_save_animation(
                    str(path_gif),
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
            _save_payload(
                "crps",
                var,
                level,
                "ensemble",
                {"crps": crps_data},
                ensemble_artifacts,
                qualifier="ensemble_vs_truth",
            )
            # Plot and save the animations
            fig, updatefig = plot_metric(crps_data, var, level, lat, lon, metric_name="CRPS")
            create_and_save_animation(
                str(path_gif), crps_data, var, level, fig, updatefig, metric_name="CRPS"
            )

            # Create static plot for CRPS
            plot_static_steps(path_gif, crps_data, var, level, lat, lon, metric_name="CRPS")

            # CRPS mean map (PNG + NPZ)
            plot_crps_mean_map(
                crps_data,
                var,
                level,
                lat,
                lon,
                path_gif,
                ensemble_artifacts,
                ensemble_token="ensemble",
            )

            # 4. Ensemble standard deviation
            ensemble_std = stats["ensemble_spread_grid"][var]
            _save_payload(
                "ensemble_spread",
                var,
                level,
                "ensemble",
                {"ensemble_spread": ensemble_std},
                ensemble_artifacts,
                qualifier="ensemble_std",
            )
            # Plot and save the animations
            fig, updatefig = plot_metric(
                ensemble_std, var, level, lat, lon, metric_name="Ensemble Std Dev"
            )
            create_and_save_animation(
                str(path_gif),
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


__all__ = [
    "process_member",
    "process_ensemble_metrics",
]
