import os

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .preprocess_data import calculate_stats, load_and_prepare_data, parse_args


def create_plot(ax, data, var, level, step, title_prefix, lat, lon):
    is_surface = level == "surface"

    if not is_surface:
        plot_data = data[var].sel(isobaricInhPa=level).isel(step=step).values
    else:
        plot_data = data[var].isel(step=step).values

    im = ax.pcolormesh(
        lon,
        lat,
        plot_data,
        cmap="plasma",
        transform=ccrs.PlateCarree(),
        animated=True)

    ax.set_title(
        f"{title_prefix} {var} at {'surface' if is_surface else level}, {(step+1)*6} hours")
    ax.coastlines()
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def create_update_function(forecast, ground_truth, var, level,
                           image1, image2, axes, lat, lon):
    def updatefig(i):
        is_surface = level == "surface"
        for image, data, ax, title_prefix in zip(
            [image1, image2],
            [forecast, ground_truth],
            axes, ["Forecast", "Ground Truth"]):
            if not is_surface:
                plot_data = data[var].sel(isobaricInhPa=level).isel(step=i).values
            else:
                plot_data = data[var].isel(step=i).values
            image.set_array(plot_data.ravel())
            ax.set_title(
                f"{title_prefix} {var} at {'surface' if is_surface else level}, {(i+1)*6} hours")
        return image1, image2,
    return updatefig

def plot_variable(forecast, ground_truth, var, level, lat, lon):
    fig, axes = plt.subplots(2, figsize=(10, 15), subplot_kw={
                             'projection': ccrs.PlateCarree()})
    image1 = create_plot(
        axes[0],
        forecast,
        var,
        level,
        0,
        "Forecast",
        lat,
        lon)
    image2 = create_plot(
        axes[1],
        ground_truth,
        var,
        level,
        0,
        "Ground Truth",
        lat,
        lon)
    updatefig = create_update_function(
        forecast,
        ground_truth,
        var,
        level,
        image1,
        image2,
        axes,
        lat,
        lon)
    return fig, updatefig

def create_and_save_animation(path, data, var, level, fig, updatefig, metric_name='comparison'):
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=data.step.size,
        interval=200,
        blit=True)
    ani.save(f"{path}/{metric_name}_{var}_{level}.gif", writer="imagemagick")
    plt.close()

def create_plot_metric(ax, metric_data, var, level, step, title_prefix, lat, lon):
    is_surface = level == "surface"

    if not is_surface:
        plot_data = metric_data.isel(step=step).values
    else:
        plot_data = metric_data.isel(step=step).values

    # Set color map limits based on the metric
    if title_prefix == "Error":
        vmax = np.nanmax(np.abs(metric_data.values))
        vmin = -vmax
        cmap = "bwr"
    elif title_prefix == "CRPS":
        vmin = 0
        vmax = np.nanmax(metric_data.values)
        cmap = "viridis"
    else:
        vmin = 0
        vmax = np.nanmax(metric_data.values)
        cmap = "plasma"

    im = ax.pcolormesh(
        lon,
        lat,
        plot_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        animated=True)

    ax.set_title(
        f"{title_prefix} of {var} at {'surface' if is_surface else level}, {(step+1)*6} hours")
    ax.coastlines()
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def create_update_function_metric(metric_data, var, level, image, ax, lat, lon, metric_name):
    def updatefig(i):
        is_surface = level == "surface"
        if not is_surface:
            plot_data = metric_data.isel(step=i).values
        else:
            plot_data = metric_data.isel(step=i).values
        image.set_array(plot_data.ravel())
        ax.set_title(
            f"{metric_name} of {var} at {'surface' if is_surface else level}, {(i+1)*6} hours")
        return image,
    return updatefig

def plot_metric(metric_data, var, level, lat, lon, metric_name):
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={
                           'projection': ccrs.PlateCarree()})
    image = create_plot_metric(ax, metric_data, var, level,
                               0, metric_name, lat, lon)
    updatefig = create_update_function_metric(
        metric_data, var, level, image, ax, lat, lon, metric_name)
    return fig, updatefig

def process_member(member, forecast, ground_truth, stats,
                   path_forecast, lat, lon, args):
    path_gif = f"{path_forecast}/{args.crop_region}/{member}/animations"
    os.makedirs(path_gif, exist_ok=True)
    variables = forecast.data_vars
    for var in variables:
        print("Creating animations for member", member, "and variable:", var)
        if "isobaricInhPa" in forecast[var].dims:
            for level in forecast.isobaricInhPa.values:
                # Existing forecast and ground truth animation
                fig, updatefig = plot_variable(
                    forecast.sel(member=member),
                    ground_truth, var, level, lat, lon)
                create_and_save_animation(
                    path_gif, forecast.sel(member=member), var, level, fig, updatefig, metric_name="Forecast_vs_GroundTruth")

                # Retrieve error data from stats
                error_data = stats['diff'][var].sel(member=member)
                if "isobaricInhPa" in error_data.dims:
                    error_data = error_data.sel(isobaricInhPa=level)
                else:
                    level = "surface"
                # 1. Error
                fig, updatefig = plot_metric(
                    error_data, var, level, lat, lon, metric_name="Error")
                create_and_save_animation(
                    path_gif, error_data, var, level, fig, updatefig, metric_name='Error')

                # Retrieve RMSE data (root of squared error)
                rmse_data = np.sqrt(error_data ** 2)
                # 2. RMSE
                fig, updatefig = plot_metric(
                    rmse_data, var, level, lat, lon, metric_name="RMSE")
                create_and_save_animation(
                    path_gif, rmse_data, var, level, fig, updatefig, metric_name='RMSE')

        else:
            level = "surface"
            # Existing forecast and ground truth animation
            fig, updatefig = plot_variable(
                forecast.sel(member=member),
                ground_truth, var, level, lat, lon)
            create_and_save_animation(
                path_gif, forecast.sel(member=member), var, level, fig, updatefig, metric_name="Forecast_vs_GroundTruth")

            # Retrieve error data
            error_data = stats['diff'][var].sel(member=member)
            # 1. Error
            fig, updatefig = plot_metric(
                error_data, var, level, lat, lon, metric_name="Error")
            create_and_save_animation(
                path_gif, error_data, var, level, fig, updatefig, metric_name='Error')

            # Retrieve RMSE data
            rmse_data = np.sqrt(error_data ** 2)
            # 2. RMSE
            fig, updatefig = plot_metric(
                rmse_data, var, level, lat, lon, metric_name="RMSE")
            create_and_save_animation(
                path_gif, rmse_data, var, level, fig, updatefig, metric_name='RMSE')

def process_ensemble_metrics(forecast, ground_truth, stats,
                             path_forecast, lat, lon, args):
    path_gif = f"{path_forecast}/{args.crop_region}/ensemble/animations"
    os.makedirs(path_gif, exist_ok=True)
    variables = forecast.data_vars
    for var in variables:
        print("Creating ensemble metrics animations for variable:", var)
        if "isobaricInhPa" in forecast[var].dims:
            for level in forecast.isobaricInhPa.values:
                # 3. CRPS between ensemble members and ground_truth
                crps_data = stats['crps'][var]
                if "isobaricInhPa" in crps_data.dims:
                    crps_data = crps_data.sel(isobaricInhPa=level)
                else:
                    level = "surface"

                # 4. Standard deviations across ensemble members
                ensemble_std = stats['ensemble_spread_grid'][var]
                if "isobaricInhPa" in ensemble_std.dims:
                    ensemble_std = ensemble_std.sel(isobaricInhPa=level)
                else:
                    level = "surface"

                # Plot and save the animations
                fig, updatefig = plot_metric(
                    crps_data, var, level, lat, lon, metric_name='CRPS')
                create_and_save_animation(
                    path_gif, crps_data, var, level, fig, updatefig, metric_name='CRPS')

                fig, updatefig = plot_metric(
                    ensemble_std, var, level, lat, lon, metric_name='Ensemble Std Dev')
                create_and_save_animation(
                    path_gif, ensemble_std, var, level, fig, updatefig, metric_name='Ensemble_Std_Dev')
        else:
            level = "surface"
            # 3. CRPS
            crps_data = stats['crps'][var]
            # 4. Ensemble standard deviation
            ensemble_std = stats['ensemble_spread_grid'][var]

            # Plot and save the animations
            fig, updatefig = plot_metric(
                crps_data, var, level, lat, lon, metric_name='CRPS')
            create_and_save_animation(
                path_gif, crps_data, var, level, fig, updatefig, metric_name='CRPS')

            fig, updatefig = plot_metric(
                ensemble_std, var, level, lat, lon, metric_name='Ensemble Std Dev')
            create_and_save_animation(
                path_gif, ensemble_std, var, level, fig, updatefig, metric_name='Ensemble_Std_Dev')
def main():
    args, config = parse_args()
    path_in = os.path.join(args.out_dir, str(args.date_time), args.model_name)
    path_forecast = os.path.join(
        args.out_dir,
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}")

    data = load_and_prepare_data(
        path_in,
        config["selected_vars"],
        args.crop_region,
        args.model_name,
        args.perturbation_init,
        args.perturbation_latent,
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

    members_to_plot = data["forecast"].member.values[:3]

    for member in members_to_plot:
        process_member(
            member, data["forecast"],
            data["ground_truth"],
            stats,
            path_forecast, lat, lon, args)

    # Process ensemble metrics
    process_ensemble_metrics(
        data['forecast'],
        data['ground_truth'],
        stats,
        path_forecast, lat, lon, args)


if __name__ == "__main__":
    main()
