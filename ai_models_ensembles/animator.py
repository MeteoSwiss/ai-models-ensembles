import os

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from .data_load_preproc import load_and_prepare_data, parse_args


def create_plot(ax, data, var, level, step, title_prefix, lat, lon):
    is_surface = level == "surface"

    if not is_surface:
        plot_data = data[var].isel(isobaricInhPa=level, step=step).values
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
            dim = "step"
            image.set_array(data[var].sel(isobaricInhPa=level).isel(
                {dim: i}).values
                if not is_surface else
                data[var].isel({dim: i}).values)
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


def create_and_save_animation(path, ground_truth, var, level, fig, updatefig):
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=ground_truth.time.size,
        interval=200,
        blit=True)
    ani.save(f"{path}/{var}_{level}_comparison.gif", writer="imagemagick")
    plt.close()


def process_member(member, forecast, ground_truth,
                   path_forecast, lat, lon, args):
    path_gif = f"{path_forecast}/{args.crop_region}/{member}/animations"
    os.makedirs(path_gif, exist_ok=True)
    variables = forecast.data_vars
    pressure_levels = forecast.isobaricInhPa.values if "isobaricInhPa" in forecast.dims else []
    for var in variables:
        print("Creating animation for member", member, "and variable:", var)
        if "isobaricInhPa" in forecast[var].dims:
            if args.print_pressure_levels:
                for level in pressure_levels:
                    fig, updatefig = plot_variable(
                        forecast.sel(member=member),
                        ground_truth, var, level, lat, lon)
                    create_and_save_animation(
                        path_gif, ground_truth, var, level, fig, updatefig)
        else:
            fig, updatefig = plot_variable(
                forecast.sel(member=member),
                ground_truth, var, "surface", lat, lon)
            create_and_save_animation(
                path_gif, ground_truth, var, "surface", fig, updatefig)


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

    members_to_plot = data["forecast"].member.values[:3]

    # with multiprocessing.Pool() as pool:
    #     pool.starmap(
    #         process_member,
    #         [(member, data["forecast"],
    #         data["ground_truth"],
    #         path_forecast, lat, lon, args)
    #         for member in members_to_plot])

    for member in members_to_plot:
        process_member(
            member, data["forecast"],
            data["ground_truth"],
            path_forecast, lat, lon, args)


if __name__ == "__main__":
    main()
