import argparse
import multiprocessing
import os

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description="Generate 3D gif-animations.")
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
parser.add_argument(
    "print_pressure_levels",
    action='store_false',
    help="print pressure levels")
args = parser.parse_args()


def create_plot(ax, data, var, level, step, title_prefix, lat, lon):
    dim = "step"
    is_surface = level == "surface"
    im = ax.pcolormesh(
        lon, lat,
        data[var].sel(isobaricInhPa=level).isel({dim: step}).values
        if not is_surface else data[var].isel({dim: step}).values,
        cmap="plasma", transform=ccrs.PlateCarree(), animated=True)
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
                {dim: i}).values if not is_surface else data[var].isel({dim: i}).values)
            ax.set_title(
                f"{title_prefix} {var} at {'surface' if is_surface else level}, {(i+1)*6} hours")
        return image1, image2,
    return updatefig


def plot_variable(forecast, ground_truth, var, level, lat, lon):
    fig, axes = plt.subplots(2, figsize=(10, 15), subplot_kw={
                             'projection': ccrs.PlateCarree()})
    image1 = create_plot(axes[0], forecast, var, level, 0, "Forecast", lat, lon)
    image2 = create_plot(axes[1], ground_truth, var, level, 0, "Ground Truth", lat, lon)
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


def process_member(member, forecast, ground_truth, path_forecast, lat, lon):
    print("Creating animations for member: ", member)
    path_gif = f"{path_forecast}/{member}/animations"
    variables = forecast.data_vars
    pressure_levels = forecast.isobaricInhPa.values if "isobaricInhPa" in forecast.dims else []
    for var in variables:
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
    path_forecast = os.path.join(
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}")
    forecast = xr.open_zarr(path_forecast + "/forecast.zarr", consolidated=True)
    ground_truth = xr.open_zarr(
        f"{args.date_time}/{args.model_name}/ground_truth.zarr",
        consolidated=True)
    ground_truth = ground_truth.isel(
        surface=0, step=0, number=0).drop_isel(
        time=0)
    ground_truth["step"] = ("time", np.arange(len(ground_truth["time"])))
    ground_truth = ground_truth.swap_dims({"time": "step"})

    lat = ground_truth.latitude.values
    lon = ground_truth.longitude.values

    members_to_plot = forecast.member.values[:5]

    with multiprocessing.Pool() as pool:
        pool.starmap(process_member,
                     [(member, forecast, ground_truth, path_forecast, lat, lon)
                      for member in members_to_plot])


if __name__ == "__main__":
    main()
