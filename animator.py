import argparse
import multiprocessing

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import xarray as xr

parser = argparse.ArgumentParser(description="Generate gif-animations.")
# Add the arguments
parser.add_argument("path_out", type=str, help="The path to the output directory")
parser.add_argument(
    "print_pressure_levels",
    action='store_false',
    help="print pressure levels")
args = parser.parse_args()


def create_plot(ax, data, var, level, step, title_prefix):
    dim = "step" if "step" in data.dims else "time"
    is_surface = level == "surface"
    im = ax.imshow(
        data[var].sel(isobaricInhPa=level).isel({dim: step})
        if not is_surface else data[var].isel({dim: step}), animated=True,
        cmap="plasma")
    ax.set_title(
        f"{title_prefix} {var} at {'surface' if is_surface else level}, {step*6} hours")
    return im


def create_update_function(forecast, ground_truth, var, level, image1, image2, axes):
    def updatefig(i):
        is_surface = level == "surface"
        for image, data, ax, title_prefix in zip(
            [image1, image2],
            [forecast, ground_truth],
                axes, ["Forecast", "Ground Truth"]):
            dim = "step" if "step" in data.dims else "time"
            image.set_array(data[var].sel(isobaricInhPa=level).isel(
                {dim: i}) if not is_surface else data[var].isel({dim: i}))
            ax.set_title(
                f"{title_prefix} {var} at {'surface' if is_surface else level}, {i*6} hours")
        return image1, image2,
    return updatefig


def plot_variable(forecast, ground_truth, var, level):
    fig, axes = plt.subplots(2, figsize=(10, 15))
    image1 = create_plot(axes[0], forecast, var, level, 0, "Forecast")
    image2 = create_plot(axes[1], ground_truth, var, level, 0, "Ground Truth")
    updatefig = create_update_function(
        forecast, ground_truth, var, level, image1, image2, axes)
    return fig, updatefig


def create_and_save_animation(path, ground_truth, var, level, fig, updatefig):
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=ground_truth.time.size - 1,
        interval=200,
        blit=True)
    ani.save(f"{path}/{var}_{level}_comparison.gif", writer="imagemagick")
    plt.close()


def process_member(member, forecast, ground_truth):
    print("Creating animations for member: ", member)
    path_gif = f"{args.path_out}/{member}/animations"
    variables = forecast.data_vars
    pressure_levels = forecast.isobaricInhPa.values if "isobaricInhPa" in forecast.dims else []
    for var in variables:
        if "isobaricInhPa" in forecast[var].dims:
            if args.print_pressure_levels:
                for level in pressure_levels:
                    fig, updatefig = plot_variable(
                        forecast.sel(member=member),
                        ground_truth, var, level)
                    create_and_save_animation(
                        path_gif, ground_truth, var, level, fig, updatefig)
        else:
            fig, updatefig = plot_variable(
                forecast.sel(member=member),
                ground_truth, var, "surface")
            create_and_save_animation(
                path_gif, ground_truth, var, "surface", fig, updatefig)


def main():
    ground_truth = xr.open_zarr("ground_truth.zarr", consolidated=True)
    forecast = xr.open_zarr(
        f"{args.path_out}/forecast.zarr", consolidated=True)

    with multiprocessing.Pool() as pool:
        pool.starmap(process_member,
                     [(member, forecast, ground_truth)
                      for member in forecast.member.values])

if __name__ == "__main__":
    main()
