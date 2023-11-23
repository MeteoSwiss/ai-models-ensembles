import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import xarray as xr

parser = argparse.ArgumentParser(description='Generate gif-animations.')
parser.add_argument(
    'MODEL_OUTPUT',
    metavar='MODEL_OUTPUT',
    type=str,
    help='the model grib output')
parser.add_argument(
    'PRINT_PRESSURE_LEVELS',
    type=bool,
    help='print pressure levels')
args = parser.parse_args()


def create_plot(ax, data, var, level, step, title_prefix):
    dim = 'step' if 'step' in data.dims else 'time'
    im = ax.imshow(
        data[var].sel(
            isobaricInhPa=level).isel(
            {dim: step}) if level != 'surface' else data[var].isel(
                {dim: step}),
        animated=True,
        cmap="plasma")
    ax.set_title(
        f"{title_prefix} {var} at {level if level != 'surface' else 'surface'}, {step*6} hours")
    return im


def create_update_function(forecast, ground_truth, var, level, im1, im2, axs):
    def updatefig(i):
        for im, data, ax, title_prefix in zip(
            [im1, im2],
            [forecast, ground_truth],
                axs, ["Forecast", "Ground Truth"]):
            dim = 'step' if 'step' in data.dims else 'time'
            im.set_array(
                data[var].sel(
                    isobaricInhPa=level).isel(
                    {dim: i}) if level != 'surface' else data[var].isel(
                    {dim: i}))
            ax.set_title(
                f"{title_prefix} {var} at {level if level != 'surface' else 'surface'}, {i*6} hours")
        return im1, im2,
    return updatefig


def plot_variable(forecast, ground_truth, var, level):
    fig, axs = plt.subplots(2, figsize=(10, 15))
    print("Variable: ", var)
    print("Level: ", level)
    im1 = create_plot(axs[0], forecast, var, level, 0, "Forecast")
    im2 = create_plot(axs[1], ground_truth, var, level, 0, "Ground Truth")
    updatefig = create_update_function(
        forecast, ground_truth, var, level, im1, im2, axs)
    return fig, updatefig


def create_and_save_animation(ground_truth, var, level, fig, updatefig):
    ani = animation.FuncAnimation(
        fig,
        updatefig,
        frames=ground_truth.time.size - 1,
        interval=200,
        blit=True)
    ani.save(f"animations/{var}_{level}_comparison.gif", writer='imagemagick')
    plt.close()


def main():
    # here. Open datasets for different 'shortName' values
    print("Creating animations...")

    ground_truth = xr.open_dataset("era5.grib")
    forecast = xr.open_dataset("")

    for forecast, shortName in zip(forecasts, shortNames):
        variables = forecast.data_vars
        pressure_levels = forecast.isobaricInhPa.values if 'isobaricInhPa' in forecast.dims else []
        for var in variables:
            if 'isobaricInhPa' in forecast[var].dims and args.PRINT_PRESSURE_LEVELS:
                for level in pressure_levels:
                    fig, updatefig = plot_variable(forecast, ground_truth, var, level)
                    create_and_save_animation(ground_truth, var, level, fig, updatefig)
            else:
                fig, updatefig = plot_variable(forecast, ground_truth, var, 'surface')
                create_and_save_animation(ground_truth, var, 'surface', fig, updatefig)


if __name__ == "__main__":
    main()
