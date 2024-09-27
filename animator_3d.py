import argparse
import multiprocessing
import os

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

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
    "crop_region",
    type=str,
    help="The region to crop the data to")
args = parser.parse_args()


def power_alpha_scale(data, epsilon=1e-6, power=0.5):
    max_abs_value = np.max(np.abs(data))
    if max_abs_value == 0:
        return np.zeros_like(data)
    else:
        normalized_data = np.abs(data) / max_abs_value + epsilon
        # Apply a power scaling to make values close to zero more transparent
        alpha = normalized_data ** power
        alpha = np.clip(alpha, 0, 1)
        if not isinstance(alpha, np.ndarray):
            alpha = alpha.values
        return alpha


def calculate_rgba(data, norm, cmap_name='RdBu_r'):
    rgb = plt.get_cmap(cmap_name)(norm(data))
    alpha = power_alpha_scale(data)
    alpha = alpha[:, :, np.newaxis]  # Add a new axis to make alpha three-dimensional
    rgba = np.concatenate((rgb[:, :, :3], alpha), axis=-1)
    return rgba


def plot_variable_3d(difference, var, member, step, fig, ax, mappable, vmin, vmax):
    ax.cla()

    ax.set_title(
        "Differences Perturbed - Unperturbed Forecast \n"
        # BUG: seperate the two
        f"Initial Perturbation of T: {args.perturbation_init} - Latent Perturbation: {args.perturbation_latent}\n"
        f"Variable: {var.upper()} - Member: {member:02} - Step: {step:02}")
    ax.title.set_position([0.6, 1.03])
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    data = difference[var].isel(step=step)

    # Create a grid from latitude and longitude
    X, Y = np.meshgrid(data.longitude, data.latitude)
    # Define the normalization with the center at zero

    if "isobaricInhPa" in data.dims:
        for level in data.isobaricInhPa:
            data_level = data.sel(isobaricInhPa=level)
            Z = np.ones_like(X) * level.values  # Convert level to numpy array
            rgba = calculate_rgba(data_level, norm)
            ax.plot_surface(X, Y, Z, facecolors=rgba, shade=False)
        ax.invert_zaxis()
    else:
        # If data doesn't have a vertical dimension, plot it at a constant level of 0
        Z = np.zeros_like(X)
        rgba = calculate_rgba(data, norm)
        ax.plot_surface(X, Y, Z, facecolors=rgba, shade=False)
        ax.set_zticks([0])  # Only show the 0 label on the z-axis

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Level')
    ax.tick_params(axis='x', colors='darkgray')
    ax.tick_params(axis='y', colors='darkgray')
    ax.tick_params(axis='z', colors='darkgray')

    return fig, ax

    # Set the color of the tick labels to dark gray


def update_plot(num, difference, var, fig, ax, member, mappable, vmin, vmax):
    fig, ax = plot_variable_3d(difference, var, member, num,
                               fig, ax, mappable, vmin, vmax)
    return fig, ax


def create_and_save_animation(path, difference, var, member, unit, vmin, vmax):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mappable = cm.ScalarMappable(cmap='RdBu_r')
    mappable.set_array([])

    # Ensure vmin and vmax are symmetrical around zero
    max_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -max_abs, max_abs

    mappable.set_clim(vmin, vmax)
    fig, ax = plot_variable_3d(difference, var, member, 0,
                               fig, ax, mappable, vmin, vmax)
    cbar = fig.colorbar(mappable, ax=ax, orientation='vertical', shrink=0.5, pad=0.2,
                        label=f"{var.upper()} [{unit}]")

    for label in cbar.ax.get_yticklabels():
        label.set_color('darkgray')

    ani = animation.FuncAnimation(
        fig, update_plot, frames=difference.step.size,
        fargs=(difference, var, fig, ax, member, mappable, vmin, vmax))
    ani.save(f"{path}/{var}_difference.gif", writer='imagemagick')
    plt.close()


def process_member(member, perturbed, unperturbed, path_perturbed, crop_region):
    init_perturbed = xr.open_dataset(
        os.path.join(path_perturbed, str(member), "era5_init.grib"),
        engine="cfgrib")
    init_perturbed = init_perturbed.expand_dims({"step": [np.timedelta64(0, 'ns')]})
    perturbed = xr.concat([init_perturbed, perturbed], dim="step")
    path_gif = os.path.join(
        args.date_time,
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}",
        str(member),
        crop_region,
        "animations")
    os.makedirs(path_gif, exist_ok=True)
    variables = perturbed.data_vars
    difference = perturbed.sel(member=member) - unperturbed.sel(member=0)

    for var in variables:
        print("Creating difference animation for variable: ", var)
        unit = variables[var].units
        vmin = difference[var].min().values
        vmax = difference[var].max().values
        create_and_save_animation(path_gif, difference, var, member, unit, vmin, vmax)


def main():
    path_unperturbed = os.path.join(args.date_time, args.model_name)
    path_perturbed = os.path.join(
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}")
    forecast_unperturbed = xr.open_zarr(
        path_unperturbed + "/forecast.zarr",
        consolidated=True)
    forecast_perturbed = xr.open_zarr(
        path_perturbed + "/forecast.zarr",
        consolidated=True)
    init_unperturbed = xr.open_dataset(
        path_unperturbed +
        "/era5_init.grib",
        engine="cfgrib")

    init_unperturbed = init_unperturbed.expand_dims({"step": [np.timedelta64(0, 'ns')]})
    unperturbed = xr.concat([init_unperturbed, forecast_unperturbed], dim="step")

    members_to_plot = forecast_perturbed.member.values[:5]

    with multiprocessing.Pool() as pool:
        pool.starmap(process_member,
                     [(member, forecast_perturbed, unperturbed, path_perturbed, crop_region)
                      for member in members_to_plot])


if __name__ == "__main__":
    main()
