import os

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .preprocess_data import load_and_prepare_data, parse_args


def power_alpha_scale(data, epsilon=1e-6, power=0.5):
    max_abs_value = np.max(np.abs(data))
    if max_abs_value == 0:
        return np.zeros_like(data)
    else:
        normalized_data = np.abs(data) / max_abs_value + epsilon
        # Apply a power scaling to make values close to zero more transparent
        alpha = normalized_data**power
        alpha = np.clip(alpha, 0, 1)
        if not isinstance(alpha, np.ndarray):
            alpha = alpha.values
        return alpha


def calculate_rgba(data, norm, cmap_name="RdBu_r"):
    rgb = plt.get_cmap(cmap_name)(norm(data))
    alpha = power_alpha_scale(data)
    # Add a new axis to make alpha three-dimensional
    alpha = alpha[:, :, np.newaxis]
    rgba = np.concatenate((rgb[:, :, :3], alpha), axis=-1)
    return rgba


def plot_variable_3d(
    difference, var, member, step, fig, ax, mappable, vmin, vmax, args
):
    ax.cla()

    ax.set_title(
        "Differences Perturbed - Unperturbed Forecast \n"
        # BUG: seperate the two
        f"Initial Perturbation of T: {args.perturbation_init} - Latent Perturbation: {args.perturbation_latent}\n"
        f"Variable: {var.upper()} - Member: {member:02} - Step: {step:02}"
    )
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
        # If data doesn't have a vertical dimension, plot it at a constant
        # level of 0
        Z = np.zeros_like(X)
        rgba = calculate_rgba(data, norm)
        ax.plot_surface(X, Y, Z, facecolors=rgba, shade=False)
        ax.set_zticks([0])  # Only show the 0 label on the z-axis

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Level")
    ax.tick_params(axis="x", colors="darkgray")
    ax.tick_params(axis="y", colors="darkgray")
    ax.tick_params(axis="z", colors="darkgray")

    return fig, ax

    # Set the color of the tick labels to dark gray


def update_plot(num, difference, var, fig, ax, member, mappable, vmin, vmax, args):
    fig, ax = plot_variable_3d(
        difference, var, member, num, fig, ax, mappable, vmin, vmax, args
    )
    return fig, ax


def create_and_save_animation(path, difference, var, member, unit, vmin, vmax, args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mappable = cm.ScalarMappable(cmap="RdBu_r")
    mappable.set_array([])

    # Ensure vmin and vmax are symmetrical around zero
    max_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -max_abs, max_abs

    mappable.set_clim(vmin, vmax)
    fig, ax = plot_variable_3d(
        difference, var, member, 0, fig, ax, mappable, vmin, vmax, args
    )
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="vertical",
        shrink=0.5,
        pad=0.2,
        label=f"{var.upper()} [{unit}]",
    )

    for label in cbar.ax.get_yticklabels():
        label.set_color("darkgray")

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=difference.step.size,
        fargs=(difference, var, fig, ax, member, mappable, vmin, vmax, args),
    )
    ani.save(f"{path}/{var}_difference.gif", writer="imagemagick")
    plt.close()


def process_member(member, forecast, forecast_unperturbed, path_forecast, args, config):
    path_gif = os.path.join(path_forecast, args.crop_region, str(member), "animations")
    os.makedirs(path_gif, exist_ok=True)
    variables = config["selected_vars"]
    difference = forecast.sel(member=member) - forecast_unperturbed

    for var in variables:
        print("Creating 3d differences animation for variable: ", var)
        unit = forecast[var].attrs["units"]
        vmin = difference[var].min().values
        vmax = difference[var].max().values
        create_and_save_animation(
            path_gif, difference, var, member, unit, vmin, vmax, args
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

    members_to_plot = [0, 1]

    for member in members_to_plot:
        process_member(
            member,
            data["forecast"],
            data["forecast_unperturbed"],
            path_forecast,
            args,
            config,
        )


if __name__ == "__main__":
    main()
