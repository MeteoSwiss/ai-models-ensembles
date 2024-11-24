import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.stats import gaussian_kde, wasserstein_distance

from .preprocess_data import load_and_prepare_data, parse_args


def subsample_da(data_da: xr.DataArray, max_samples: int) -> xr.DataArray:
    """
    Subsample a DataArray efficiently, avoiding out-of-order indexing.
    """
    N = data_da.size
    if N > max_samples:
        indices = np.sort(np.random.choice(N, size=max_samples, replace=False))
        data_sampled = data_da.isel(all_points=indices)
    else:
        data_sampled = data_da
    return data_sampled


def plot_density_distribution(
    variable: str,
    forecast: xr.Dataset,
    ground_truth: xr.Dataset,
    path_out: str,
    color_palette: List[str],
    model_name: str,
    level: Optional[float] = None,
    region: str = "",
    date_time: str = "",
    max_samples: int = 100000,
):
    print(
        f"Creating density distribution plot for variable: {variable}, level: {level}"
    )

    if level is not None:
        forecast_var = forecast[variable].sel(isobaricInhPa=level)
        ground_truth_var = ground_truth[variable].sel(isobaricInhPa=level)
    else:
        forecast_var = forecast[variable]
        ground_truth_var = ground_truth[variable]

    # Stack all dimensions into a single dimension called 'all_points'
    forecast_var_stacked = forecast_var.stack(all_points=forecast_var.dims)
    ground_truth_var_stacked = ground_truth_var.stack(all_points=ground_truth_var.dims)

    # Subsample the data using xarray before calling .values
    ensemble_values_sampled_da = subsample_da(forecast_var_stacked, max_samples)
    ground_truth_values_sampled_da = subsample_da(ground_truth_var_stacked, max_samples)

    ensemble_values_sampled = ensemble_values_sampled_da.values
    ground_truth_values_sampled = ground_truth_values_sampled_da.values

    # Fit KDEs
    ensemble_kde = gaussian_kde(ensemble_values_sampled)
    ground_truth_kde = gaussian_kde(ground_truth_values_sampled)

    # Calculate combined min and max for normalization
    combined_min = min(ensemble_values_sampled.min(), ground_truth_values_sampled.min())
    combined_max = max(ensemble_values_sampled.max(), ground_truth_values_sampled.max())
    x = np.linspace(combined_min, combined_max, 1000)

    # Normalize data before calculating Wasserstein distance
    data_range = combined_max - combined_min
    if data_range != 0:
        ensemble_values_norm = (ensemble_values_sampled - combined_min) / data_range
        ground_truth_values_norm = (
            ground_truth_values_sampled - combined_min
        ) / data_range
        w_distance_normalized = wasserstein_distance(
            ensemble_values_norm, ground_truth_values_norm
        )
    else:
        w_distance_normalized = 0

    # Evaluate PDFs
    ensemble_pdf = ensemble_kde(x)
    ground_truth_pdf = ground_truth_kde(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, ensemble_pdf, label="Ensemble Members", color=color_palette[1])
    plt.plot(x, ground_truth_pdf, label="Ground Truth", color=color_palette[0])
    plt.title(
        f"Density Distribution of Values for {variable}"
        f"{' at level ' + str(level) + ' hPa' if level else ''}\n"
        f"Region: {region}, Init Date: {date_time}, Model: {model_name}"
    )
    plt.xlabel(f'Value in {ground_truth_var.attrs.get("units", "unknown units")}')
    plt.ylabel("Density")
    plt.legend()

    # Add normalized Wasserstein distance to the plot
    plt.text(
        0.5,
        0.1,
        f"Normalized Wasserstein distance: {w_distance_normalized:.4f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.savefig(
        os.path.join(
            path_out,
            f"density_distribution_{variable}{'_' + str(level) if level else ''}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_combined_density_distribution(
    variable_level_pairs: List[Tuple[str, Optional[float]]],
    forecast: xr.Dataset,
    ground_truth: xr.Dataset,
    path_out: str,
    color_palette: List[str],
    model_name: str,
    region: str = "",
    date_time: str = "",
    max_samples: int = 100000,
):
    print(
        f"Creating combined density distribution plot for variable-level pairs: {variable_level_pairs}"
    )

    # Prepare data for each variable-level pair
    data_dict = {}
    for var, level in variable_level_pairs:
        print(f"Processing variable {var} at level {level}")
        if level is not None:
            forecast_var = forecast[var].sel(isobaricInhPa=level)
        else:
            forecast_var = forecast[var]

        # Stack all dimensions into a single dimension called 'all_points'
        forecast_var_stacked = forecast_var.stack(all_points=forecast_var.dims)

        # Subsample the data using our optimized function
        forecast_var_sampled = subsample_da(forecast_var_stacked, max_samples)

        # Get the sampled values
        values = forecast_var_sampled.values

        # Create a key name for the variable-level
        var_name = f"{var}_{level}hPa" if level is not None else var
        data_dict[var_name] = values

    # Create a DataFrame from the data_dict
    data_df = pd.DataFrame(data_dict)

    # Create pairplot
    sns.set(style="whitegrid")
    g = sns.PairGrid(data_df, diag_sharey=False)
    g.map_upper(sns.kdeplot, cmap="Blues_d")
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_diag(sns.kdeplot, lw=3)

    # Adjust titles and labels
    plt.subplots_adjust(top=0.9)
    title = f"Combined Density Distribution - {model_name}"
    if region:
        title += f" - {region}"
    if date_time:
        title += f" - {date_time}"
    plt.suptitle(title, y=0.95)

    filename = f"combined_density_distribution_{model_name}"
    if region:
        filename += f"_{region}"
    if date_time:
        filename += f"_{date_time}"
    filename += ".png"
    plt.savefig(os.path.join(path_out, filename), dpi=300, bbox_inches="tight")
    plt.close()


def prepare_density_distribution_args(
    data: Dict[str, xr.Dataset],
    vars_3d: List[str],
    vars_2d: List[str],
    config: Dict[str, Any],
    path_out: str,
    model_name: str,
    region: str,
    date_time: str,
    max_samples: int = 100000,
    max_variables: int = 5,
    combined: bool = False,
) -> List[Dict[str, Any]]:
    forecast_key = "forecast_ifs" if "forecast_ifs" in data else "forecast"
    forecast = data[forecast_key]
    ground_truth = data["ground_truth"]

    base_args = {
        "forecast": forecast,
        "ground_truth": ground_truth,
        "path_out": path_out,
        "color_palette": config["color_palette"],
        "model_name": model_name,
        "region": region,
        "date_time": date_time,
        "max_samples": max_samples,
    }

    args_list = []

    if combined:
        # Collect variable-level combinations
        variable_level_pairs = []
        for var in vars_2d:
            variable_level_pairs.append((var, None))
        for var in vars_3d:
            levels = forecast[var].coords["isobaricInhPa"].values
            for level in levels:
                variable_level_pairs.append((var, level))

        # Limit to max_variables combinations
        variable_level_pairs = variable_level_pairs[:max_variables]

        args_list.append(
            {
                **base_args,
                "variable_level_pairs": variable_level_pairs,
            }
        )
    else:
        # Prepare arguments for individual density distribution plots
        args_list.extend(
            [{**base_args, "variable": var, "level": None} for var in vars_2d]
        )
        for var in vars_3d:
            levels = forecast[var].coords["isobaricInhPa"].values
            args_list.extend(
                [{**base_args, "variable": var, "level": level} for level in levels]
            )

    return args_list


def main():
    args, config = parse_args()
    data = load_and_prepare_data(
        os.path.join(args.out_dir, str(args.date_time), args.model_name),
        config["selected_vars"],
        args.crop_region,
        args.model_name,
        args.perturbation_init,
        args.perturbation_latent,
        args.layer,
        args.members,
        debug_mode=args.debug,
    )

    path_out = os.path.join(
        args.out_dir,
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}_layer_{args.layer}",
        args.crop_region,
        f"png_{args.model_name}",
    )
    os.makedirs(path_out, exist_ok=True)

    variables = list(data["forecast"].data_vars)
    vars_3d = [
        var for var in variables if "isobaricInhPa" in data["forecast"][var].dims
    ]
    vars_2d = [
        var for var in variables if "isobaricInhPa" not in data["forecast"][var].dims
    ]

    # Prepare arguments for individual density distribution plots
    individual_plot_args = prepare_density_distribution_args(
        data,
        vars_3d,
        vars_2d,
        config,
        path_out,
        args.model_name,
        args.crop_region,
        args.date_time,
        max_samples=config["sample_size"],
        combined=False,
    )

    # Plot individual density distributions
    for plot_arg in individual_plot_args:
        plot_density_distribution(**plot_arg)

    # Prepare arguments for combined density distribution plots
    combined_plot_args = prepare_density_distribution_args(
        data,
        vars_3d,
        vars_2d,
        config,
        path_out,
        args.model_name,
        args.crop_region,
        args.date_time,
        max_samples=config["sample_size"],
        max_variables=5,
        combined=True,
    )

    # Plot combined density distributions
    for plot_arg in combined_plot_args:
        plot_combined_density_distribution(**plot_arg)


if __name__ == "__main__":
    main()
