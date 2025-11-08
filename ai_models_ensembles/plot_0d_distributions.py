from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.stats import gaussian_kde, wasserstein_distance

from ai_models_ensembles.utils import (
    build_output_filename,
    ensure_dir,
    save_dataframe,
    save_npz,
)

__all__ = [
    "subsample_da",
    "plot_density_distribution",
    "plot_combined_density_distribution",
    "plot_latitude_band_histograms",
    "plot_histogram_global_grid",
    "plot_kde_evolution_ridgeline",
    "plot_pit_histogram",
    "plot_pit_histogram_by_lead",
    "prepare_density_distribution_args",
]


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


def _lat_bands() -> tuple[np.ndarray, int, int]:
    lat_bins = np.arange(-90, 91, 10)
    n_bands = len(lat_bins) - 1
    n_rows = n_bands // 2
    return lat_bins, n_bands, n_rows


def _choose_edges_from_samples(
    a_true: np.ndarray, a_pred: np.ndarray, bins: int = 400
) -> np.ndarray:
    if a_true.size == 0 or a_pred.size == 0:
        return np.linspace(-1.0, 1.0, bins + 1)
    both = np.concatenate([a_true, a_pred])
    try:
        qlow, qhigh = np.quantile(both, [0.001, 0.999])
        if not np.isfinite(qlow) or not np.isfinite(qhigh) or qlow == qhigh:
            qlow, qhigh = -1.0, 1.0
    except Exception:
        qlow, qhigh = -1.0, 1.0
    return np.linspace(qlow, qhigh, bins + 1)


def _resolve_io(
    path_out: str, artifact_root: str | Path | None, metric: str, output_mode: str
) -> tuple[Path, Optional[Path], bool, bool]:
    mode = (output_mode or "plot").lower()
    save_fig = mode in {"plot", "both"}
    save_data = mode in {"both", "data", "npz"}
    fig_dir = ensure_dir(path_out)
    data_dir: Optional[Path] = None
    if save_data:
        data_root = (
            Path(artifact_root) if artifact_root is not None else fig_dir / "data"
        )
        data_dir = ensure_dir(data_root / metric)
    return fig_dir, data_dir, save_fig, save_data


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
    artifact_root: Optional[str | Path] = None,
    output_mode: str = "both",
    ensemble: Optional[str | int] = None,
) -> None:
    print(
        f"Creating density distribution plot for variable: {variable}, level: {level}"
    )

    mode = (output_mode or "plot").lower()
    save_fig = mode in {"plot", "both"}
    save_data = mode in {"both", "data", "npz"}

    figure_dir = ensure_dir(path_out)
    data_dir: Optional[Path] = None
    if save_data:
        data_root = (
            Path(artifact_root) if artifact_root is not None else figure_dir / "data"
        )
        data_dir = ensure_dir(data_root / "density")

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

    filename_core_args = dict(
        metric="density",
        variable=variable,
        level=level,
        qualifier="pdf",
        ensemble=ensemble,
    )

    if save_data and data_dir is not None:
        payload = {
            "x": x,
            "ensemble_pdf": ensemble_pdf,
            "ground_truth_pdf": ground_truth_pdf,
            "ensemble_sample": ensemble_values_sampled,
            "ground_truth_sample": ground_truth_values_sampled,
            "wasserstein_normalized": np.array([w_distance_normalized], dtype=float),
            "variable": np.array([variable]),
            "level": np.array(
                [level if level is not None else "surface"], dtype=object
            ),
            "region": np.array([region], dtype=object),
            "date_time": np.array([date_time], dtype=object),
            "model_name": np.array([model_name], dtype=object),
        }
        data_filename = build_output_filename(ext="npz", **filename_core_args)
        save_npz(payload, data_dir, data_filename)

    plt.figure(figsize=(10, 6))
    plt.plot(x, ensemble_pdf, label="Ensemble Members", color=color_palette[1])
    plt.plot(x, ground_truth_pdf, label="Ground Truth", color=color_palette[0])
    plt.title(
        f"Density Distribution of Values for {variable}"
        f"{' at level ' + str(level) + ' hPa' if level else ''}\n"
        f"Region: {region}, Init Date: {date_time}, Model: {model_name}"
    )
    plt.xlabel(f"Value in {ground_truth_var.attrs.get('units', 'unknown units')}")
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

    if save_fig:
        fig_filename = build_output_filename(ext="png", **filename_core_args)
        plt.savefig(figure_dir / fig_filename, dpi=300, bbox_inches="tight")
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
    artifact_root: str | Path | None = None,
    output_mode: str = "both",
    ensemble: str | int | None = None,
) -> None:
    print(
        f"Creating combined density distribution plot for variable-level pairs: {variable_level_pairs}"
    )

    mode = (output_mode or "plot").lower()
    save_fig = mode in {"plot", "both"}
    save_data = mode in {"both", "data", "npz"}

    figure_dir = ensure_dir(path_out)
    data_dir: Optional[Path] = None
    if save_data:
        data_root = (
            Path(artifact_root) if artifact_root is not None else figure_dir / "data"
        )
        data_dir = ensure_dir(data_root / "density")

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

    filename_args = dict(
        metric="density",
        variable="combined",
        qualifier="pairgrid",
        ensemble=ensemble,
    )

    if save_data and data_dir is not None:
        data_filename = build_output_filename(ext="csv", **filename_args)
        save_dataframe(data_df, data_dir, data_filename)

    if save_fig:
        fig_filename = build_output_filename(ext="png", **filename_args)
        plt.savefig(figure_dir / fig_filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_latitude_band_histograms(
    variable: str,
    forecast: xr.Dataset,
    ground_truth: xr.Dataset,
    path_out: str,
    color_palette: List[str],
    model_name: str,
    level: Optional[float] = None,
    max_samples: int = 200000,
    output_mode: str = "both",
    artifact_root: str | Path | None = None,
    ensemble: str | int | None = None,
) -> None:
    """Plot histograms by latitude bands comparing ground truth vs forecast.

    Saves PNG and NPZ with per-band densities and bin edges.
    """
    print(f"Creating latitude-band histograms for {variable}, level={level}")
    fig_dir, data_dir, save_fig, save_data = _resolve_io(
        path_out, artifact_root, "histograms", output_mode
    )

    da_true = (
        ground_truth[variable].sel(isobaricInhPa=level)
        if level is not None
        else ground_truth[variable]
    )
    da_pred = (
        forecast[variable].sel(isobaricInhPa=level)
        if level is not None
        else forecast[variable]
    )

    lat_bins, n_bands, n_rows = _lat_bands()
    fig, axs = plt.subplots(
        n_rows, 2, figsize=(16, 3 * n_rows), constrained_layout=True
    )

    combined_npz: Dict[str, list] = {
        "neg_counts": [],
        "neg_bins": [],
        "pos_counts": [],
        "pos_bins": [],
        "neg_lat_min": [],
        "neg_lat_max": [],
        "pos_lat_min": [],
        "pos_lat_max": [],
    }
    global_x_min: Optional[float] = None
    global_x_max: Optional[float] = None
    global_y_max: float = 0.0

    # Southern hemisphere (right column)
    for j in range(n_bands // 2):
        lat_max = lat_bins[j]
        lat_min = lat_bins[j + 1]
        t_sel = da_true.sel(latitude=slice(lat_min, lat_max))
        p_sel = da_pred.sel(latitude=slice(lat_min, lat_max))
        a_true = subsample_da(t_sel.stack(all_points=t_sel.dims), max_samples).values
        a_pred = subsample_da(p_sel.stack(all_points=p_sel.dims), max_samples).values
        edges = _choose_edges_from_samples(a_true, a_pred, bins=400)
        ct, _ = np.histogram(a_true, bins=edges)
        cp, _ = np.histogram(a_pred, bins=edges)
        width = np.diff(edges)
        area_t = ct.sum() * width.mean() if ct.sum() > 0 else 1.0
        area_p = cp.sum() * width.mean() if cp.sum() > 0 else 1.0
        dt = ct / area_t
        dp = cp / area_p
        ax = axs[j, 1]
        ax.bar(
            edges[:-1],
            dt,
            width=width,
            align="edge",
            alpha=0.5,
            color="skyblue",
            label="Ground Truth",
        )
        ax.bar(
            edges[:-1],
            dp,
            width=width,
            align="edge",
            alpha=0.5,
            color="salmon",
            label=f"{model_name}",
        )
        ax.set_title(f"Lat {lat_min}° to {lat_max}°")
        if global_x_min is None or float(edges[0]) < global_x_min:
            global_x_min = float(edges[0])
        if global_x_max is None or float(edges[-1]) > global_x_max:
            global_x_max = float(edges[-1])
        local_y = float(
            max(np.nanmax(dt) if dt.size else 0.0, np.nanmax(dp) if dp.size else 0.0)
        )
        global_y_max = max(global_y_max, local_y)
        ax.legend(loc="upper right")
        if save_data and data_dir is not None:
            combined_npz["neg_counts"].append((dt, dp))
            combined_npz["neg_bins"].append(edges)
            combined_npz["neg_lat_min"].append(float(lat_min))
            combined_npz["neg_lat_max"].append(float(lat_max))

    # Northern hemisphere (left column)
    for j in range(n_bands // 2):
        idx = -(j + 1)
        lat_max = lat_bins[idx - 1]
        lat_min = lat_bins[idx]
        t_sel = da_true.sel(latitude=slice(lat_min, lat_max))
        p_sel = da_pred.sel(latitude=slice(lat_min, lat_max))
        a_true = subsample_da(t_sel.stack(all_points=t_sel.dims), max_samples).values
        a_pred = subsample_da(p_sel.stack(all_points=p_sel.dims), max_samples).values
        edges = _choose_edges_from_samples(a_true, a_pred, bins=400)
        ct, _ = np.histogram(a_true, bins=edges)
        cp, _ = np.histogram(a_pred, bins=edges)
        width = np.diff(edges)
        area_t = ct.sum() * width.mean() if ct.sum() > 0 else 1.0
        area_p = cp.sum() * width.mean() if cp.sum() > 0 else 1.0
        dt = ct / area_t
        dp = cp / area_p
        ax = axs[j, 0]
        ax.bar(
            edges[:-1],
            dt,
            width=width,
            align="edge",
            alpha=0.5,
            color="skyblue",
            label="Ground Truth",
        )
        ax.bar(
            edges[:-1],
            dp,
            width=width,
            align="edge",
            alpha=0.5,
            color="salmon",
            label=f"{model_name}",
        )
        ax.set_title(f"Lat {lat_min}° to {lat_max}°")
        if global_x_min is None or float(edges[0]) < global_x_min:
            global_x_min = float(edges[0])
        if global_x_max is None or float(edges[-1]) > global_x_max:
            global_x_max = float(edges[-1])
        local_y = float(
            max(np.nanmax(dt) if dt.size else 0.0, np.nanmax(dp) if dp.size else 0.0)
        )
        global_y_max = max(global_y_max, local_y)
        ax.legend(loc="upper right")
        if save_data and data_dir is not None:
            combined_npz["pos_counts"].append((dt, dp))
            combined_npz["pos_bins"].append(edges)
            combined_npz["pos_lat_min"].append(float(lat_min))
            combined_npz["pos_lat_max"].append(float(lat_max))

    # unify axes
    if global_x_min is not None and global_x_max is not None:
        for j in range(n_bands // 2):
            axs[j, 0].set_xlim(global_x_min, global_x_max)
            axs[j, 1].set_xlim(global_x_min, global_x_max)
            axs[j, 0].set_ylim(0.0, global_y_max * 1.05 if global_y_max > 0 else 1.0)
            axs[j, 1].set_ylim(0.0, global_y_max * 1.05 if global_y_max > 0 else 1.0)

    units = ground_truth[variable].attrs.get("units", "")
    plt.suptitle(f"Distribution of {variable} ({units}) by latitude bands", y=1.02)

    filename_args = dict(
        metric="hist",
        variable=variable,
        level=level,
        qualifier="latbands",
        ensemble=ensemble,
    )
    if save_fig:
        fig_filename = build_output_filename(ext="png", **filename_args)
        plt.savefig(fig_dir / fig_filename, dpi=300, bbox_inches="tight")
    if save_data and data_dir is not None:
        data_filename = build_output_filename(ext="npz", **filename_args)
        np.savez(
            data_dir / data_filename,
            neg_counts=np.array(combined_npz["neg_counts"], dtype=object),
            neg_bins=np.array(combined_npz["neg_bins"], dtype=object),
            pos_counts=np.array(combined_npz["pos_counts"], dtype=object),
            pos_bins=np.array(combined_npz["pos_bins"], dtype=object),
            neg_lat_min=np.array(combined_npz["neg_lat_min"]),
            neg_lat_max=np.array(combined_npz["neg_lat_max"]),
            pos_lat_min=np.array(combined_npz["pos_lat_min"]),
            pos_lat_max=np.array(combined_npz["pos_lat_max"]),
            allow_pickle=True,
        )
    plt.close(fig)


def plot_histogram_global_grid(
    variable: str,
    forecast: xr.Dataset,
    ground_truth: xr.Dataset,
    path_out: str,
    color_palette: List[str],
    model_name: str,
    level: Optional[float] = None,
    output_mode: str = "both",
    artifact_root: str | Path | None = None,
    ensemble: str | int | None = None,
) -> None:
    """Plot a grid of global histograms across all lead steps (6h spacing)."""
    print(f"Creating global histogram grid for {variable}, level={level}")
    fig_dir, data_dir, save_fig, save_data = _resolve_io(
        path_out, artifact_root, "histograms", output_mode
    )

    da_true = (
        ground_truth[variable].sel(isobaricInhPa=level)
        if level is not None
        else ground_truth[variable]
    )
    da_pred = (
        forecast[variable].sel(isobaricInhPa=level)
        if level is not None
        else forecast[variable]
    )
    n_steps = int(da_pred.sizes.get("step", 0))
    if n_steps == 0:
        print("[histogram_global_grid] No step dimension; skipping")
        return
    hours = [int(i * 6) for i in range(n_steps)]
    ncols = 2
    nrows = (n_steps + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, max(2.5, 2.2 * nrows)), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    # Pre-compute global ranges to standardize axes
    panel_results = []  # (edges, dens_true, dens_pred, hour)
    xmins, xmaxs, ymaxs = [], [], []
    for i in range(n_steps):
        t_i = da_true.isel(step=i) if "step" in da_true.dims else da_true
        p_i = da_pred.isel(step=i) if "step" in da_pred.dims else da_pred
        a_true = subsample_da(t_i.stack(all_points=t_i.dims), 200000).values
        a_pred = subsample_da(p_i.stack(all_points=p_i.dims), 200000).values
        edges = _choose_edges_from_samples(a_true, a_pred, bins=400)
        ct, _ = np.histogram(a_true, bins=edges)
        cp, _ = np.histogram(a_pred, bins=edges)
        width = np.diff(edges)
        area_t = ct.sum() * width.mean() if ct.sum() > 0 else 1.0
        area_p = cp.sum() * width.mean() if cp.sum() > 0 else 1.0
        dt = ct / area_t
        dp = cp / area_p
        panel_results.append((edges, dt, dp, hours[i]))
        xmins.append(edges[0])
        xmaxs.append(edges[-1])
        ymaxs.append(
            float(
                max(
                    np.nanmax(dt) if dt.size else 0.0, np.nanmax(dp) if dp.size else 0.0
                )
            )
        )

    x_min = float(min(xmins)) if xmins else -1.0
    x_max = float(max(xmaxs)) if xmaxs else 1.0
    y_max = float(max(ymaxs)) if ymaxs else 1.0

    for i, (edges, dt, dp, h) in enumerate(panel_results):
        ax = axes[i]
        width = np.diff(edges)
        ax.bar(
            edges[:-1],
            dt,
            width=width,
            align="edge",
            alpha=0.5,
            color="skyblue",
            label="Ground Truth" if i == 0 else None,
        )
        ax.bar(
            edges[:-1],
            dp,
            width=width,
            align="edge",
            alpha=0.5,
            color="salmon",
            label=f"{model_name}" if i == 0 else None,
        )
        ax.set_title(f"Lead {int(h)}h", fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_max)
        if i % ncols != 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("Density")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(variable)
        else:
            ax.tick_params(axis="x", labelbottom=False)
    if axes.size > len(panel_results):
        for j in range(len(panel_results), axes.size):
            axes[j].axis("off")

    axes[0].legend(loc="upper right")
    filename_args = dict(
        metric="hist_global",
        variable=variable,
        level=level,
        qualifier="grid",
        ensemble=ensemble,
    )
    if save_fig:
        fig_filename = build_output_filename(ext="png", **filename_args)
        plt.savefig(fig_dir / fig_filename, dpi=300, bbox_inches="tight")
    if save_data and data_dir is not None:
        data_filename = build_output_filename(
            ext="npz", **filename_args | {"qualifier": "grid_data"}
        )
        np.savez(
            data_dir / data_filename,
            lead_hours=np.array(hours, dtype=float),
            densities_true=np.array([pr[1] for pr in panel_results], dtype=object),
            densities_pred=np.array([pr[2] for pr in panel_results], dtype=object),
            edges=np.array([pr[0] for pr in panel_results], dtype=object),
            allow_pickle=True,
        )
    plt.close(fig)


def plot_kde_evolution_ridgeline(
    variable: str,
    forecast: xr.Dataset,
    ground_truth: xr.Dataset,
    path_out: str,
    model_name: str,
    level: Optional[float] = None,
    output_mode: str = "both",
    artifact_root: str | Path | None = None,
    ensemble: str | int | None = None,
    y_points: int = 200,
) -> None:
    """Global KDE evolution over lead time (ridgeline): filled=model, line=target.

    Saves PNG and NPZ with (lead_hours, y_eval, density_target, density_model).
    """
    print(f"Creating KDE ridgeline evolution for {variable}, level={level}")
    fig_dir, data_dir, save_fig, save_data = _resolve_io(
        path_out, artifact_root, "wd_kde", output_mode
    )

    da_t_all = (
        ground_truth[variable].sel(isobaricInhPa=level)
        if level is not None
        else ground_truth[variable]
    )
    da_p_all = (
        forecast[variable].sel(isobaricInhPa=level)
        if level is not None
        else forecast[variable]
    )

    # Evaluate y-range from robust quantiles across all data
    both = xr.concat([da_t_all, da_p_all], dim="_tmp")
    q = both.quantile([0.001, 0.999], skipna=True).compute()
    vmin = float(q.isel(quantile=0).item()) if np.isfinite(q.isel(quantile=0)) else -3.0
    vmax = float(q.isel(quantile=1).item()) if np.isfinite(q.isel(quantile=1)) else 3.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -3.0, 3.0
    y_eval = np.linspace(vmin, vmax, y_points)

    n_steps = int(da_p_all.sizes.get("step", 0))
    if n_steps == 0:
        n_steps = 1
    lead_hours = [int(i * 6) for i in range(n_steps)]
    Z_t = []
    Z_p = []

    def _eval_kde(arr: np.ndarray) -> np.ndarray:
        arr = arr[np.isfinite(arr)]
        if arr.size < 10:
            return np.zeros_like(y_eval)
        kde = gaussian_kde(arr)
        return kde(y_eval)

    for i in range(n_steps):
        da_t = da_t_all.isel(step=i) if "step" in da_t_all.dims else da_t_all
        da_p = da_p_all.isel(step=i) if "step" in da_p_all.dims else da_p_all
        a_t = da_t.stack(all_points=da_t.dims).values
        a_p = da_p.stack(all_points=da_p.dims).values
        Z_t.append(_eval_kde(a_t))
        Z_p.append(_eval_kde(a_p))

    Z_t_arr = np.asarray(Z_t)
    Z_p_arr = np.asarray(Z_p)

    # Ridgeline figure
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    offset = (
        1.05 * max(float(np.max(Z_t_arr)), float(np.max(Z_p_arr)))
        if Z_t_arr.size
        else 1.0
    )
    cmap = plt.cm.viridis
    for i, h in enumerate(lead_hours):
        color = cmap(i / max(1, len(lead_hours) - 1))
        y_target = i * offset + Z_t_arr[i]
        y_model = i * offset + Z_p_arr[i]
        ax.fill_between(
            y_eval, i * offset, y_model, color=color, alpha=0.55, linewidth=0.0
        )
        ax.plot(y_eval, y_target, color="black", lw=0.7)
        ax.text(
            y_eval[-1] + (y_eval[1] - y_eval[0]) * 0.5,
            i * offset + 0.02,
            f"{int(h)}h",
            fontsize=8,
        )
    ax.set_yticks([])
    ax.set_xlabel(
        f"{variable}{' @ ' + str(level) + ' hPa' if level is not None else ''}"
    )
    ax.set_title("Global KDE evolution — ridgeline (filled=model, line=target)")

    filename_args = dict(
        metric="wd_kde_evolve",
        variable=variable,
        level=level,
        qualifier="ridgeline",
        ensemble=ensemble,
    )
    if save_fig:
        fig_filename = build_output_filename(ext="png", **filename_args)
        plt.savefig(fig_dir / fig_filename, dpi=300, bbox_inches="tight")
    if save_data and data_dir is not None:
        data_filename = build_output_filename(
            ext="npz", **(filename_args | {"qualifier": "ridgeline_data"})
        )
        np.savez(
            data_dir / data_filename,
            lead_hours=np.array(lead_hours, dtype=float),
            y_eval=y_eval,
            density_target=Z_t_arr,
            density_model=Z_p_arr,
        )
    plt.close(fig)


def _compute_pit(target_vals: np.ndarray, ens_vals: np.ndarray) -> np.ndarray:
    """Probability integral transform: fraction of ensemble members below target.

    target_vals: shape (N,)
    ens_vals: shape (N, M) broadcast-compatible; if flattened, reshape externally.
    Returns PIT values in [0,1].
    """
    if target_vals.size == 0 or ens_vals.size == 0:
        return np.array([], dtype=float)
    return np.mean(ens_vals < target_vals[:, None], axis=1)


def plot_pit_histogram(
    variable: str,
    forecast: xr.Dataset,
    ground_truth: xr.Dataset,
    path_out: str,
    model_name: str,
    level: Optional[float] = None,
    output_mode: str = "both",
    artifact_root: str | Path | None = None,
    ensemble: str | int | None = None,
    bins: int = 20,
) -> None:
    """Global PIT histogram (forecast ensemble vs ground truth) with PNG + NPZ.

    Follows naming metric="pit_hist"; NPZ keys: counts, edges, variable.
    """
    print(f"Creating PIT histogram for {variable}, level={level}")
    fig_dir, data_dir, save_fig, save_data = _resolve_io(
        path_out, artifact_root, "pit", output_mode
    )

    da_p = (
        forecast[variable].sel(isobaricInhPa=level)
        if level is not None
        else forecast[variable]
    )
    da_t = (
        ground_truth[variable].sel(isobaricInhPa=level)
        if level is not None
        else ground_truth[variable]
    )
    if "member" not in da_p.dims:
        print("[pit] forecast lacks 'member' dimension; skipping PIT histogram.")
        return
    # Stack non-member dims
    other_dims = [d for d in da_p.dims if d != "member"]
    da_p_flat = da_p.stack(sample=other_dims).transpose("sample", "member")
    da_t_flat = da_t.stack(sample=da_t.dims)
    # Align lengths (drop any mismatch due to broadcasting differences)
    n = min(da_p_flat.sample.size, da_t_flat.sample.size)
    ens_vals = da_p_flat.isel(sample=slice(0, n)).values  # shape (N,M)
    tgt_vals = da_t_flat.isel(sample=slice(0, n)).values  # shape (N,)
    pit_vals = _compute_pit(tgt_vals, ens_vals)
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(pit_vals, bins=edges)
    width = np.diff(edges)
    total = counts.sum()
    dens = counts / (total * width.mean()) if total > 0 else counts
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(
        edges[:-1], dens, width=width, align="edge", color="#4C78A8", edgecolor="white"
    )
    ax.axhline(1.0, color="brown", linestyle="--", linewidth=1)
    ax.set_title(
        f"PIT histogram — {variable}{' @ ' + str(level) + ' hPa' if level is not None else ''}"
    )
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    filename_args = dict(
        metric="pit_hist",
        variable=variable,
        level=level,
        qualifier=None,
        ensemble=ensemble,
    )
    if save_fig:
        fig_filename = build_output_filename(ext="png", **filename_args)
        plt.savefig(fig_dir / fig_filename, dpi=300, bbox_inches="tight")
    if save_data and data_dir is not None:
        data_filename = build_output_filename(ext="npz", **filename_args)
        np.savez(
            data_dir / data_filename,
            counts=counts.astype(float),
            edges=edges.astype(float),
            variable=np.array([variable]),
        )
    plt.close(fig)


def plot_pit_histogram_by_lead(
    variable: str,
    forecast: xr.Dataset,
    ground_truth: xr.Dataset,
    path_out: str,
    model_name: str,
    level: Optional[float] = None,
    output_mode: str = "both",
    artifact_root: str | Path | None = None,
    ensemble: str | int | None = None,
    bins: int = 20,
    panel_cols: int = 2,
) -> None:
    """PIT histograms per lead (step) in a panel grid. Saves PNG + NPZ line summary.

    NPZ (qualifier uniform_diff_by_lead_data) keys: lead_hours, uniform_diff, variable.
    """
    print(f"Creating PIT per-lead panel for {variable}, level={level}")
    fig_dir, data_dir, save_fig, save_data = _resolve_io(
        path_out, artifact_root, "pit", output_mode
    )
    da_p_all = (
        forecast[variable].sel(isobaricInhPa=level)
        if level is not None
        else forecast[variable]
    )
    da_t_all = (
        ground_truth[variable].sel(isobaricInhPa=level)
        if level is not None
        else ground_truth[variable]
    )
    if "member" not in da_p_all.dims or "step" not in da_p_all.dims:
        print("[pit] missing 'member' or 'step' dims; skipping per-lead PIT panels.")
        return
    n_steps = int(da_p_all.sizes.get("step", 0))
    hours = [int(i * 6) for i in range(n_steps)]
    ncols = panel_cols
    nrows = (n_steps + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 3.0 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    uniform_rows = []
    for i in range(n_steps):
        da_p = da_p_all.isel(step=i)
        da_t = da_t_all.isel(step=i) if "step" in da_t_all.dims else da_t_all
        other_dims = [d for d in da_p.dims if d not in ("member",)]
        da_p_flat = da_p.stack(sample=other_dims).transpose("sample", "member")
        da_t_flat = da_t.stack(sample=da_t.dims)
        n = min(da_p_flat.sample.size, da_t_flat.sample.size)
        pit_vals = _compute_pit(
            da_t_flat.isel(sample=slice(0, n)).values,
            da_p_flat.isel(sample=slice(0, n)).values,
        )
        counts, _ = np.histogram(pit_vals, bins=edges)
        width = np.diff(edges)
        total = counts.sum()
        dens = counts / (total * width.mean()) if total > 0 else counts
        uniform_diff = float(np.nanmean(np.abs(dens - 1.0))) if dens.size else np.nan
        uniform_rows.append(
            {"lead_time_hours": float(hours[i]), "uniform_diff": uniform_diff}
        )
        ax = axes[i]
        ax.bar(
            edges[:-1],
            dens,
            width=width,
            align="edge",
            color="#4C78A8",
            edgecolor="white",
        )
        ax.axhline(1.0, color="brown", linestyle="--", linewidth=1)
        ax.set_title(f"Lead {hours[i]}h", fontsize=11)
        if (i // ncols) == nrows - 1:
            ax.set_xlabel("PIT value")
        if (i % ncols) == 0:
            ax.set_ylabel("Density")
    if axes.size > n_steps:
        for j in range(n_steps, axes.size):
            axes[j].axis("off")
    plt.suptitle(f"PIT histograms per lead — {variable}", fontsize=14)
    filename_args = dict(
        metric="pit_hist",
        variable=variable,
        level=level,
        qualifier="grid",
        ensemble=ensemble,
    )
    if save_fig:
        fig_filename = build_output_filename(ext="png", **filename_args)
        plt.savefig(fig_dir / fig_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Line summary artifact
    if save_data and data_dir is not None and uniform_rows:
        import pandas as _pd

        df = _pd.DataFrame(uniform_rows).sort_values("lead_time_hours")
        data_filename = build_output_filename(
            ext="npz",
            metric="pit_hist",
            variable=variable,
            level=level,
            qualifier="uniform_diff_by_lead_data",
            ensemble=ensemble,
        )
        np.savez(
            data_dir / data_filename,
            lead_hours=df["lead_time_hours"].values.astype(float),
            uniform_diff=df["uniform_diff"].values.astype(float),
            variable=np.array([variable]),
        )


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
    artifact_root: str | Path | None = None,
    output_mode: str = "both",
    ensemble: str | int | None = None,
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
        "artifact_root": artifact_root,
        "output_mode": output_mode,
        "ensemble": ensemble,
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
