import argparse

import dask
import numpy as np
import scores
import seaborn as sns
import xarray as xr

dask.config.set(**{"array.slicing.split_large_chunks": True})


def parse_args():
    """
    Parse the command line arguments.

    Returns:
        tuple: A tuple containing the arguments and the configuration.
            args (argparse.Namespace): The parsed arguments.
            config (dict): The configuration.
    """

    parser = argparse.ArgumentParser(description="Evaluate the NeurWP Ensemble.")
    parser.add_argument("out_dir", type=str, help="The output directory")
    parser.add_argument(
        "date_time", type=str, help="Date and time in the format YYYYMMDDHHMM"
    )
    parser.add_argument("model_name", type=str, help="The ai-model name")
    parser.add_argument(
        "perturbation_init", type=float, help="The init perturbation size"
    )
    parser.add_argument(
        "perturbation_latent", type=float, help="The latent perturbation size"
    )
    parser.add_argument("layer", type=int, help="The layer to evaluate")
    parser.add_argument("members", type=int, help="The number of ensemble members")
    parser.add_argument("crop_region", type=str, help="The region to crop the data to")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode",
        default=False,
    )

    args = parser.parse_args()

    config = {
        "color_palette": sns.color_palette(
            ["#f75b78", "#6495ed", "#0e2d75", "#f9c740", "#45b7aa", "#353434"]
        ),
        # TODO: make this a user INPUT
        "sample_size": 100000,
        "selected_vars": ["u10"],
    }

    return args, config


def load_and_prepare_data(
    path_in,
    selected_vars,
    crop_region,
    model_name,
    perturb_init,
    perturb_latent,
    layer,
    num_members,
    debug_mode=False,
):
    """
    Load the data and prepare it for evaluation.

    Args:
        path_in (str): The path to the input data.
        selected_vars (list): The variables to select.
        crop_region (str): The region to crop the data to.
        model_name (str): The name of the model.
        perturb_init (float): The initial perturbation size.
        perturb_latent (float): The latent perturbation size.
        num_members (int): The number of members to select.
        debug_mode (bool): Whether to run in debug mode.

    Returns:
        dict: A dictionary containing
            ground truth (xr.Dataset): The ground truth data.
            forecast (xr.Dataset): The forecast data.
            forecast_unperturbed (xr.Dataset): The unperturbed forecast data.
            forecast_ifs (xr.Dataset): The IFS forecast data.
            forecast_ifs_unperturbed (xr.Dataset): The unperturbed IFS forecast data.
    """

    ground_truth = xr.open_zarr(f"{path_in}/ground_truth.zarr", consolidated=True).isel(
        number=0, surface=0
    )
    forecast = xr.open_zarr(
        f"{path_in}/init_{perturb_init}_latent_{perturb_latent}_layer_{layer}/forecast.zarr",
        consolidated=True,
    )
    forecast_unperturbed = xr.open_zarr(f"{path_in}/forecast.zarr", consolidated=True)
    forecast_ifs = xr.open_zarr(f"{path_in}/ifs_ens.zarr", consolidated=True)
    forecast_ifs_unperturbed = xr.open_zarr(
        f"{path_in}/ifs_control.zarr", consolidated=True
    )

    if crop_region == "europe":
        lat_min, lat_max = 25, 80
        lon_min, lon_max = 340, 50
        # Use modulo arithmetic to ensure longitudes are in 0-360 range
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        # Create a list of longitudes that wraps around 0/360
        lats = list(range(lat_min, lat_max + 1))
        lons = list(range(lon_min, 360)) + list(range(0, lon_max + 1))
    else:
        lats = ground_truth.latitude.isel(latitude=slice(1, -1)).values
        lons = ground_truth.longitude.isel(longitude=slice(1, -1)).values

    # Crop all datasets to the lat-lon box
    ground_truth = ground_truth.sel(latitude=lats, longitude=lons)
    forecast = forecast.sel(latitude=lats, longitude=lons)
    forecast_unperturbed = forecast_unperturbed.sel(latitude=lats, longitude=lons)
    forecast_ifs = forecast_ifs.sel(latitude=lats, longitude=lons)
    forecast_ifs_unperturbed = forecast_ifs_unperturbed.sel(
        latitude=lats, longitude=lons
    )

    # align ifs forecast with forecast dimensions
    forecast_ifs = (
        forecast_ifs.rename({"number": "member"})
        .set_index(member="member")
        .isel(surface=0, time=0)
    )

    # Select the appropriate members in all datasets
    forecast_ifs["member"] = forecast_ifs["member"] - 1
    forecast_unperturbed = forecast_unperturbed.sel(member=0)
    forecast_ifs_unperturbed = forecast_ifs_unperturbed.isel(
        number=0, surface=0, time=0
    )
    indices = np.random.permutation(forecast.member.size)
    indices_ifs = np.random.permutation(forecast_ifs.member.size)
    selected_indices = indices[:num_members]
    selected_indices_ifs = indices_ifs[:num_members]
    forecast = forecast.isel(member=selected_indices)
    forecast_ifs = forecast_ifs.isel(member=selected_indices_ifs)
    forecast = forecast.transpose(..., "member")
    forecast_ifs = forecast_ifs.transpose(..., "member")

    # Ensure all datasets have the same step coordinates
    # For plotting the easiest approach is conversion to float
    # For ground_truth, step and time have the reversed meaning
    ground_truth = ground_truth.isel(step=0)
    step_values = np.int64(forecast.step.values) / 1e9 / 3600
    ground_truth["step"] = ("time", np.arange(len(ground_truth["time"])))
    ground_truth = ground_truth.swap_dims({"time": "step"})
    ground_truth["step"] = step_values
    forecast["step"] = step_values
    forecast_unperturbed["step"] = step_values
    forecast_ifs["step"] = step_values
    forecast_ifs_unperturbed["step"] = step_values

    # Ensure all datasets have the same isobaricInhPa coordinates
    full_levels = forecast_ifs.isobaricInhPa.where(
        ~forecast_ifs.t.isnull()
        .any(dim=["latitude", "longitude", "step", "member"])
        .compute(),
        drop=True,
    ).values
    forecast = forecast.sel(isobaricInhPa=full_levels)
    forecast_ifs = forecast_ifs.sel(isobaricInhPa=full_levels)
    forecast_unperturbed = forecast_unperturbed.sel(isobaricInhPa=full_levels)
    forecast_ifs_unperturbed = forecast_ifs_unperturbed.sel(isobaricInhPa=full_levels)
    ground_truth = ground_truth.sel(isobaricInhPa=full_levels)

    # Select the data variables
    ground_truth = ground_truth[selected_vars]
    forecast = forecast[selected_vars]
    forecast_unperturbed = forecast_unperturbed[selected_vars]
    forecast_ifs = forecast_ifs[selected_vars]
    forecast_ifs_unperturbed = forecast_ifs_unperturbed[selected_vars]

    if debug_mode:
        max_members = 10
        max_time_steps = 5

        forecast = forecast.isel(member=slice(0, max_members))
        forecast_ifs = forecast_ifs.isel(member=slice(0, max_members))

        ground_truth = ground_truth.isel(step=slice(0, max_time_steps))
        forecast = forecast.isel(step=slice(0, max_time_steps))
        forecast_unperturbed = forecast_unperturbed.isel(step=slice(0, max_time_steps))
        forecast_ifs = forecast_ifs.isel(step=slice(0, max_time_steps))
        forecast_ifs_unperturbed = forecast_ifs_unperturbed.isel(
            step=slice(0, max_time_steps)
        )

    for ds_name in [
        "ground_truth",
        "forecast",
        "forecast_unperturbed",
        "forecast_ifs",
        "forecast_ifs_unperturbed",
    ]:
        ds = locals()[ds_name]

    return {
        "ground_truth": ground_truth,
        "forecast": forecast,
        "forecast_unperturbed": forecast_unperturbed,
        "forecast_ifs": forecast_ifs,
        "forecast_ifs_unperturbed": forecast_ifs_unperturbed,
    }


def calculate_stats(ground_truth, forecast, forecast_unperturbed, crop_region):
    """
    Calculate the statistics for the evaluation.

    Args:
        ground_truth (xr.Dataset): The ground truth data.
        forecast (xr.Dataset): The forecast data.
        forecast_unperturbed (xr.Dataset): The unperturbed forecast data.

    Returns:
        dict: A dictionary containing the statistics.
    """
    fc_mean = forecast.mean(dim=["latitude", "longitude"])
    fc_mean_unperturbed = forecast_unperturbed.mean(dim=["latitude", "longitude"])
    gt_mean = ground_truth.mean(dim=["latitude", "longitude"])
    diff = forecast - ground_truth
    squared_diff = diff**2
    squared_diff_mean = (forecast.mean(dim="member") - ground_truth) ** 2
    rmse_grid = np.sqrt(squared_diff.mean(dim="member"))
    rmse = np.sqrt(squared_diff.mean(dim=["latitude", "longitude"]))
    rmse_mean = np.sqrt(squared_diff_mean.mean(dim=["latitude", "longitude"]))

    squared_diff_unperturbed = (forecast_unperturbed - ground_truth) ** 2
    rmse_unperturbed = np.sqrt(
        squared_diff_unperturbed.mean(dim=["latitude", "longitude"])
    )

    if crop_region == "europe":
        lat_min, lat_max = 25, 80
    else:
        lat_min, lat_max = ground_truth.latitude.isel(latitude=[0, -1]).values

    ensemble_spread_grid = forecast.std(dim="member")
    # Prevent division by zero
    spread_skill_ratio_grid = ensemble_spread_grid / (rmse_grid + 1e-6)
    spread_skill_ratio = spread_skill_ratio_grid.mean(dim=["latitude", "longitude"])
    ensemble_spread = ensemble_spread_grid.mean(dim=["latitude", "longitude"])

    # TODO remove this again after init from oper analysis
    spread_skill_ratio = spread_skill_ratio.isel(step=slice(1, None))
    ensemble_spread = ensemble_spread.isel(step=slice(1, None))

    preserve_dims = (
        ["step", "latitude", "longitude", "isobaricInhPa"]
        if "isobaricInhPa" in forecast.dims
        else ["step", "latitude", "longitude"]
    )
    # Now compute CRPS
    crps = scores.probability.crps_for_ensemble(
        forecast,
        ground_truth,
        ensemble_member_dim="member",
        method="ecdf",
        preserve_dims=preserve_dims,
    )

    # Detrending the data using the detrend function from scipy.signal to remove any
    # linear trends. Applying a window function (Hann window) to the detrended data
    # using the windows.hann function from scipy.signal to reduce spectral leakage.
    # Normalizing the power spectrum by dividing it by (n * np.sum(window ** 2)) to
    # account for the window function and the data size. Updating the wavenumber
    # calculation to convert from wavenumber to rad/km, assuming Earth's radius of 6371
    # km.
    def calculate_energy_spectra(
        forecast, forecast_unperturbed, ground_truth, lat_band
    ):
        EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers
        energy_spectra_forecast = []
        energy_spectra_unperturbed = []
        energy_spectra_ground_truth = []

        for data, energy_spectra_list in zip(
            [forecast, forecast_unperturbed, ground_truth],
            [
                energy_spectra_forecast,
                energy_spectra_unperturbed,
                energy_spectra_ground_truth,
            ],
        ):
            lat_slice = slice(lat_band[0], lat_band[1])
            data_lat_band = data.sel(latitude=lat_slice)

            # Select the last time step
            data_last_step = data_lat_band.isel(step=-1)

            for var in data_last_step.data_vars:
                var_data = data_last_step[var]

                # Check if 'isobaricInhPa' is a dimension
                if "isobaricInhPa" in var_data.dims:
                    levels = var_data.isobaricInhPa.values
                else:
                    levels = [0]

                power_spectra = []
                wavenumbers_list = []
                wavenumbers_earth_list = []
                for level in levels:
                    # Select data for the current level
                    data_level = (
                        var_data.sel(isobaricInhPa=level)
                        if "isobaricInhPa" in var_data.dims
                        else var_data
                    )

                    # Calculate mean over latitude band
                    latitudes = np.deg2rad(data_level.latitude.values)
                    cos_latitudes = np.cos(latitudes)

                    # Get the index of the longitude axis
                    n = data_level.longitude.size

                    # Initialize an array to accumulate the weighted power spectra
                    summed_power_spectrum = None

                    for i, lat in enumerate(latitudes):
                        # Extract data for the current latitude
                        data_lat = data_level.isel(latitude=i)
                        longitude_index = data_lat.get_axis_num("longitude")

                        # Calculate FFT along longitude
                        fft = np.fft.rfft(data_lat, axis=longitude_index)

                        # Calculate the circumference at this latitude (in km)
                        circumference = 2 * np.pi * EARTH_RADIUS_KM * cos_latitudes[i]
                        # Calculate the physical distance between longitude points (in km)
                        dx = circumference / n  # km between longitude points
                        wavenumber = np.fft.rfftfreq(n, d=dx) * dx * n  # in cycles/latitude band (around the earth)
                        wavenumbers_list.append(wavenumber)

                        # Calculate power spectrum (mathematical "Energy")
                        power_spectrum = np.abs(fft) ** 2

                        # Weight the power spectrum by cosine of latitude
                        weighted_power_spectrum = power_spectrum * cos_latitudes[i]

                        # Sum the weighted power spectra
                        if summed_power_spectrum is None:
                            summed_power_spectrum = weighted_power_spectrum
                        else:
                            summed_power_spectrum += weighted_power_spectrum

                    summed_power_spectrum /= np.sum(cos_latitudes)
                    power_spectra.append(summed_power_spectrum)
                    # Since wavenumbers vary with latitude, calculate an average
                    # wavenumber weighted by cos(lat) Convert the list of
                    # wavenumbers to a 2D array ([latitudes, wavenumbers])
                    average_wavenumber = np.average(np.array(wavenumbers_list), axis=0)

                if "isobaricInhPa" in var_data.dims:
                    if "member" in var_data.dims:
                        energy_spectra_list.append(
                            xr.Dataset(
                                {
                                    var: (
                                        ["isobaricInhPa", "wavenumber", "member"],
                                        np.array(power_spectra),
                                    ),
                                },
                                coords={
                                    "isobaricInhPa": ("isobaricInhPa", levels),
                                    "wavenumber": ("wavenumber", average_wavenumber),
                                    "member": ("member", data_last_step.member.values),
                                },
                            )
                        )
                    else:
                        energy_spectra_list.append(
                            xr.Dataset(
                                {
                                    var: (
                                        ["isobaricInhPa", "wavenumber"],
                                        np.array(power_spectra),
                                    ),
                                },
                                coords={
                                    "isobaricInhPa": ("isobaricInhPa", levels),
                                    "wavenumber": ("wavenumber", average_wavenumber),
                                },
                            )
                        )
                else:
                    if "member" in var_data.dims:
                        energy_spectra_list.append(
                            xr.Dataset(
                                {
                                    var: (
                                        ["wavenumber", "member"],
                                        np.squeeze(np.array(power_spectra)),
                                    ),
                                },
                                coords={
                                    "wavenumber": ("wavenumber", average_wavenumber),
                                    "member": ("member", data_last_step.member.values),
                                },
                            )
                        )
                    else:
                        energy_spectra_list.append(
                            xr.Dataset(
                                {
                                    var: (
                                        ["wavenumber"],
                                        np.squeeze(np.array(power_spectra)),
                                    ),
                                },
                                coords={
                                    "wavenumber": ("wavenumber", average_wavenumber),
                                },
                            )
                        )

        energy_spectra_forecast_concat = xr.merge(energy_spectra_forecast)
        energy_spectra_unperturbed_concat = xr.merge(energy_spectra_unperturbed)
        energy_spectra_ground_truth_concat = xr.merge(energy_spectra_ground_truth)

        return (
            energy_spectra_forecast_concat.drop_isel(wavenumber=0),
            energy_spectra_unperturbed_concat.drop_isel(wavenumber=0),
            energy_spectra_ground_truth_concat.drop_isel(wavenumber=0),
        )

    (
        energy_spectra_forecast,
        energy_spectra_unperturbed,
        energy_spectra_ground_truth,
    ) = calculate_energy_spectra(
        forecast,
        forecast_unperturbed,
        ground_truth,
        lat_band=(lat_min, lat_max),
    )

    return {
        "ts_fc_mean": fc_mean,
        "ts_fc_mean_unperturbed": fc_mean_unperturbed,
        "ts_gt_mean": gt_mean,
        "diff": diff,
        "rmse": rmse,
        "rmse_mean": rmse_mean,
        "rmse_unperturbed": rmse_unperturbed,
        "sr_ensemble_spread": ensemble_spread,
        "sr_spread_skill_ratio": spread_skill_ratio,
        "energy_spectra_forecast": energy_spectra_forecast,
        "energy_spectra_unperturbed": energy_spectra_unperturbed,
        "energy_spectra_ground_truth": energy_spectra_ground_truth,
        "crps": crps,
        "ensemble_spread_grid": ensemble_spread_grid,
    }


def calculate_y_lims(
    vars_3d, vars_2d, forecast, forecast_ifs, default_stats, ifs_stats
):
    y_lims = {
        "y_lims_rmse": {},
        "y_lims_spread_skill_ratio": {},
        "y_lims_timeseries": {},
        "y_lims_energy_spectra": {},
    }

    def get_stat(stat_dict, stat_name, variable, is_3d, sel_kwargs):
        return (
            stat_dict[stat_name][variable].sel(**sel_kwargs)
            if is_3d
            else stat_dict[stat_name][variable]
        )

    stat_names = [
        "ts_fc_mean",
        "ts_fc_mean_unperturbed",
        "ts_gt_mean",
        "rmse",
        "rmse_mean",
        "rmse_unperturbed",
        "sr_ensemble_spread",
        "sr_spread_skill_ratio",
        "energy_spectra_forecast",
        "energy_spectra_unperturbed",
        "energy_spectra_ground_truth",
    ]

    for variable in vars_3d + vars_2d:
        is_3d = variable in vars_3d
        levels = forecast.isobaricInhPa.values if is_3d else [None]

        for level in levels:
            sel_kwargs = {"isobaricInhPa": level} if is_3d else {}

            ts_min, ts_max = float("inf"), float("-inf")
            energy_min, energy_max = float("inf"), float("-inf")
            rmse_min, rmse_max = float("inf"), float("-inf")
            spread_skill_min, spread_skill_max = float("inf"), float("-inf")
            ensemble_spread_min, ensemble_spread_max = float("inf"), float("-inf")

            for stat_name in stat_names:
                default_stat = get_stat(
                    default_stats, stat_name, variable, is_3d, sel_kwargs
                )
                ifs_stat = get_stat(ifs_stats, stat_name, variable, is_3d, sel_kwargs)

                stat_min = min(
                    default_stat.min().values,
                    ifs_stat.min().values,
                )
                stat_max = max(
                    default_stat.max().values,
                    ifs_stat.max().values,
                )

                if stat_name.startswith("ts"):
                    ts_min = min(ts_min, stat_min)
                    ts_max = max(ts_max, stat_max)
                elif stat_name.startswith("energy"):
                    energy_min = min(energy_min, stat_min)
                    energy_max = max(energy_max, stat_max)
                elif stat_name.startswith("rmse"):
                    rmse_min = min(rmse_min, stat_min)
                    rmse_max = max(rmse_max, stat_max)
                elif stat_name == "sr_spread_skill_ratio":
                    spread_skill_min = min(spread_skill_min, stat_min)
                    spread_skill_max = max(spread_skill_max, stat_max)
                elif stat_name == "sr_ensemble_spread":
                    ensemble_spread_min = min(ensemble_spread_min, stat_min)
                    ensemble_spread_max = max(ensemble_spread_max, stat_max)

            y_lims["y_lims_timeseries"][(variable, level)] = (ts_min, ts_max)
            y_lims["y_lims_energy_spectra"][(variable, level)] = (
                energy_min,
                energy_max,
            )
            y_lims["y_lims_rmse"][(variable, level)] = (rmse_min, rmse_max)
            y_lims["y_lims_spread_skill_ratio"][(variable, level)] = (
                (spread_skill_min, spread_skill_max),
                (ensemble_spread_min, ensemble_spread_max),
            )

    print("Y-limits calculated")

    return y_lims
