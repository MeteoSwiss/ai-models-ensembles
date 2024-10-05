import numpy as np
import xarray as xr


def load_and_prepare_data(
    path_in,
    selected_vars,
    crop_region,
    model_name,
    perturb_init,
    perturb_latent,
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
            forecast_unperturbed_ifs (xr.Dataset): The unperturbed IFS forecast data.
    """

    ground_truth = xr.open_zarr(f"{path_in}/ground_truth.zarr", consolidated=True).isel(
        number=0, surface=0
    )
    forecast = xr.open_zarr(
        f"{path_in}/init_{perturb_init}_latent_{perturb_latent}/forecast.zarr",
        consolidated=True,
    )
    forecast_unperturbed = xr.open_zarr(f"{path_in}/forecast.zarr", consolidated=True)
    forecast_ifs = xr.open_zarr(f"{path_in}/ifs_ens.zarr", consolidated=True)

    if crop_region == "europe":
        lat_min, lat_max = 25, 80
        lon_min, lon_max = 340, 50

        # Use modulo arithmetic to ensure longitudes are in 0-360 range
        lon_min = lon_min % 360
        lon_max = lon_max % 360

        # Create a list of longitudes that wraps around 0/360
        lats = list(range(lat_min, lat_max + 1))
        lons = list(range(lon_min, 360)) + list(range(0, lon_max + 1))

        # Crop all datasets to the European lat-lon box
        ground_truth = ground_truth.sel(latitude=lats, longitude=lons)
        forecast = forecast.sel(latitude=lats, longitude=lons)
        forecast_unperturbed = forecast_unperturbed.sel(latitude=lats, longitude=lons)
        forecast_ifs = forecast_ifs.sel(latitude=lats, longitude=lons)
    else:
        lat_min, lat_max = -90, 90
        lon_min, lon_max = 0, 360

    forecast_ifs = (
        forecast_ifs.rename({"number": "member"})
        .set_index(member="member")
        .isel(surface=0)
    )

    ground_truth = ground_truth.isel(step=0)
    if model_name == "graphcast":
        forecast = forecast.drop_isel(step=0)
        forecast["step"] = forecast["step"] - np.timedelta64(6, "h")
        forecast["step"] = forecast["step"].values / 1e9 / 3600
        forecast_unperturbed = forecast_unperturbed.drop_isel(step=0)
        forecast_unperturbed["step"] = forecast_unperturbed["step"] - np.timedelta64(
            6, "h"
        )
    forecast_ifs["member"] = forecast_ifs["member"] - 1
    forecast_unperturbed = forecast_unperturbed.sel(
        member=0,
    )
    forecast_unperturbed_ifs = forecast_ifs.isel(time=0).sel(
        member=0,
    )
    forecast_ifs = forecast_ifs.isel(member=slice(1, None))
    forecast_unperturbed = xr.concat([ground_truth, forecast_unperturbed], "step")
    forecast = xr.concat(
        [ground_truth, forecast],
        "step",
    )

    indices = np.random.permutation(forecast.member.size)
    indices_ifs = np.random.permutation(forecast_ifs.member.size)
    selected_indices = indices[:num_members]
    selected_indices_ifs = indices_ifs[:num_members]
    forecast = forecast.isel(member=selected_indices, time=0)
    forecast_ifs = forecast_ifs.isel(member=selected_indices_ifs, time=0)
    forecast_unperturbed = forecast_unperturbed.isel(time=0)

    ground_truth["step"] = ("time", np.arange(len(ground_truth["time"])))
    ground_truth = ground_truth.swap_dims({"time": "step"})
    forecast_ifs["step"] = forecast["step"]
    forecast_unperturbed["step"] = forecast["step"]
    forecast_unperturbed_ifs["step"] = forecast["step"]
    ground_truth["step"] = forecast["step"]

    forecast = forecast.transpose(..., "member")
    forecast_ifs = forecast_ifs.transpose(..., "member")

    full_levels = forecast_ifs.isobaricInhPa.where(
        ~forecast_ifs.t.isnull()
        .any(dim=["latitude", "longitude", "step", "member"])
        .compute(),
        drop=True,
    ).values
    forecast = forecast.sel(isobaricInhPa=full_levels)
    forecast_ifs = forecast_ifs.sel(isobaricInhPa=full_levels)
    forecast_unperturbed = forecast_unperturbed.sel(isobaricInhPa=full_levels)
    forecast_unperturbed_ifs = forecast_unperturbed_ifs.sel(isobaricInhPa=full_levels)
    ground_truth = ground_truth.sel(isobaricInhPa=full_levels)

    ground_truth = ground_truth[selected_vars]
    forecast = forecast[selected_vars]
    forecast_unperturbed = forecast_unperturbed[selected_vars]
    forecast_ifs = forecast_ifs[selected_vars]
    forecast_unperturbed_ifs = forecast_unperturbed_ifs[selected_vars]

    if debug_mode:
        max_members = 10
        max_time_steps = 5

        forecast = forecast.isel(member=slice(0, max_members))
        forecast_ifs = forecast_ifs.isel(member=slice(0, max_members))

        ground_truth = ground_truth.isel(step=slice(0, max_time_steps))
        forecast = forecast.isel(step=slice(0, max_time_steps))
        forecast_unperturbed = forecast_unperturbed.isel(step=slice(0, max_time_steps))
        forecast_ifs = forecast_ifs.isel(step=slice(0, max_time_steps))
        forecast_unperturbed_ifs = forecast_unperturbed_ifs.isel(
            step=slice(0, max_time_steps)
        )

    for ds_name in [
        "ground_truth",
        "forecast",
        "forecast_unperturbed",
        "forecast_ifs",
        "forecast_unperturbed_ifs",
    ]:
        ds = locals()[ds_name]
        print(f"{ds_name}: {ds.sizes}")

    return {
        "ground_truth": ground_truth,
        "forecast": forecast,
        "forecast_unperturbed": forecast_unperturbed,
        "forecast_ifs": forecast_ifs,
        "forecast_unperturbed_ifs": forecast_unperturbed_ifs,
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
    squared_diff = (forecast - ground_truth) ** 2
    squared_diff_mean = (forecast.mean(dim="member") - ground_truth) ** 2
    rmse_grid = np.sqrt(squared_diff.mean(dim="member")).drop_isel(step=0)
    rmse = np.sqrt(squared_diff.mean(dim=["latitude", "longitude"])).drop_isel(step=0)
    rmse_mean = np.sqrt(
        squared_diff_mean.mean(dim=["latitude", "longitude"])
    ).drop_isel(step=0)

    squared_diff_unperturbed = (forecast_unperturbed - ground_truth) ** 2
    rmse_unperturbed = np.sqrt(
        squared_diff_unperturbed.mean(dim=["latitude", "longitude"])
    ).drop_isel(step=0)

    if crop_region == "europe":
        lat_min, lat_max = 25, 80

    ensemble_spread_grid = forecast.std(dim="member").drop_isel(step=0)
    spread_skill_ratio_grid = ensemble_spread_grid / rmse_grid
    spread_skill_ratio = spread_skill_ratio_grid.mean(dim=["latitude", "longitude"])
    ensemble_spread = ensemble_spread_grid.mean(dim=["latitude", "longitude"])

    def calculate_energy_spectra(data, lat_band=(30, 60)):
        lat_slice = slice(lat_band[0], lat_band[1])
        data_lat_band = data.sel(latitude=lat_slice)

        # Select the last time step
        data_last_step = data_lat_band.isel(step=-1)

        energy_spectra_members = []

        # Calculate energy spectra for each member
        for member in data_last_step.member:
            data_member = data_last_step.sel(member=member)
            # Calculate mean over latitude band
            data_mean = data_member.mean(dim="latitude")

            # Calculate FFT along longitude
            fft = np.fft.fft(data_mean.values, axis=-1)

            # Calculate power spectrum
            power_spectrum = np.abs(fft) ** 2

            # Calculate wavelengths
            n = data_mean.longitude.size
            wavelengths = (360 / np.fft.fftfreq(n)[1 : n // 2]) * 111  # Convert to km

            energy_spectra_members.append(
                xr.Dataset(
                    {
                        "power": ("wavelength", power_spectrum[1 : n // 2]),
                        "wavelength": ("wavelength", wavelengths),
                    }
                )
            )

        # Concatenate energy spectra of all members along a new dimension
        energy_spectra_concat = xr.concat(energy_spectra_members, dim="member")

        return energy_spectra_concat

    energy_spectra = {}
    for var in forecast.data_vars:
        energy_spectra[var] = {
            "forecast": calculate_energy_spectra(
                forecast[var], lat_band=(lat_min, lat_max)
            ),
            "unperturbed": calculate_energy_spectra(
                forecast_unperturbed[var].expand_dims("member"),
                lat_band=(lat_min, lat_max),
            ).squeeze(),  # Remove singleton 'member' dimension
            "ground_truth": calculate_energy_spectra(
                ground_truth[var].expand_dims("member"), lat_band=(lat_min, lat_max)
            ).squeeze(),
        }

    return {
        "fc_mean": fc_mean,
        "fc_mean_unperturbed": fc_mean_unperturbed,
        "gt_mean": gt_mean,
        "rmse": rmse,
        "rmse_mean": rmse_mean,
        "rmse_unperturbed": rmse_unperturbed,
        "spread_skill_ratio": spread_skill_ratio,
        "ensemble_spread": ensemble_spread,
        "energy_spectra": energy_spectra,
    }


def calculate_y_lims(
    vars_3d, vars_2d, forecast, forecast_ifs, default_stats, ifs_stats
):
    """
    Calculate the y-limits for the plots.

    Args:
        vars_3d (list): The 3D variables.
        vars_2d (list): The 2D variables.
        forecast (xr.Dataset): The forecast data.
        forecast_ifs (xr.Dataset): The IFS forecast data.
        default_stats (dict): The default statistics.
        ifs_stats (dict): The IFS statistics.

    Returns:
        dict: A dictionary containing the y-limits.
    """
    y_lims_rmse = {}
    y_lims_spread_skill_ratio = {}
    y_lims_timeseries = {}
    y_lims_energy_spectra = {}

    for variable in vars_3d + vars_2d:
        is_3d = variable in vars_3d
        levels = forecast.isobaricInhPa.values if is_3d else [None]

        for level in levels:
            sel_kwargs = {"isobaricInhPa": level} if is_3d else {}

            def get_stat(stat_dict, stat_name):
                return (
                    stat_dict[stat_name][variable].sel(**sel_kwargs)
                    if is_3d
                    else stat_dict[stat_name][variable]
                )

            rmse_min = min(
                get_stat(default_stats, "rmse").min().values,
                get_stat(ifs_stats, "rmse").min().values,
            )
            rmse_max = max(
                get_stat(default_stats, "rmse").max().values,
                get_stat(ifs_stats, "rmse").max().values,
            )
            spread_skill_ratio_min = min(
                get_stat(default_stats, "spread_skill_ratio").min().values,
                get_stat(ifs_stats, "spread_skill_ratio").min().values,
            )
            spread_skill_ratio_max = max(
                get_stat(default_stats, "spread_skill_ratio").max().values,
                get_stat(ifs_stats, "spread_skill_ratio").max().values,
            )
            ensemble_spread_min = min(
                get_stat(default_stats, "ensemble_spread").min().values,
                get_stat(ifs_stats, "ensemble_spread").min().values,
            )
            ensemble_spread_max = max(
                get_stat(default_stats, "ensemble_spread").max().values,
                get_stat(ifs_stats, "ensemble_spread").max().values,
            )
            timeseries_min = min(
                get_stat(default_stats, "fc_mean").min().values,
                get_stat(ifs_stats, "fc_mean").min().values,
            )
            timeseries_max = max(
                get_stat(default_stats, "fc_mean").max().values,
                get_stat(ifs_stats, "fc_mean").max().values,
            )

            y_lims_rmse[(variable, level)] = (rmse_min, rmse_max)
            y_lims_spread_skill_ratio[(variable, level)] = (
                (spread_skill_ratio_min, spread_skill_ratio_max),
                (ensemble_spread_min, ensemble_spread_max),
            )
            y_lims_timeseries[(variable, level)] = (timeseries_min, timeseries_max)

        energy_spectra_min = min(
            default_stats["energy_spectra"][variable]["forecast"].power.min().values,
            ifs_stats["energy_spectra"][variable]["forecast"].power.min().values,
        )
        energy_spectra_max = max(
            default_stats["energy_spectra"][variable]["forecast"].power.max().values,
            ifs_stats["energy_spectra"][variable]["forecast"].power.max().values,
        )
        energy_level = None if variable in vars_2d else forecast.isobaricInhPa.values[0]
        y_lims_energy_spectra[(variable, energy_level)] = (
            energy_spectra_min,
            energy_spectra_max,
        )

    print("Y-limits calculated")

    return {
        "y_lims_rmse": y_lims_rmse,
        "y_lims_spread_skill_ratio": y_lims_spread_skill_ratio,
        "y_lims_timeseries": y_lims_timeseries,
        "y_lims_energy_spectra": y_lims_energy_spectra,
    }
