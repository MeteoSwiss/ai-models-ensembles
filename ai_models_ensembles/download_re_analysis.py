from __future__ import annotations

import ast
import os
from datetime import datetime, timedelta
from typing import List

import earthkit.data
import xarray as xr
from earthkit.data import settings


def download_re_analysis(
    out_dir: str, start_date: str, end_date: str, interval: int, model_name: str
) -> None:
    settings.set("cache-policy", "user")
    # Optional Earthkit cache directory via environment variable
    cache_dir = os.environ.get("EARTHKIT_CACHE_DIR")
    if cache_dir:
        settings.set("user-cache-directory", cache_dir)

    path = os.path.join(out_dir, start_date, model_name)
    with open(path + "/fields.txt", "r") as f:
        lines = f.readlines()

    grid = ast.literal_eval(lines[0].split(": ")[1].strip())
    area = ast.literal_eval(lines[1].split(": ")[1].strip())
    pressure_levels = ast.literal_eval(lines[3].split(": ")[1].strip())
    pressure_level_params = ast.literal_eval(lines[4].split(": ")[1].strip())
    single_level_params = ast.literal_eval(lines[6].split(": ")[1].strip())
    single_level_params_notp = [param for param in single_level_params if param != "tp"]

    start_dt = datetime.strptime(start_date, "%Y%m%d%H%M")
    end_dt = datetime.strptime(end_date, "%Y%m%d%H%M")

    pretty_names = {
        "graphcast": "GraphCast",
        "fourcastnetv2-small": "FourCastNetV2-Small",
        "gencast": "GenCast",
    }
    print(
        f"Downloading data for {pretty_names.get(model_name, model_name)} model...",
        flush=True,
    )

    graphcast_like = model_name in {"graphcast", "gencast"}

    if graphcast_like:
        date = int(start_dt.strftime("%Y%m%d"))
        time = int(start_dt.strftime("%H%M"))
        start_date_prev = start_dt - timedelta(hours=6)
        date_prev = int(start_date_prev.strftime("%Y%m%d"))
        time_prev = int(start_date_prev.strftime("%H%M"))
    else:
        date = int(start_dt.strftime("%Y%m%d"))
        time = int(start_dt.strftime("%H%M"))

    analysis_times: List[datetime] = []
    current_date = start_dt
    while current_date <= end_dt:
        analysis_times.extend(
            [current_date + timedelta(hours=h) for h in range(0, 24, int(interval))]
        )
        current_date += timedelta(days=1)

    dates = sorted(list({t.strftime("%Y-%m-%d") for t in analysis_times}))
    times = sorted(list({t.strftime("%H:%M") for t in set(analysis_times)}))

    ds_single_init = earthkit.data.from_source(
        "cds",
        "reanalysis-era5-single-levels",
        variable=single_level_params,
        product_type="reanalysis",
        area=area,
        grid=grid,
        date=date,
        time=time,
    )

    if graphcast_like:
        ds_single_prev = earthkit.data.from_source(
            "cds",
            "reanalysis-era5-single-levels",
            variable=single_level_params,
            product_type="reanalysis",
            area=area,
            grid=grid,
            date=date_prev,
            time=time_prev,
        )

    ds_pressure_init = earthkit.data.from_source(
        "cds",
        "reanalysis-era5-pressure-levels",
        variable=pressure_level_params,
        product_type="reanalysis",
        area=area,
        grid=grid,
        date=date,
        time=time,
        levels=pressure_levels,
    )

    if graphcast_like:
        ds_pressure_prev = earthkit.data.from_source(
            "cds",
            "reanalysis-era5-pressure-levels",
            variable=pressure_level_params,
            product_type="reanalysis",
            area=area,
            grid=grid,
            date=date_prev,
            time=time_prev,
            levels=pressure_levels,
        )

    if graphcast_like:
        ds_combined = ds_single_init + ds_pressure_init + ds_single_prev + ds_pressure_prev
    else:
        ds_combined = ds_single_init + ds_pressure_init

    print("Saving initial conditions to grib...")
    ds_combined.save(f"{path}/init_field.grib")

    ds_single = earthkit.data.from_source(
        "cds",
        "reanalysis-era5-single-levels",
        variable=single_level_params_notp,
        product_type="reanalysis",
        area=area,
        grid=grid,
        date=dates,
        time=times,
    )

    ds_pressure = earthkit.data.from_source(
        "cds",
        "reanalysis-era5-pressure-levels",
        variable=pressure_level_params,
        product_type="reanalysis",
        area=area,
        grid=grid,
        date=dates,
        time=times,
        levels=pressure_levels,
    )
    if graphcast_like:
        ds_single_xr = ds_single.to_xarray().drop_vars(["z"])
        ds_single_xr["surface_z"] = ds_single.sel({"shortName": "z"}).to_xarray()["z"]
        ds_combined_xr = xr.merge([ds_single_xr, ds_pressure.to_xarray()])
    else:
        ds_combined_xr = xr.merge([ds_single.to_xarray(), ds_pressure.to_xarray()])

    ds_combined_xr = ds_combined_xr.sel(time=slice(start_dt, end_dt + timedelta(hours=1)))
    chunks = {"latitude": -1, "longitude": -1, "time": 1, "isobaricInhPa": -1}
    print("Saving ground truth to zarr...")
    ds_combined_xr.chunk(chunks=chunks).drop_vars(["valid_time"]).to_zarr(
        f"{path}/ground_truth.zarr", mode="w", consolidated=True
    )


__all__ = ["download_re_analysis"]
