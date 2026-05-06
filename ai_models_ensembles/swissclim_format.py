"""Conversion of earth2studio xarray Datasets to the SwissClim Evaluations schema.

Required schema (forecast):
    dims  = (init_time, lead_time, ensemble, latitude, longitude[, level])
    coords:
        init_time  - datetime64[ns]
        lead_time  - timedelta64[ns]
        ensemble   - int
        latitude   - float
        longitude  - float
        level      - int (hPa), only for 3D variables
    data_vars use ECMWF long names (e.g. "10m_u_component_of_wind").

Required schema (ERA5 ground truth): same as above but `time` is acceptable as
a stand-in for `init_time` (SwissClim normalizes it).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Short / cfgrib name -> ECMWF (WeatherBench2) long name. Used for both
# legacy cfgrib outputs and the surface tokens that earth2studio shares.
SHORT_TO_LONG: dict[str, str] = {
    # Single level
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "100u": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "2t": "2m_temperature",
    "2d": "2m_dewpoint_temperature",
    "msl": "mean_sea_level_pressure",
    "sp": "surface_pressure",
    "tp": "total_precipitation",
    "tcwv": "total_column_water_vapour",
    "lsm": "land_sea_mask",
    "z": "geopotential",
    "surface_z": "geopotential_at_surface",
    # Pressure level
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
    "r": "relative_humidity",
}

# earth2studio variable lexicon -> ECMWF long name.
# Per-level variables are emitted as "<head><level>" (e.g. "u500", "t850") and
# parsed by `_e2s_var_to_long`; only the head needs to be in the dict.
E2S_TO_SWISSCLIM: dict[str, str] = {
    # Surface
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "u100m": "100m_u_component_of_wind",
    "v100m": "100m_v_component_of_wind",
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "msl": "mean_sea_level_pressure",
    "sp": "surface_pressure",
    "tp": "total_precipitation",
    "tcwv": "total_column_water_vapour",
    "lsm": "land_sea_mask",
    "zsl": "geopotential_at_surface",
    # Pressure-level heads
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
    "r": "relative_humidity",
    "z": "geopotential",
}

# Dim and coord renames that always apply.
DIM_RENAMES: dict[str, str] = {
    "step": "lead_time",
    "isobaricInhPa": "level",
}


def _rename_present(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    present = {k: v for k, v in mapping.items() if k in ds.variables or k in ds.dims}
    return ds.rename(present) if present else ds


def to_swissclim_forecast(ds: xr.Dataset, member_dim: str = "member") -> xr.Dataset:
    """Convert a cfgrib-style forecast dataset to the SwissClim schema.

    Assumes a single initialization time present as a scalar `time` coord (or
    already as a singleton `init_time` dim). The ensemble axis can be named
    `member` (default) or `ensemble`; SwissClim aliases `member` automatically
    but we normalize here so the on-disk store reads identically without alias.
    """
    if "valid_time" in ds.coords or "valid_time" in ds.variables:
        ds = ds.drop_vars("valid_time", errors="ignore")

    ds = _rename_present(ds, DIM_RENAMES)

    if member_dim != "ensemble" and member_dim in ds.dims:
        ds = ds.rename({member_dim: "ensemble"})

    if "lead_time" in ds.coords:
        lt = ds["lead_time"].values
        if not np.issubdtype(lt.dtype, np.timedelta64):
            ds = ds.assign_coords(
                lead_time=np.array(lt, dtype="timedelta64[h]").astype("timedelta64[ns]")
            )
        else:
            ds = ds.assign_coords(lead_time=lt.astype("timedelta64[ns]"))

    if "init_time" not in ds.dims:
        if "time" in ds.dims:
            ds = ds.rename({"time": "init_time"})
        elif "time" in ds.coords:
            init_val = np.atleast_1d(ds["time"].values).astype("datetime64[ns]")
            ds = ds.drop_vars("time")
            ds = ds.expand_dims({"init_time": init_val})
        elif "init_time" in ds.coords:
            ds = ds.expand_dims("init_time")

    ds = _rename_present(ds, SHORT_TO_LONG)
    return ds


def to_swissclim_target(ds: xr.Dataset) -> xr.Dataset:
    """Convert a cfgrib/ERA5 reanalysis dataset to the SwissClim target schema.

    Keeps `time` as the temporal axis (SwissClim accepts `time` for targets and
    aliases it to `init_time` internally). Renames `isobaricInhPa` -> `level`
    and short variable names to long ECMWF names.
    """
    if "valid_time" in ds.coords or "valid_time" in ds.variables:
        ds = ds.drop_vars("valid_time", errors="ignore")
    ds = _rename_present(ds, DIM_RENAMES)
    ds = _rename_present(ds, SHORT_TO_LONG)
    return ds


def _e2s_var_to_long(name: str) -> tuple[str, int | None]:
    """Map an earth2studio variable token to (long_name, level_hPa or None).

    earth2studio emits per-level variables as "<head><level>", e.g. "u500",
    "t850", "z200". Surface variables are emitted without a level suffix.
    """
    if name in E2S_TO_SWISSCLIM:
        return E2S_TO_SWISSCLIM[name], None
    for i, c in enumerate(name):
        if c.isdigit():
            head, tail = name[:i], name[i:]
            try:
                level = int(tail)
            except ValueError:
                continue
            long_name = E2S_TO_SWISSCLIM.get(head) or SHORT_TO_LONG.get(head)
            if long_name:
                return long_name, level
            return name, level
    return name, None


def _unstack_e2s_variable_axis(ds: xr.Dataset) -> xr.Dataset:
    """Convert e2s's stacked 'variable' axis into one data_var per ECMWF name.

    earth2studio writes outputs with a single data_var of shape
    (..., variable, lat, lon) where 'variable' is a string axis enumerating
    each output channel. SwissClim expects one data_var per ECMWF long name
    with a separate 'level' dimension for 3D fields.
    """
    if "variable" not in ds.dims:
        return ds
    payload_name = next(iter(ds.data_vars))
    arr = ds[payload_name]

    by_long: dict[str, list[tuple[int | None, str]]] = {}
    for token in [str(v) for v in ds["variable"].values]:
        long_name, level = _e2s_var_to_long(token)
        by_long.setdefault(long_name, []).append((level, token))

    out_vars: dict[str, xr.DataArray] = {}
    for long_name, entries in by_long.items():
        levels = [lvl for lvl, _ in entries]
        tokens = [tok for _, tok in entries]
        if all(lvl is None for lvl in levels):
            sub = arr.sel(variable=tokens[0]).drop_vars("variable")
            out_vars[long_name] = sub
        else:
            level_values = np.array(
                [lvl for lvl in levels if lvl is not None], dtype="int64"
            )
            sub = arr.sel(variable=tokens).rename({"variable": "level"})
            sub = sub.assign_coords(level=level_values).sortby("level")
            out_vars[long_name] = sub

    return xr.Dataset(
        out_vars,
        coords={k: v for k, v in ds.coords.items() if k != "variable"},
        attrs=ds.attrs,
    )


def e2s_to_swissclim(ds: xr.Dataset, ensemble_id: int = 0) -> xr.Dataset:
    """Bridge an earth2studio output dataset to SwissClim-native zarr layout.

    Steps:
      1. Unstack the 'variable' axis into per-ECMWF-name data_vars.
      2. Rename 'lat'/'lon' to 'latitude'/'longitude'.
      3. Ensure 'ensemble' dim is present (singleton if not).
      4. Apply `to_swissclim_forecast` for the rest (init_time/lead_time
         dtype/coord normalisation, valid_time stripping).
    """
    ds = _unstack_e2s_variable_axis(ds)
    rename_dims = {k: v for k, v in {"lat": "latitude", "lon": "longitude"}.items() if k in ds.dims}
    if rename_dims:
        ds = ds.rename(rename_dims)
    if "ensemble" not in ds.dims:
        ds = ds.expand_dims({"ensemble": [ensemble_id]})
    return to_swissclim_forecast(ds, member_dim="ensemble")


__all__ = [
    "SHORT_TO_LONG",
    "E2S_TO_SWISSCLIM",
    "DIM_RENAMES",
    "to_swissclim_forecast",
    "to_swissclim_target",
    "e2s_to_swissclim",
]
