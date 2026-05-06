"""Conformance tests for ai_models_ensembles.swissclim_format.

We don't run the full SwissClim validator (heavy import); instead we assert
the schema invariants the validator checks: dims, coords, dtype of lead_time,
and absence of forbidden artefacts.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ai_models_ensembles.swissclim_format import (
    SHORT_TO_LONG,
    to_swissclim_forecast,
    to_swissclim_target,
)


def _cfgrib_like_forecast() -> xr.Dataset:
    step = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    lat = np.linspace(90.0, -90.0, 5, dtype="float32")
    lon = np.linspace(0.0, 359.0, 6, dtype="float32")
    levels = np.array([500, 850], dtype="int64")
    init_time = np.datetime64("2018-01-01T00:00", "ns")
    rng = np.random.default_rng(0)

    ds = xr.Dataset(
        {
            "10u": (("step", "latitude", "longitude"), rng.standard_normal((3, 5, 6))),
            "2t": (("step", "latitude", "longitude"), rng.standard_normal((3, 5, 6))),
            "t": (
                ("step", "isobaricInhPa", "latitude", "longitude"),
                rng.standard_normal((3, 2, 5, 6)),
            ),
        },
        coords={
            "step": step,
            "isobaricInhPa": levels,
            "latitude": lat,
            "longitude": lon,
            "time": init_time,
            "valid_time": ("step", step + init_time),
        },
    )
    ds = ds.assign_coords(member=0).expand_dims({"member": 1})
    return ds


def test_forecast_renames_dims_and_vars():
    out = to_swissclim_forecast(_cfgrib_like_forecast())
    assert "step" not in out.dims
    assert "isobaricInhPa" not in out.dims
    assert "member" not in out.dims
    assert {"init_time", "lead_time", "ensemble", "latitude", "longitude", "level"}.issubset(
        set(out.dims)
    )


def test_forecast_lead_time_dtype():
    out = to_swissclim_forecast(_cfgrib_like_forecast())
    assert np.issubdtype(out["lead_time"].dtype, np.timedelta64)


def test_forecast_drops_valid_time():
    out = to_swissclim_forecast(_cfgrib_like_forecast())
    assert "valid_time" not in out.coords
    assert "valid_time" not in out.variables


def test_forecast_long_variable_names():
    out = to_swissclim_forecast(_cfgrib_like_forecast())
    expected = {SHORT_TO_LONG[s] for s in ("10u", "2t", "t")}
    assert expected.issubset(set(out.data_vars))


def test_forecast_init_time_is_dim():
    out = to_swissclim_forecast(_cfgrib_like_forecast())
    assert "init_time" in out.dims
    assert out.sizes["init_time"] == 1


def test_target_renames_isobaric_and_drops_valid_time():
    ds = xr.Dataset(
        {
            "2t": (("time", "latitude", "longitude"), np.zeros((2, 3, 4))),
            "t": (
                ("time", "isobaricInhPa", "latitude", "longitude"),
                np.zeros((2, 2, 3, 4)),
            ),
        },
        coords={
            "time": np.array(["2018-01-01T00", "2018-01-01T06"], dtype="datetime64[ns]"),
            "isobaricInhPa": np.array([500, 850], dtype="int64"),
            "latitude": np.linspace(90, -90, 3),
            "longitude": np.linspace(0, 359, 4),
            "valid_time": ("time", np.zeros(2, dtype="datetime64[ns]")),
        },
    )
    out = to_swissclim_target(ds)
    assert "isobaricInhPa" not in out.dims
    assert "level" in out.dims
    assert "valid_time" not in out.coords
    assert "time" in out.dims  # SwissClim aliases `time` -> `init_time` itself
    assert "2m_temperature" in out.data_vars
    assert "temperature" in out.data_vars
