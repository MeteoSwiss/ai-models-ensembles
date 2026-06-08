"""earth2studio DataSource adapters.

Two complementary helpers:

* `XarrayDataSource` - implements the e2s `DataSource` protocol over an
  in-memory `xr.Dataset`. Used to feed perturbed initial conditions back into
  a model rollout without round-tripping through disk.

* `build_data_source` - factory mapping a string spec ("arco", "cds", "gfs",
  "ifs", "file:/path/to.zarr") to the appropriate e2s data source.

The e2s `DataSource` protocol is duck-typed: any callable with signature
`(time: datetime | array, variable: str | list[str]) -> xr.DataArray` works.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .swissclim_format import _e2s_var_to_long


class XarrayDataSource:
    """In-memory `xr.Dataset` exposed as an earth2studio `DataSource`.

    The wrapped dataset must already use earth2studio's variable lexicon
    (lowercase tokens like "u10m", "t2m", "u500"). Use `from_swissclim` to
    build one from a SwissClim-format dataset (long ECMWF names + 'level' dim).
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds

    @classmethod
    def from_swissclim(cls, ds: xr.Dataset) -> "XarrayDataSource":
        """Build an XarrayDataSource from a SwissClim-format dataset.

        Inverts `e2s_to_swissclim`: stack `level` into per-channel variables
        like 'u500', 't850' and rename 'latitude'/'longitude' to 'lat'/'lon'.
        """
        from .swissclim_format import E2S_TO_SWISSCLIM

        long_to_e2s_head = {v: k for k, v in E2S_TO_SWISSCLIM.items()}
        flat: dict[str, xr.DataArray] = {}
        for name, da in ds.data_vars.items():
            head = long_to_e2s_head.get(name, name)
            if "level" in da.dims:
                for lvl in da["level"].values:
                    token = f"{head}{int(lvl)}"
                    flat[token] = da.sel(level=int(lvl)).drop_vars("level")
            else:
                flat[head] = da
        out = xr.Dataset(flat)
        if "latitude" in out.dims:
            out = out.rename({"latitude": "lat"})
        if "longitude" in out.dims:
            out = out.rename({"longitude": "lon"})
        if "init_time" in out.dims:
            out = out.rename({"init_time": "time"})
        return cls(out)

    def __call__(
        self,
        time: datetime | np.datetime64 | np.ndarray,
        variable: str | list[str],
    ) -> xr.DataArray:
        times = np.atleast_1d(np.array(time, dtype="datetime64[ns]"))
        var_list = [variable] if isinstance(variable, str) else list(variable)

        chans = []
        for v in var_list:
            if v in self.ds.data_vars:
                chans.append(self.ds[v].sel(time=times))
            else:
                head, level = _e2s_var_to_long(v)
                if head in self.ds.data_vars and "level" in self.ds[head].dims:
                    chans.append(self.ds[head].sel(time=times, level=level))
                else:
                    raise KeyError(f"Variable '{v}' not present in XarrayDataSource")

        stacked = xr.concat(chans, dim="variable")
        stacked = stacked.assign_coords(variable=("variable", var_list))
        return stacked


def fetch_initial_conditions(
    source: str,
    init_time: datetime,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Materialise the initial-condition fields needed at `init_time` into memory.

    Returned dataset uses earth2studio variable tokens (e.g. 'u10m', 't500')
    and has a 'time' dim of length 1.
    """
    ds_handle = build_data_source(source, materialise=False)
    if variables is None:
        # Best-effort: pull a small default surface set; callers should pass
        # the model's input variables explicitly when they know them.
        variables = ["u10m", "v10m", "t2m", "msl"]
    arr = ds_handle(np.datetime64(init_time, "ns"), variables)
    return arr.to_dataset("variable") if "variable" in arr.dims else arr.to_dataset()


def _ifs_analysis_source(zarr_path: str) -> XarrayDataSource:
    """Build an XarrayDataSource serving the lead_time=0 slice of an IFS zarr.

    The lead_time=0 step of an IFS forecast is the IFS analysis at that
    init_time. If the zarr has an `ensemble` dim (IFS ENS), we keep only
    member 0 (the control's analysis is identical to ensemble member 0's at
    t=0; with IFS ENS members 1..50 also share the analysis at t=0).

    Expects a SwissClim-format zarr (init_time, lead_time, ensemble?,
    latitude, longitude, level?, ECMWF long names). Variable names are
    converted to the earth2studio lexicon by `XarrayDataSource.from_swissclim`.
    """
    ds = xr.open_zarr(zarr_path, consolidated=True)
    if "lead_time" in ds.dims:
        ds = ds.isel(lead_time=0).drop_vars("lead_time", errors="ignore")
    if "ensemble" in ds.dims:
        ds = ds.isel(ensemble=0).drop_vars("ensemble", errors="ignore")
    return XarrayDataSource.from_swissclim(ds)


def ifs_ens_member_ic_source(
    zarr_path: str,
    member_id: int,
    cached_ds: xr.Dataset | None = None,
) -> tuple[XarrayDataSource, xr.Dataset]:
    """Serve IFS-ENS perturbed-analysis member `member_id` as a per-member IC.

    Phase 5: instead of every ensemble member starting from one shared ERA5
    analysis, member k is initialised from IFS-ENS perturbed-analysis member k,
    combining real EDA-derived IC spread with the weight perturbation applied
    elsewhere. `cached_ds` lets the caller open the (lazy) store once and reuse
    it across members. The store is SwissClim-format with an `ensemble` dim and
    either a pure-analysis layout (no `lead_time`) or a forecast layout whose
    `lead_time=0` slice is the analysis. Variable-name -> earth2studio lexicon
    conversion and the init_time -> time rename happen in `from_swissclim`, so
    the model can request both T and T-6h as long as the store carries those
    init_times.
    """
    if cached_ds is None:
        cached_ds = xr.open_zarr(zarr_path)
        if "lead_time" in cached_ds.dims:
            cached_ds = cached_ds.isel(lead_time=0).drop_vars("lead_time", errors="ignore")
    member = cached_ds.isel(ensemble=member_id).drop_vars("ensemble", errors="ignore")
    return XarrayDataSource.from_swissclim(member), cached_ds


def build_data_source(spec: str, materialise: bool = False) -> Any:
    """Factory for earth2studio DataSource objects.

    Accepted specs:
      * "arco"               - Google's ARCO ERA5
      * "cds"                - Copernicus CDS (requires ~/.cdsapirc)
      * "gfs"                - NOAA GFS analysis
      * "ifs"                - ECMWF IFS open data
      * "ifs_ens"            - ECMWF IFS ensemble open data
      * "wb2"                - WeatherBench2 ERA5 zarr
      * "file:PATH"          - local zarr/netCDF (raw xr.open_dataset/open_zarr)
      * "ifs_analysis:PATH"  - SwissClim-format IFS zarr; serves lead_time=0
                               slice as the analysis IC (recommended for fair
                               cross-model comparison).
    """
    if spec.startswith("ifs_analysis:"):
        return _ifs_analysis_source(spec[len("ifs_analysis:") :])
    if spec.startswith("file:"):
        path = spec[len("file:") :]
        return XarrayDataSource(
            xr.open_dataset(path) if Path(path).is_file() else xr.open_zarr(path)
        )

    from earth2studio import data as e2s_data  # type: ignore

    table = {
        "arco": "ARCO",
        "cds": "CDS",
        "gfs": "GFS",
        "ifs": "IFS",
        "ifs_ens": "IFS_ENS",
        "wb2": "WB2ERA5",
    }
    cls_name = table.get(spec.lower())
    if cls_name is None:
        raise ValueError(f"Unknown data source '{spec}'. Known: {sorted(table)} or 'file:PATH'.")
    cls = getattr(e2s_data, cls_name)
    return cls()


__all__ = [
    "XarrayDataSource",
    "build_data_source",
    "fetch_initial_conditions",
    "ifs_ens_member_ic_source",
]
