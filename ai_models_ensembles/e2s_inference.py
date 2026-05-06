"""End-to-end inference driver: earth2studio model -> SwissClim-format zarr.

Public entry: `run_inference(...)`. Used by `ai-ens infer`.

Supported combinations:

  members  ic_perturb  weight_perturb   behaviour
  -------  ----------  --------------   ---------
   1        no          no              deterministic single forecast
   N        yes         no              N IC-perturbed members (one model load)
   N        no          yes             N weight-perturbed members (model loaded
                                        N times from per-member mirror packages)
   N        yes         yes             N members with both perturbations
                                        (per-member weights + per-member IC)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .e2s_data import XarrayDataSource, build_data_source, fetch_initial_conditions
from .e2s_models import get_spec, load_model, steps_from_hours
from .e2s_perturbation import (
    materialise_perturbed_package,
    perturb_initial_conditions,
)
from .swissclim_format import e2s_to_swissclim


def _seed_rngs(seed: int) -> None:
    """Seed every RNG that earth2studio's stochastic models read from.

    Probabilistic models (FCN3, Atlas, ...) draw their per-call
    sample from torch's global RNG. We seed numpy too so any IC-perturbation
    or ancillary noise is reproducible per-member.
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_one_member(
    model: Any,
    data: Any,
    init_time: datetime,
    steps: int,
    ensemble_id: int,
    seed: int,
) -> xr.Dataset:
    """Run e2s deterministic rollout for a single member and bridge to SwissClim."""
    from earth2studio.io import XarrayBackend
    from earth2studio.run import deterministic

    _seed_rngs(seed)
    init_array = np.array([np.datetime64(init_time, "ns")])
    backend = XarrayBackend()
    deterministic(init_array, steps, model, data, backend)
    raw = getattr(backend, "root", None) or getattr(backend, "dataset", None)
    if raw is None:
        raise RuntimeError("XarrayBackend exposed neither .root nor .dataset")
    return e2s_to_swissclim(raw, ensemble_id=ensemble_id)


def _filter_levels(ds: xr.Dataset, levels: list[int] | None) -> xr.Dataset:
    """Keep only the requested pressure levels in 3D variables.

    Surface variables (no `level` dim) pass through untouched. If `levels` is
    None/empty, the dataset is returned unchanged. Levels not present in the
    dataset are silently dropped from the request.
    """
    if not levels or "level" not in ds.dims:
        return ds
    available = set(int(v) for v in ds["level"].values.tolist())
    keep = [lvl for lvl in levels if lvl in available]
    if not keep:
        return ds
    return ds.sel(level=keep)


def _data_for_member(
    base_source: str,
    ic_magnitude: float,
    init_time: datetime,
    member_id: int,
    seed: int,
    cached_ic: xr.Dataset | None,
) -> tuple[Any, xr.Dataset | None]:
    """Build the data source for a member, optionally perturbing the IC."""
    if ic_magnitude <= 0:
        return build_data_source(base_source), cached_ic
    if cached_ic is None:
        cached_ic = fetch_initial_conditions(base_source, init_time)
    perturbed = perturb_initial_conditions(cached_ic, ic_magnitude, seed=seed + member_id)
    return XarrayDataSource(perturbed), cached_ic


def _model_for_member(
    model_name: str,
    weight_magnitude: float,
    layer: int | None,
    member_id: int,
    seed: int,
    work_dir: Path,
) -> Any:
    if weight_magnitude <= 0:
        model, _ = load_model(model_name)
        return model
    pkg_dir = work_dir / f"weights_member_{member_id:03d}"
    package_root = materialise_perturbed_package(
        model_name=model_name,
        magnitude=weight_magnitude,
        layer=layer,
        seed=seed + member_id,
        out_dir=pkg_dir,
    )
    model, _ = load_model(model_name, package_root=package_root)
    return model


def run_inference(
    model_name: str,
    init_time: datetime,
    lead_hours: int,
    output: Path,
    *,
    n_members: int = 1,
    ic_magnitude: float = 0.0,
    weight_magnitude: float = 0.0,
    layer: int | None = None,
    data_source: str = "arco",
    seed: int = 0,
    work_dir: Path | None = None,
    output_levels: list[int] | None = None,
) -> Path:
    """Run an earth2studio model and emit a SwissClim-format zarr at `output`.

    The output is built incrementally: each member's xr.Dataset is computed
    in turn and either written (member 0) or appended along the `ensemble`
    dim. This keeps memory bounded by one rollout at a time.
    """
    spec = get_spec(model_name)
    steps = steps_from_hours(spec, lead_hours)
    output = Path(output)
    work_dir = Path(work_dir or output.parent / "_e2s_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Reuse one model when weights are not perturbed.
    shared_model = None if weight_magnitude > 0 else load_model(model_name)[0]
    cached_ic: xr.Dataset | None = None

    for m in range(n_members):
        print(f"[member {m + 1}/{n_members}] preparing data + model")
        data, cached_ic = _data_for_member(
            base_source=data_source,
            ic_magnitude=ic_magnitude,
            init_time=init_time,
            member_id=m,
            seed=seed,
            cached_ic=cached_ic,
        )
        model = (
            shared_model
            if shared_model is not None
            else _model_for_member(
                model_name=model_name,
                weight_magnitude=weight_magnitude,
                layer=layer,
                member_id=m,
                seed=seed,
                work_dir=work_dir,
            )
        )

        print(f"[member {m + 1}/{n_members}] running rollout")
        ds = _run_one_member(
            model, data, init_time, steps, ensemble_id=m, seed=seed + m
        )
        ds = _filter_levels(ds, output_levels)

        if m == 0:
            ds.to_zarr(output, mode="w", consolidated=True)
        else:
            ds.to_zarr(output, mode="a", append_dim="ensemble", consolidated=True)
        print(f"[member {m + 1}/{n_members}] wrote ensemble={m} -> {output}")

    return output


__all__ = ["run_inference"]
