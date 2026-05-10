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

When multiple GPUs are available, members are distributed across GPUs
(one process per GPU, round-robin assignment). Each GPU worker runs its
assigned members sequentially.
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
    _parse_layer_spec,
    cache_default_checkpoints,
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
    import os

    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


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
    if hasattr(model, "set_rng"):
        model.set_rng(seed=seed, reset=True)
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


def _filter_vars(ds: xr.Dataset, variables: list[str] | None) -> xr.Dataset:
    """Keep only the requested data variables, dropping the rest."""
    if not variables:
        return ds
    keep = [v for v in variables if v in ds.data_vars]
    return ds[keep]


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
    # Offset IC seed by 10_000 to avoid sharing a seed with weight perturbation
    perturbed = perturb_initial_conditions(cached_ic, ic_magnitude, seed=seed + member_id + 10_000)
    return XarrayDataSource(perturbed), cached_ic


def _model_for_member(
    model_name: str,
    weight_magnitude: float,
    layer: str | None,
    member_id: int,
    seed: int,
    work_dir: Path,
    cached_checkpoints: dict[str, str] | None = None,
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
        cached_checkpoints=cached_checkpoints,
    )
    model, _ = load_model(model_name, package_root=package_root)
    return model


# -- Multi-GPU parallelism ---------------------------------------------------


def _available_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except (ImportError, RuntimeError):
        return 1


def _gpu_worker(
    gpu_id: int,
    member_id: int,
    model_name: str,
    init_time: datetime,
    steps: int,
    n_members_total: int,
    ic_magnitude: float,
    weight_magnitude: float,
    layer: str | None,
    data_source: str,
    seed: int,
    work_dir: str,
    output_dir: str,
    output_levels: list[int] | None,
    output_vars: list[str] | None,
    cached_checkpoints: dict[str, str] | None,
) -> None:
    """Run a single member on a specific GPU. Called via multiprocessing spawn.

    Each invocation runs exactly one member and then exits, ensuring all
    runtime state (including JAX/XLA compiled programs) is fully released.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["BLOSC_NTHREADS"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import torch

    torch.cuda.set_device(0)

    work_path = Path(work_dir)
    output_path = Path(output_dir)
    m = member_id

    print(
        f"[GPU {gpu_id}][member {m + 1}/{n_members_total}] preparing data + model",
        flush=True,
    )
    data, _ = _data_for_member(
        base_source=data_source,
        ic_magnitude=ic_magnitude,
        init_time=init_time,
        member_id=m,
        seed=seed,
        cached_ic=None,
    )
    if weight_magnitude > 0:
        model = _model_for_member(
            model_name=model_name,
            weight_magnitude=weight_magnitude,
            layer=layer,
            member_id=m,
            seed=seed,
            work_dir=work_path,
            cached_checkpoints=cached_checkpoints,
        )
    else:
        model, _ = load_model(model_name)

    print(
        f"[GPU {gpu_id}][member {m + 1}/{n_members_total}] running rollout",
        flush=True,
    )
    ds = _run_one_member(model, data, init_time, steps, ensemble_id=m, seed=seed + m)
    ds = _filter_levels(ds, output_levels)
    ds = _filter_vars(ds, output_vars)

    member_zarr = output_path / f"member_{m:03d}.zarr"
    ds.to_zarr(member_zarr, mode="w", consolidated=True)
    print(
        f"[GPU {gpu_id}][member {m + 1}/{n_members_total}] wrote {member_zarr}",
        flush=True,
    )


_INNER_CHUNKS: dict[str, int] = {
    "ensemble": 1,
    "level": 1,
    "init_time": 1,
    "lead_time": 6,
    "latitude": 90,
    "longitude": 360,
}


def _merge_member_zarrs(tmp_dir: Path, output: Path, n_members: int) -> None:
    """Concatenate per-member zarr files into a single sharded zarr v3 store."""
    import math
    import shutil

    import zarr
    from zarr.storage import LocalStore

    members = []
    for m in range(n_members):
        member_zarr = tmp_dir / f"member_{m:03d}.zarr"
        members.append(xr.open_zarr(member_zarr, consolidated=True))
    ds = xr.concat(members, dim="ensemble")

    store = LocalStore(str(output))
    root = zarr.open_group(store, mode="w", zarr_format=3)

    for vname in ds.data_vars:
        da = ds[vname]
        shape = da.shape
        inner = tuple(min(s, _INNER_CHUNKS.get(d, s)) for d, s in zip(da.dims, shape))
        shards = tuple(math.ceil(s / c) * c for s, c in zip(shape, inner))
        dtype = "float32" if da.dtype == np.float64 else str(da.dtype)

        arr = root.create_array(
            vname,
            shape=shape,
            chunks=inner,
            shards=shards,
            dtype=dtype,
            dimension_names=da.dims,
            attributes=dict(da.attrs),
        )
        arr[:] = da.values.astype(dtype)

    for cname in ds.coords:
        if cname not in ds.data_vars:
            ca = ds.coords[cname]
            values = ca.values
            arr = root.create_array(
                cname,
                shape=values.shape,
                dtype=str(ca.dtype),
                dimension_names=ca.dims if ca.dims else (cname,),
                attributes=dict(ca.attrs),
            )
            arr[:] = values

    zarr.consolidate_metadata(store)
    shutil.rmtree(tmp_dir)


def _run_members_sequential(
    model_name: str,
    init_time: datetime,
    steps: int,
    n_members: int,
    ic_magnitude: float,
    weight_magnitude: float,
    layer: str | None,
    data_source: str,
    seed: int,
    work_dir: Path,
    output: Path,
    output_levels: list[int] | None,
    output_vars: list[str] | None,
    cached_checkpoints: dict[str, str] | None,
) -> None:
    shared_model = None if weight_magnitude > 0 else load_model(model_name)[0]
    cached_ic: xr.Dataset | None = None
    tmp_dir = work_dir / "_seq_members"
    tmp_dir.mkdir(parents=True, exist_ok=True)

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
                cached_checkpoints=cached_checkpoints,
            )
        )

        print(f"[member {m + 1}/{n_members}] running rollout")
        ds = _run_one_member(model, data, init_time, steps, ensemble_id=m, seed=seed + m)
        ds = _filter_levels(ds, output_levels)
        ds = _filter_vars(ds, output_vars)

        member_zarr = tmp_dir / f"member_{m:03d}.zarr"
        ds.to_zarr(member_zarr, mode="w", consolidated=True)
        print(f"[member {m + 1}/{n_members}] wrote {member_zarr}")

    print("[sequential] Merging member outputs...")
    _merge_member_zarrs(tmp_dir, output, n_members)
    print(f"[sequential] Final output: {output}")


def _run_members_parallel(
    model_name: str,
    init_time: datetime,
    steps: int,
    n_members: int,
    n_gpus: int,
    ic_magnitude: float,
    weight_magnitude: float,
    layer: str | None,
    data_source: str,
    seed: int,
    work_dir: Path,
    output: Path,
    output_levels: list[int] | None,
    output_vars: list[str] | None,
    cached_checkpoints: dict[str, str] | None,
) -> None:
    """Run members in parallel across GPUs, one process per member.

    Members are launched in rounds of `n_gpus`. Each process runs exactly
    one member then exits, fully releasing all runtime state (including
    JAX/XLA compiled programs and GPU memory).
    """
    import torch.multiprocessing as mp

    tmp_dir = work_dir / "_member_outputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build rounds: each round runs up to n_gpus members in parallel
    rounds: list[list[tuple[int, int]]] = []  # [(gpu_id, member_id), ...]
    for m in range(n_members):
        round_idx = m // n_gpus
        if round_idx >= len(rounds):
            rounds.append([])
        gpu_id = m % n_gpus
        rounds[round_idx].append((gpu_id, m))

    print(f"[main] Launching {n_members} members across {n_gpus} GPUs in {len(rounds)} rounds")

    ctx = mp.get_context("spawn")
    worker_args = (
        model_name,
        init_time,
        steps,
        n_members,
        ic_magnitude,
        weight_magnitude,
        layer,
        data_source,
        seed,
        str(work_dir),
        str(tmp_dir),
        output_levels,
        output_vars,
        cached_checkpoints,
    )

    for round_idx, round_members in enumerate(rounds):
        print(
            f"[main] Round {round_idx + 1}/{len(rounds)}: "
            f"members {[m for _, m in round_members]}",
            flush=True,
        )
        processes = []
        for gpu_id, member_id in round_members:
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, member_id, *worker_args),
            )
            p.start()
            processes.append((gpu_id, member_id, p))

        errors = []
        for gpu_id, member_id, p in processes:
            p.join()
            if p.exitcode != 0:
                errors.append(f"GPU {gpu_id} member {member_id} exited with code {p.exitcode}")

        if errors:
            raise RuntimeError(
                f"Parallel inference failed in round {round_idx + 1}:\n" + "\n".join(errors)
            )

    print("[main] Merging member outputs...")
    _merge_member_zarrs(tmp_dir, output, n_members)
    print(f"[main] Final output: {output}")


def _prefetch_ic_data(
    model_name: str,
    data_source: str,
    init_time: datetime,
) -> None:
    """Pre-fetch IC data to populate the fsspec disk cache.

    Discovers the model's input variables via a subprocess (so no GPU memory
    is consumed in the main process), then fetches them from the remote source
    (ARCO/CDS). Workers subsequently read from the already-cached data,
    avoiding concurrent downloads and blosc race conditions.
    """
    import json
    import subprocess
    import sys

    # Write a helper script to /tmp and run it as a file. Using `python -c`
    # with inline code can pick up stale bytecode from the container's
    # installed package instead of the bind-mounted source.
    helper = (
        "import json, os, sys\n"
        "import numpy as np\n"
        f"os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"
        "from ai_models_ensembles.e2s_models import load_model\n"
        f"m, _ = load_model('{model_name}')\n"
        "c = m.input_coords()\n"
        "lts = [int(np.timedelta64(lt, 'ns') / np.timedelta64(1, 'ns'))"
        " for lt in c.get('lead_time', [0])]\n"
        "print(json.dumps({'variables': list(c.get('variable', [])),"
        " 'lead_times': lts}))\n"
    )
    helper_path = Path("/tmp") / "_discover_vars.py"
    try:
        helper_path.write_text(helper)
        result = subprocess.run(
            [sys.executable, str(helper_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        helper_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"[main] Could not discover input vars: {result.stderr[-500:]}", flush=True)
        return

    info = json.loads(result.stdout.strip().split("\n")[-1])
    variables = info["variables"]
    lead_times_ns = [np.timedelta64(lt, "ns") for lt in info["lead_times"]]

    if not variables:
        return

    init_np = np.datetime64(init_time, "ns")
    fetch_times = np.array(sorted(set(init_np + lt for lt in lead_times_ns)))

    print(
        f"[main] Pre-fetching {len(variables)} variables x {len(fetch_times)} times "
        f"from {data_source} to warm cache...",
        flush=True,
    )
    source = build_data_source(data_source)
    source(fetch_times, variables)
    print("[main] IC cache warmed.", flush=True)


def run_inference(
    model_name: str,
    init_time: datetime,
    lead_hours: int,
    output: Path,
    *,
    n_members: int = 1,
    ic_magnitude: float = 0.0,
    weight_magnitude: float = 0.0,
    layer: str | None = None,
    data_source: str = "arco",
    seed: int = 0,
    work_dir: Path | None = None,
    output_levels: list[int] | None = None,
    output_vars: list[str] | None = None,
) -> Path:
    """Run an earth2studio model and emit a SwissClim-format zarr at `output`.

    When multiple GPUs are available and n_members > 1, members are distributed
    across GPUs (one process per GPU, round-robin). Checkpoint files are
    downloaded once and copied locally per member to avoid redundant downloads.
    """
    spec = get_spec(model_name)
    steps = steps_from_hours(spec, lead_hours)
    layer = _parse_layer_spec(layer, model_name=model_name)
    output = Path(output)
    work_dir = Path(work_dir or output.parent / "_e2s_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    cached_checkpoints = None
    if weight_magnitude > 0:
        print("[main] Pre-caching model checkpoints...")
        cached_checkpoints = cache_default_checkpoints(model_name)
        print(f"[main] Cached {len(cached_checkpoints)} checkpoint file(s)")

    n_gpus = _available_gpus()

    # Pre-fetch IC data to warm fsspec cache before spawning GPU workers
    if n_members > 1 and n_gpus > 1:
        _prefetch_ic_data(model_name, data_source, init_time)

    if n_members > 1 and n_gpus > 1:
        _run_members_parallel(
            model_name=model_name,
            init_time=init_time,
            steps=steps,
            n_members=n_members,
            n_gpus=n_gpus,
            ic_magnitude=ic_magnitude,
            weight_magnitude=weight_magnitude,
            layer=layer,
            data_source=data_source,
            seed=seed,
            work_dir=work_dir,
            output=output,
            output_levels=output_levels,
            output_vars=output_vars,
            cached_checkpoints=cached_checkpoints,
        )
    else:
        _run_members_sequential(
            model_name=model_name,
            init_time=init_time,
            steps=steps,
            n_members=n_members,
            ic_magnitude=ic_magnitude,
            weight_magnitude=weight_magnitude,
            layer=layer,
            data_source=data_source,
            seed=seed,
            work_dir=work_dir,
            output=output,
            output_levels=output_levels,
            output_vars=output_vars,
            cached_checkpoints=cached_checkpoints,
        )

    # Clean up work directory (perturbed weight copies, etc.)
    import shutil

    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    return output


__all__ = ["run_inference"]
