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
    coarse_mode_cut: int | None = None,
    coarse_mode_skip_first: int = 0,
    graph_coarse_sigma: float = 0.0,
    graph_coarse_nodes: int = 0,
) -> Any:
    # GraphCast Phase 3: install the coarse-perturbation subclass BEFORE
    # load_model() runs. The subclass is monkey-patched onto
    # graphcast.GraphCast, which earth2studio's loader then instantiates.
    if graph_coarse_sigma > 0 and graph_coarse_nodes > 0:
        if not model_name.startswith("graphcast"):
            raise ValueError(
                "--graph-coarse-sigma only applies to graphcast models, "
                f"got model_name={model_name!r}"
            )
        from .graphcast_coarse_perturbation import install as _install_coarse_gc

        _install_coarse_gc(graph_coarse_sigma, graph_coarse_nodes)

    # SFNO Phase 6 fresh-per-step: skip the checkpoint perturbation entirely
    # so the model loads with default weights, and let the forward-pre-hook
    # in sfno_fresh_perturbation install the fresh noise per AR step.
    import os as _os_fresh

    _sfno_fresh_active = _os_fresh.environ.get("SFNO_FRESH", "0") == "1" and model_name == "sfno"

    if weight_magnitude <= 0 or _sfno_fresh_active:
        model, _ = load_model(model_name)
    else:
        pkg_dir = work_dir / f"weights_member_{member_id:03d}"
        package_root = materialise_perturbed_package(
            model_name=model_name,
            magnitude=weight_magnitude,
            layer=layer,
            seed=seed + member_id,
            out_dir=pkg_dir,
            cached_checkpoints=cached_checkpoints,
            coarse_mode_cut=coarse_mode_cut,
            coarse_mode_skip_first=coarse_mode_skip_first,
        )
        model, _ = load_model(model_name, package_root=package_root)

    # Per-member RNG seed for GraphCast coarse perturbation (JAX PRNGKey).
    if graph_coarse_sigma > 0 and graph_coarse_nodes > 0:
        import os as _os

        import jax

        if hasattr(model, "prng_key"):
            model.prng_key = jax.random.PRNGKey(seed + member_id)
        # Frozen-noise mode: also populate the module-level slot so the
        # subclass forward uses the same noise tensor at every rollout step.
        if _os.environ.get("GC_FROZEN", "0") == "1":
            from . import graphcast_coarse_perturbation as _gc_pert

            _gc_pert._FROZEN_MEMBER_KEY = jax.random.PRNGKey(seed + member_id)

    # SFNO Phase 6 fresh-per-step weight perturbation, gated by SFNO_FRESH=1.
    # Sigma comes from SFNO_FRESH_SIGMA, mode_cut from SFNO_FRESH_MODE_CUT.
    # When this branch fires, the checkpoint perturbation upstream is skipped
    # by the inference driver (it sets coarse_mode_cut=None in the perturb
    # call but keeps it here so the hook still knows where the mode boundary
    # is). See [[phase6-fresh-per-step-weight]].
    import os as _os2

    if _os2.environ.get("SFNO_FRESH", "0") == "1":
        from . import sfno_fresh_perturbation as _sfno_fresh

        _sfno_fresh.maybe_install_from_env(model, seed + member_id)
    return model


# -- Multi-GPU parallelism ---------------------------------------------------


def _available_gpus() -> int:
    """Detect the number of GPUs without initializing the CUDA runtime.

    Importing torch and calling torch.cuda.device_count() would initialize a
    CUDA context in the parent process. Spawned children then inherit a
    corrupted copy of the driver state, which causes SIGSEGV in later rounds.
    """
    import os
    import subprocess

    # 1. Honour explicit env-var override
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis:
        ids = [x for x in vis.split(",") if x.strip()]
        if ids:
            return len(ids)

    # 2. Ask nvidia-smi (no CUDA runtime init)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return max(len(out.strip().splitlines()), 1)
    except Exception:
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
    coarse_mode_cut: int | None = None,
    coarse_mode_skip_first: int = 0,
    graph_coarse_sigma: float = 0.0,
    graph_coarse_nodes: int = 0,
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
    import os as _os_gpu

    _sfno_fresh_gpu = _os_gpu.environ.get("SFNO_FRESH", "0") == "1" and model_name == "sfno"
    if weight_magnitude > 0 or graph_coarse_sigma > 0 or _sfno_fresh_gpu:
        model = _model_for_member(
            model_name=model_name,
            weight_magnitude=weight_magnitude,
            layer=layer,
            member_id=m,
            seed=seed,
            work_dir=work_path,
            cached_checkpoints=cached_checkpoints,
            coarse_mode_cut=coarse_mode_cut,
            coarse_mode_skip_first=coarse_mode_skip_first,
            graph_coarse_sigma=graph_coarse_sigma,
            graph_coarse_nodes=graph_coarse_nodes,
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
    encoding = {}
    for vname, da in ds.data_vars.items():
        encoding[vname] = {
            "chunks": tuple(
                s if _INNER_CHUNKS.get(d, -1) == -1 else min(s, _INNER_CHUNKS[d])
                for d, s in zip(da.dims, da.shape)
            )
        }
    ds.to_zarr(member_zarr, mode="w", consolidated=True, encoding=encoding)
    print(
        f"[GPU {gpu_id}][member {m + 1}/{n_members_total}] wrote {member_zarr}",
        flush=True,
    )

    # Explicit GPU cleanup so the driver fully releases resources before
    # the next round's processes claim the same GPU (prevents SIGSEGV on GH200).
    del model, data, ds
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    import gc

    gc.collect()


_INNER_CHUNKS: dict[str, int] = {
    "ensemble": -1,
    "level": 1,
    "init_time": 1,
    "lead_time": 1,
    "latitude": -1,
    "longitude": -1,
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
        inner = tuple(
            s if _INNER_CHUNKS.get(d, -1) == -1 else min(s, _INNER_CHUNKS[d])
            for d, s in zip(da.dims, shape)
        )
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
    coarse_mode_cut: int | None = None,
    coarse_mode_skip_first: int = 0,
    graph_coarse_sigma: float = 0.0,
    graph_coarse_nodes: int = 0,
) -> None:
    # GraphCast Phase 3 (graph_coarse_sigma > 0) needs per-member RNG even
    # though weights are unperturbed, so we cannot use shared_model.
    # SFNO Phase 6 fresh-per-step installs per-member forward hooks so it
    # also needs a fresh model per member.
    import os as _os_seq

    _sfno_fresh_seq = _os_seq.environ.get("SFNO_FRESH", "0") == "1" and model_name == "sfno"
    use_shared_model = weight_magnitude <= 0 and graph_coarse_sigma <= 0 and not _sfno_fresh_seq
    shared_model = load_model(model_name)[0] if use_shared_model else None
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
                coarse_mode_cut=coarse_mode_cut,
                coarse_mode_skip_first=coarse_mode_skip_first,
                graph_coarse_sigma=graph_coarse_sigma,
                graph_coarse_nodes=graph_coarse_nodes,
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
    coarse_mode_cut: int | None = None,
    coarse_mode_skip_first: int = 0,
    graph_coarse_sigma: float = 0.0,
    graph_coarse_nodes: int = 0,
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
        coarse_mode_cut,
        coarse_mode_skip_first,
        graph_coarse_sigma,
        graph_coarse_nodes,
    )

    import gc

    # GH200 multi-process CUDA startup occasionally returns SIGSEGV (-11)
    # in transient races (e.g. HF Hub cache, NCCL init, torch_harmonics
    # state). One retry of the *failed members only* is enough to recover
    # in the vast majority of cases. After that we give up and let the
    # whole job fail so the user investigates rather than burning compute.
    RETRY_BACKOFF_SEC = 30
    import time

    def _launch_round(members: list[tuple[int, int]]) -> list[tuple[tuple[int, int], int]]:
        """Spawn one process per (gpu_id, member_id), join, return failures.

        Returns a list of ((gpu_id, member_id), exitcode) for every member
        whose process exited non-zero. Empty list means full success.
        """
        ctx = mp.get_context("spawn")
        processes = []
        for gpu_id, member_id in members:
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, member_id, *worker_args),
            )
            p.start()
            processes.append((gpu_id, member_id, p))

        failures = []
        for gpu_id, member_id, p in processes:
            p.join()
            if p.exitcode != 0:
                failures.append(((gpu_id, member_id), p.exitcode))

        for _, _, p in processes:
            p.close()
        del processes, ctx
        gc.collect()
        return failures

    for round_idx, round_members in enumerate(rounds):
        print(
            f"[main] Round {round_idx + 1}/{len(rounds)}: "
            f"members {[m for _, m in round_members]}",
            flush=True,
        )

        failures = _launch_round(round_members)

        if failures:
            err_lines = [
                f"GPU {gpu_id} member {member_id} exited with code {ec}"
                for (gpu_id, member_id), ec in failures
            ]
            print(
                f"[main] Round {round_idx + 1} had {len(failures)} failure(s); "
                f"retrying ONCE after {RETRY_BACKOFF_SEC}s. Errors:\n  " + "\n  ".join(err_lines),
                flush=True,
            )
            time.sleep(RETRY_BACKOFF_SEC)
            retry_members = [m for m, _ in failures]
            failures = _launch_round(retry_members)
            if failures:
                err_lines = [
                    f"GPU {gpu_id} member {member_id} exited with code {ec}"
                    for (gpu_id, member_id), ec in failures
                ]
                raise RuntimeError(
                    f"Parallel inference failed in round {round_idx + 1} "
                    f"after 1 retry:\n" + "\n".join(err_lines)
                )

        # Brief pause between rounds to let the GPU driver fully release
        # contexts from exited children (prevents SIGSEGV on GH200/NATTEN).
        # AND clear any leaked Python multiprocessing semaphores from
        # /dev/shm. resource_tracker warns "leaked semaphores" at crash;
        # accumulated leaks across rounds corrupt the spawn context and
        # the final round (e.g. round 3 of 3 for AIFS) SIGSEGVs in all
        # GPU workers simultaneously. Same root cause as the SFNO week-
        # helper IPC cleanup, but here applied per-round inside a single
        # python process.
        if round_idx < len(rounds) - 1:
            import glob
            import os

            for pat in ("sem.mp-*", "sem.pym-*", "sem.tmp.*"):
                for f in glob.glob(f"/dev/shm/{pat}"):
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
            time.sleep(5)

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
    coarse_mode_cut: int | None = None,
    coarse_mode_skip_first: int = 0,
    graph_coarse_sigma: float = 0.0,
    graph_coarse_nodes: int = 0,
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

    # try/finally so work_dir always gets removed, even on inference failure.
    # Each member's intermediate zarr inside `_e2s_work/_member_outputs/` is
    # ~4k file inodes (unsharded zarr v3); 10 members per job. A failed job
    # used to leak the full ~44k inodes per launch. This burned through the
    # a122 project inode quota overnight when 23 AIFS Phase 1+2 jobs failed
    # in round 3 with state-accumulation SIGSEGVs.
    import shutil

    try:
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
                coarse_mode_cut=coarse_mode_cut,
                coarse_mode_skip_first=coarse_mode_skip_first,
                graph_coarse_sigma=graph_coarse_sigma,
                graph_coarse_nodes=graph_coarse_nodes,
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
                coarse_mode_cut=coarse_mode_cut,
                coarse_mode_skip_first=coarse_mode_skip_first,
                graph_coarse_sigma=graph_coarse_sigma,
                graph_coarse_nodes=graph_coarse_nodes,
            )
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    return output


__all__ = ["run_inference"]
