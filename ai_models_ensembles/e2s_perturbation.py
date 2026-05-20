"""Initial-condition and model-weight perturbations for earth2studio models.

Two independent layers:

* `perturb_initial_conditions(ds, magnitude, seed)` - additive Gaussian noise
  scaled by each variable's spatial std. Operates on an in-memory dataset
  (e2s lexicon) so callers can wrap the result in `XarrayDataSource`.

* `materialise_perturbed_package(model_name, magnitude, layer, seed, out_dir)`
  - downloads the default e2s `Package` for a model, walks all checkpoint
  files, multiplies each tensor by `(1 + N(0, magnitude))`, mirrors the
  package layout under `out_dir`, and returns the local root so it can be
  passed to `e2s_models.load_model(name, package_root=...)`.

The weight-perturbation walker is intentionally generic: it inspects the
suffix of every file in the package and dispatches to a per-format handler
(.npz, .pt, .pth, .ckpt, .safetensors). For `--layer N`, only the N-th
matching tensor in deterministic file/key order is perturbed; pass
`layer=None` to perturb every tensor.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


def perturb_initial_conditions(
    ds: xr.Dataset,
    magnitude: float,
    seed: int = 0,
) -> xr.Dataset:
    """Return a copy of `ds` with multiplicative Gaussian noise on every data var.

    Each variable is perturbed independently:
        var' = var * (1 + N(0, magnitude))   per grid point.
    Use a moderate magnitude (1e-3 ... 1e-2) for IC sensitivity studies.
    """
    if magnitude <= 0:
        return ds
    rng = np.random.default_rng(seed)
    out: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        noise = rng.standard_normal(size=da.shape).astype(da.dtype)
        out[name] = da * (1.0 + magnitude * noise)
    return xr.Dataset(out, coords=ds.coords, attrs=ds.attrs)


# -- Weight perturbation -----------------------------------------------------


def _list_package_files(package: Any) -> list[str]:
    """Return relative paths of files inside an earth2studio Package."""
    fs = package.fs
    root = package.root
    return sorted(p[len(root) :].lstrip("/") for p in fs.find(root) if not fs.isdir(p))


def _copy_package_to_local(package: Any, dest: Path, suffixes: set[str] | None = None) -> None:
    """Copy files under a Package's fsspec root into a local directory.

    If `suffixes` is given, only files with matching extensions are copied.
    This avoids downloading multi-GB repos when only checkpoint files are needed.
    """
    fs = package.fs
    root = package.root
    dest.mkdir(parents=True, exist_ok=True)
    for rel in _list_package_files(package):
        if suffixes and Path(rel).suffix.lower() not in suffixes:
            continue
        src = f"{root.rstrip('/')}/{rel}"
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with fs.open(src, "rb") as fin, open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout)


def _perturb_array(arr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    # np.inexact covers both np.floating and np.complexfloating, so complex
    # spectral conv weights (e.g. SFNO `*.filter.filter.weight`) get perturbed.
    if not np.issubdtype(arr.dtype, np.inexact):
        return arr
    if np.issubdtype(arr.dtype, np.complexfloating):
        # CN(0,1) complex noise: real and imaginary parts iid N(0, 1/2) so
        # E[|noise|^2] = 1, matching the real case statistically.
        real_dtype = np.float32 if arr.dtype == np.complex64 else np.float64
        inv_sqrt2 = (
            np.float32(1.0 / np.sqrt(2.0)) if real_dtype is np.float32 else 1.0 / np.sqrt(2.0)
        )
        noise_re = (rng.standard_normal(size=arr.shape).astype(real_dtype)) * inv_sqrt2
        noise_im = (rng.standard_normal(size=arr.shape).astype(real_dtype)) * inv_sqrt2
        noise = np.empty(arr.shape, dtype=arr.dtype)
        noise.real = noise_re
        noise.imag = noise_im
    else:
        noise = rng.standard_normal(size=arr.shape).astype(arr.dtype)
    # Force result back to arr.dtype to defeat numpy's silent upcasting
    # of Python scalars in mixed arithmetic (matters for complex64).
    return (arr * (1.0 + sigma * noise)).astype(arr.dtype, copy=False)


def _is_spectral_conv_tensor(name: str) -> bool:
    """Return True if `name` matches SFNO's spherical spectral convolution weight.

    Used by Phase 3 to restrict coarse-mode perturbation to those tensors.
    """
    return name.endswith(".filter.filter.weight")


def _perturb_array_low_modes(
    arr: np.ndarray, sigma: float, rng: np.random.Generator, mode_cut: int
) -> np.ndarray:
    """Perturb only the first `mode_cut` entries of the last axis of `arr`.

    Used by Phase 3 SFNO: spectral conv weights have shape (..., lmax) and
    we want to perturb only the low-l (planetary / large-synoptic) modes.
    The remaining modes are left untouched, preserving the model's
    representation of finer scales.
    """
    if not np.issubdtype(arr.dtype, np.inexact):
        return arr
    if mode_cut <= 0 or mode_cut > arr.shape[-1]:
        raise ValueError(
            f"mode_cut={mode_cut} out of range for tensor with last axis = {arr.shape[-1]}"
        )
    sub = arr[..., :mode_cut].copy()
    perturbed = _perturb_array(sub, sigma, rng)
    out = arr.copy()
    out[..., :mode_cut] = perturbed
    return out


# Model-specific architectural layer groups.
# For .ckpt/.tar/.pt/.pth/.safetensors: keys are sorted alphabetically;
# indices refer to that order over tensors with `.numpy()` (includes complex).
# For .npz: keys are in stored order (`f.files`), all dtypes counted but
# non-float and `model_config:`/`task_config:` keys are passed through.
# Determined empirically from actual checkpoint dumps; see
# memory/checkpoint_perturbation_audit.md.
_MODEL_LAYER_GROUPS: dict[str, dict[str, tuple[int, int]]] = {
    "aurora": {
        # 644 float tensors: backbone (594) | decoder (17) | encoder (33)
        "backbone": (0, 594),
        "decoder": (594, 611),
        "encoder": (611, 644),
        # Phase 3 coarse-scale target: the deepest encoder block of the
        # Swin-3D U-net (backbone.encoder_layers.2), 2048-channel /
        # 4 deg tokens. Indices 494:590 (96 tensors) verified from
        # aurora_keys.log -- contiguous slice within "backbone".
        "coarse_encoder": (494, 590),
    },
    "graphcast_operational": {
        # 320 NPZ stored keys: 264 weight tensors + 3 buggy hyperparam floats
        # + 53 non-float config keys. STORED order (np.savez f.files), not
        # sorted. `_perturb_npz` now skips `model_config:` / `task_config:`
        # prefixed keys, so the 3 hyperparam floats at stored indices
        # 264, 269, 270 are excluded by name rather than by index range.
        "g2m": (0, 36),  # params:grid2mesh_gnn
        "m2g": (36, 66),  # params:mesh2grid_gnn
        "m2m": (66, 264),  # params:mesh_gnn
    },
    "sfno": {
        # 87 tensors total (79 float32 + 8 complex64 spectral conv filters).
        # Architectural split confirmed from direct checkpoint dump 2026-05-18.
        # Sorted-key order:
        #   blocks (8 SFNO blocks x 10 tensors each)            -> 0..79
        #   decoder.fwd.* (3 tensors)                            -> 80..82
        #   encoder.fwd.* (3 tensors)                            -> 83..85
        #   residual_transform.weight (1 tensor)                 -> 86
        "encoder": (83, 86),
        "processor": (0, 80),
        "decoder": (80, 83),
        "residual": (86, 87),
    },
}


def _parse_layer_spec(spec: str | int | None, model_name: str | None = None) -> str | None:
    """Normalise a layer specification to a canonical string form.

    Accepted formats:
      None / "all"         -> None  (perturb every tensor)
      "42"  / 42           -> "42"  (single tensor index)
      "10:50"              -> "10:50" (index range [10, 50))
      "0.0:0.33"           -> "0.0:0.33" (fraction range, first third)
      "encoder" / "m2m"    -> "611:644" (named group, requires model_name)
    """
    if spec is None:
        return None
    s = str(spec).strip().lower()
    if s in ("", "all", "none"):
        return None

    # Named architectural group (not a pure number, not a range with ':')
    if model_name and not s.replace(".", "").isdigit() and ":" not in s:
        groups = _MODEL_LAYER_GROUPS.get(model_name, {})
        if s not in groups:
            available = sorted(groups.keys()) if groups else ["(none defined)"]
            raise ValueError(f"Unknown layer group '{s}' for {model_name}. Available: {available}")
        lo, hi = groups[s]
        return f"{lo}:{hi}"

    return s


def _select_indices(total: int, layer: str | None) -> list[int]:
    if layer is None:
        return list(range(total))

    # Fraction range: "0.0:0.33"
    if ":" in layer and "." in layer:
        lo_s, hi_s = layer.split(":", 1)
        lo_f, hi_f = float(lo_s), float(hi_s)
        lo_i, hi_i = int(lo_f * total), int(hi_f * total)
        hi_i = max(hi_i, lo_i + 1)  # at least one tensor
        return list(range(lo_i, min(hi_i, total)))

    # Index range: "10:50"
    if ":" in layer:
        lo_s, hi_s = layer.split(":", 1)
        lo_i, hi_i = int(lo_s), int(hi_s)
        if lo_i < 0 or hi_i > total:
            raise ValueError(
                f"--layer {layer} out of range; package exposes {total} weight tensors."
            )
        return list(range(lo_i, hi_i))

    # Single index: "42"
    idx = int(layer)
    if idx < 0 or idx >= total:
        raise ValueError(f"--layer {idx} out of range; package exposes {total} weight tensors.")
    return [idx]


def _perturb_npz(
    path: Path,
    sigma: float,
    rng: np.random.Generator,
    layer: str | None,
    coarse_mode_cut: int | None = None,
) -> None:
    if coarse_mode_cut is not None:
        raise NotImplementedError(
            ".npz format (GraphCast) does not currently support --coarse-mode-cut"
        )
    with np.load(path, allow_pickle=False) as f:
        keys = list(f.files)
        arrays = {k: f[k] for k in keys}
    targets = _select_indices(len(keys), layer)
    for i in targets:
        k = keys[i]
        # GraphCast .npz includes float-typed model_config/task_config
        # scalars (resolution, edge-normalisation factors etc.) that define
        # graph topology, not learned weights. Skip them.
        if k.startswith(("model_config:", "task_config:")):
            continue
        arrays[k] = _perturb_array(arrays[k], sigma, rng)
    np.savez(path, **arrays)


def _extract_state_dict(obj: Any) -> dict:
    """Extract the flat tensor state dict from a checkpoint object.

    Handles both plain state dicts and training checkpoints where the model
    weights are nested under keys like 'model_state' or 'state_dict'.
    """
    if not isinstance(obj, dict):
        sd = getattr(obj, "state_dict", None)
        if callable(sd):
            return sd()
        raise RuntimeError(f"Unrecognised torch checkpoint structure: {type(obj)}")

    # Check for nested model state (e.g. SFNO training checkpoints)
    for nested_key in ("model_state", "state_dict", "model_state_dict", "model"):
        if nested_key in obj and isinstance(obj[nested_key], dict):
            candidate = obj[nested_key]
            # Verify it actually contains tensors, not just metadata
            import torch

            if any(isinstance(v, torch.Tensor) for v in list(candidate.values())[:5]):
                return candidate

    return obj


def _perturb_torch(
    path: Path,
    sigma: float,
    rng: np.random.Generator,
    layer: str | None,
    coarse_mode_cut: int | None = None,
) -> None:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)
    keys = sorted(k for k, v in state.items() if hasattr(v, "numpy"))
    if not keys:
        raise RuntimeError(
            f"No float tensors found in {path.name}. "
            f"Top-level keys: {sorted(state.keys())[:10]}"
        )
    targets = _select_indices(len(keys), layer)
    n_spectral_touched = 0
    for i in targets:
        k = keys[i]
        arr = state[k].detach().cpu().numpy()
        if coarse_mode_cut is not None:
            # Phase 3 SFNO mode: skip tensors that aren't spectral conv
            # weights; for those that are, perturb only the first
            # `coarse_mode_cut` entries of the last (l) axis.
            if not _is_spectral_conv_tensor(k):
                continue
            new = _perturb_array_low_modes(arr, sigma, rng, coarse_mode_cut)
            n_spectral_touched += 1
        else:
            new = _perturb_array(arr, sigma, rng)
        state[k] = torch.from_numpy(new)
    if coarse_mode_cut is not None and n_spectral_touched == 0:
        raise RuntimeError(
            f"--coarse-mode-cut set but no spectral conv tensors "
            f"(*.filter.filter.weight) found in {path.name}. "
            f"Selected {len(targets)} of {len(keys)} keys; widen --layer or check model."
        )
    torch.save(obj, path)


def _perturb_safetensors(
    path: Path,
    sigma: float,
    rng: np.random.Generator,
    layer: str | None,
    coarse_mode_cut: int | None = None,
) -> None:
    if coarse_mode_cut is not None:
        raise NotImplementedError(
            ".safetensors format does not currently support --coarse-mode-cut"
        )
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(path))
    keys = sorted(tensors.keys())
    targets = _select_indices(len(keys), layer)
    for i in targets:
        k = keys[i]
        tensors[k] = _perturb_array(tensors[k], sigma, rng)
    save_file(tensors, str(path))


_HANDLERS = {
    ".npz": _perturb_npz,
    ".pt": _perturb_torch,
    ".pth": _perturb_torch,
    ".ckpt": _perturb_torch,
    ".tar": _perturb_torch,
    ".safetensors": _perturb_safetensors,
}


def _checkpoint_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _HANDLERS:
            yield p


# All package files needed per model. Listed explicitly because NGC/GCS
# filesystems don't support glob/find. All files are copied into the local
# perturbed package directory; only those matching _HANDLERS get perturbed.
_MODEL_PACKAGE_FILES: dict[str, list[str]] = {
    "aurora": [
        "aurora-0.25-pretrained.ckpt",
    ],
    "sfno": [
        "config.json",
        "global_means.npy",
        "global_stds.npy",
        "orography.nc",
        "land_mask.nc",
        "training_checkpoints/best_ckpt_mp0.tar",
    ],
    "graphcast_operational": [
        "stats/diffs_stddev_by_level.nc",
        "stats/mean_by_level.nc",
        "stats/stddev_by_level.nc",
        "params/GraphCast_operational - ERA5-HRES 1979-2021"
        " - resolution 0.25 - pressure levels 13"
        " - mesh 2to6 - precipitation output only.npz",
        "dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc",
    ],
}


def cache_default_checkpoints(model_name: str) -> dict[str, str]:
    """Download default checkpoint files once, return {rel_path: local_cached_path}.

    Call once in the main process before spawning GPU workers. Workers pass the
    result to ``materialise_perturbed_package(cached_checkpoints=...)`` to copy
    from local cache instead of re-downloading.
    """
    from .e2s_models import get_spec, import_class

    spec = get_spec(model_name)
    cls = import_class(spec)
    package = cls.load_default_package()

    ckpt_suffixes = set(_HANDLERS)
    if model_name in _MODEL_PACKAGE_FILES:
        files = _MODEL_PACKAGE_FILES[model_name]
    else:
        files = [
            rel for rel in _list_package_files(package) if Path(rel).suffix.lower() in ckpt_suffixes
        ]

    cached: dict[str, str] = {}
    for rel in files:
        cached[rel] = package.resolve(rel)
    return cached


def materialise_perturbed_package(
    model_name: str,
    magnitude: float,
    layer: str | None,
    seed: int,
    out_dir: Path,
    cached_checkpoints: dict[str, str] | None = None,
    coarse_mode_cut: int | None = None,
) -> str:
    """Download + perturb + write a model's checkpoint package locally.

    Returns the local root path to be passed as `package_root` to
    `e2s_models.load_model(name, package_root=...)`.

    If ``cached_checkpoints`` is provided (from `cache_default_checkpoints`),
    files are copied from the local cache instead of re-downloading.

    If ``coarse_mode_cut`` is set (Phase 3 SFNO), perturbation is restricted
    to tensors whose names match `*.filter.filter.weight` and to the first
    ``coarse_mode_cut`` entries of their last axis. Other tensors are
    passed through unchanged regardless of ``layer``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cached_checkpoints is None:
        cached_checkpoints = cache_default_checkpoints(model_name)

    for rel, cached_path in cached_checkpoints.items():
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, target)

    rng = np.random.default_rng(seed)
    ckpts = list(_checkpoint_files(out_dir))
    if not ckpts:
        raise RuntimeError(
            f"No checkpoint files found under {out_dir}; supported suffixes: "
            f"{sorted(_HANDLERS)}. Inspect the package layout for {model_name}."
        )
    for ckpt in ckpts:
        handler = _HANDLERS[ckpt.suffix.lower()]
        handler(ckpt, magnitude, rng, layer, coarse_mode_cut=coarse_mode_cut)
    return str(out_dir)


__all__ = [
    "_parse_layer_spec",
    "cache_default_checkpoints",
    "materialise_perturbed_package",
    "perturb_initial_conditions",
]
