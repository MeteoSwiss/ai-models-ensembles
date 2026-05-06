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


def _copy_package_to_local(package: Any, dest: Path) -> None:
    """Copy every file under a Package's fsspec root into a local directory."""
    fs = package.fs
    root = package.root
    dest.mkdir(parents=True, exist_ok=True)
    for rel in _list_package_files(package):
        src = f"{root.rstrip('/')}/{rel}"
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with fs.open(src, "rb") as fin, open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout)


def _perturb_array(arr: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if not np.issubdtype(arr.dtype, np.floating):
        return arr
    noise = rng.standard_normal(size=arr.shape).astype(arr.dtype)
    return arr * (1.0 + sigma * noise)


def _select_indices(total: int, layer: int | None) -> list[int]:
    if layer is None:
        return list(range(total))
    if layer < 0 or layer >= total:
        raise ValueError(f"--layer {layer} out of range; package exposes {total} weight tensors.")
    return [layer]


def _perturb_npz(path: Path, sigma: float, rng: np.random.Generator, layer: int | None) -> None:
    with np.load(path, allow_pickle=False) as f:
        keys = list(f.files)
        arrays = {k: f[k] for k in keys}
    targets = _select_indices(len(keys), layer)
    for i in targets:
        arrays[keys[i]] = _perturb_array(arrays[keys[i]], sigma, rng)
    np.savez(path, **arrays)


def _perturb_torch(path: Path, sigma: float, rng: np.random.Generator, layer: int | None) -> None:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj if isinstance(obj, dict) else getattr(obj, "state_dict", lambda: {})()
    if not isinstance(state, dict):
        raise RuntimeError(f"Unrecognised torch checkpoint structure: {type(obj)}")
    keys = sorted(state.keys())
    targets = _select_indices(len(keys), layer)
    for i in targets:
        k = keys[i]
        t = state[k]
        if hasattr(t, "numpy"):
            arr = t.detach().cpu().numpy()
            new = _perturb_array(arr, sigma, rng)
            state[k] = torch.from_numpy(new)
    torch.save(obj if isinstance(obj, dict) else state, path)


def _perturb_safetensors(
    path: Path, sigma: float, rng: np.random.Generator, layer: int | None
) -> None:
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
    ".safetensors": _perturb_safetensors,
}


def _checkpoint_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _HANDLERS:
            yield p


def materialise_perturbed_package(
    model_name: str,
    magnitude: float,
    layer: int | None,
    seed: int,
    out_dir: Path,
) -> str:
    """Download + perturb + write a model's checkpoint package locally.

    Returns the local root path to be passed as `package_root` to
    `e2s_models.load_model(name, package_root=...)`.
    """
    from .e2s_models import import_class, get_spec

    spec = get_spec(model_name)
    cls = import_class(spec)
    package = cls.load_default_package()

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    _copy_package_to_local(package, out_dir)

    rng = np.random.default_rng(seed)
    ckpts = list(_checkpoint_files(out_dir))
    if not ckpts:
        raise RuntimeError(
            f"No checkpoint files found under {out_dir}; supported suffixes: "
            f"{sorted(_HANDLERS)}. Inspect the package layout for {model_name}."
        )
    for ckpt in ckpts:
        handler = _HANDLERS[ckpt.suffix.lower()]
        handler(ckpt, magnitude, rng, layer)
    return str(out_dir)


__all__ = [
    "perturb_initial_conditions",
    "materialise_perturbed_package",
]
