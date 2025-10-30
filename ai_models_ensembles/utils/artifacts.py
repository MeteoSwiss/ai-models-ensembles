"""Utilities for persisting plotting artefacts alongside rendered figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "ensure_dir",
    "sanitize_token",
    "format_level_suffix",
    "level_to_attr",
    "build_output_filename",
    "save_dataset",
    "save_npz",
    "save_dataframe",
]


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (including parents) if missing and return it."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def sanitize_token(value: Any) -> str:
    """Convert *value* to a filesystem-friendly ASCII token."""
    if value is None:
        return "none"
    # Normalise common numeric cases
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "none"
        if float(value).is_integer():
            value = int(value)
    text = str(value).strip().lower()
    if not text:
        return "value"
    text = "_".join(text.split())
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-.")
    cleaned = "".join(ch for ch in text if ch in allowed)
    cleaned = cleaned.strip("_-.")
    return cleaned or "value"


def format_level_suffix(level: Any) -> str:
    """Return a suffix segment (including leading underscore) for pressure levels."""
    if level is None:
        return ""
    token = sanitize_token(level)
    return f"_{token}" if token else ""


def level_to_attr(level: Any) -> str:
    """Format *level* for safe NetCDF attribute storage."""
    if level is None:
        return ""
    if isinstance(level, (float, np.floating)) and np.isfinite(level):
        if float(level).is_integer():
            return str(int(level))
        return f"{float(level):.6g}"
    if isinstance(level, (int, np.integer)):
        return str(int(level))
    return str(level)


def _normalise_range(range_tuple: Sequence[str | int] | None) -> str:
    if not range_tuple:
        return ""
    start, *rest = range_tuple
    end = rest[0] if rest else None
    if end is None:
        return sanitize_token(start)
    return f"{sanitize_token(start)}-{sanitize_token(end)}"


def build_output_filename(
    *,
    metric: str,
    variable: str | None = None,
    level: str | int | None = None,
    qualifier: str | None = None,
    init_time_range: Sequence[str | int] | None = None,
    lead_time_range: Sequence[str | int] | None = None,
    ensemble: str | int | None = None,
    ext: str,
) -> str:
    """Compose a deterministic filename following the SwissClim naming scheme."""

    parts: list[str] = [sanitize_token(metric)]
    if variable not in (None, ""):
        parts.append(sanitize_token(variable))
    if level not in (None, ""):
        parts.append(sanitize_token(level))
    if qualifier not in (None, ""):
        parts.append(sanitize_token(qualifier))

    init_segment = _normalise_range(init_time_range)
    if init_segment:
        parts.append(f"init{init_segment}")

    lead_segment = _normalise_range(lead_time_range)
    if lead_segment:
        parts.append(f"lead{lead_segment}")

    if ensemble is None or ensemble == "":
        parts.append("ensnone")
    else:
        token = sanitize_token(ensemble)
        if token.startswith("ens"):
            parts.append(token)
        elif token == "mean":
            parts.append("ensmean")
        elif token == "pooled":
            parts.append("enspooled")
        elif token == "prob":
            parts.append("ensprob")
        else:
            parts.append(f"ens{token}")

    filename = "_".join(filter(None, parts)) + f".{sanitize_token(ext)}"
    return filename


def save_dataset(
    payload: xr.Dataset | xr.DataArray,
    directory: Path | str,
    filename: str,
) -> Path:
    """Persist an ``xarray`` payload to *directory/filename*."""
    dir_path = ensure_dir(directory)
    path = dir_path / filename
    if isinstance(payload, xr.DataArray):
        name = payload.name or "values"
        payload = payload.to_dataset(name=name)
    payload.to_netcdf(path)
    return path


def save_npz(
    payload: Mapping[str, Any],
    directory: Path | str,
    filename: str,
) -> Path:
    """Persist a mapping of numpy-compatible arrays to an ``.npz`` file."""
    dir_path = ensure_dir(directory)
    path = dir_path / filename
    np.savez(path, **payload)
    return path


def save_dataframe(
    frame: pd.DataFrame,
    directory: Path | str,
    filename: str,
    *,
    index: bool = False,
    **to_csv_kwargs: Any,
) -> Path:
    """Persist *frame* as CSV (default) or Parquet if ``filename`` ends with ``.parquet``."""
    dir_path = ensure_dir(directory)
    path = dir_path / filename
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=index, **to_csv_kwargs)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=index, **to_csv_kwargs)
    else:
        raise ValueError(f"Unsupported dataframe suffix: {suffix}")
    return path
