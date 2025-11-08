"""High-level smoke tests for the repository.

Goals:
1. Import core package and CLI without errors.
2. Exercise verify path with synthetic minimal datasets (no external deps).
3. Confirm artefact directory structure is created (png_*/artifacts_*).
4. Ensure vertical profile and PIT functions callable on synthetic data.

These tests avoid coupling to internal implementation details while
asserting end-to-end viability of critical entry points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

import ai_models_ensembles.cli as cli_module
import ai_models_ensembles.plot_0d_distributions as p0
import ai_models_ensembles.plot_1d_timeseries as p1


def _synthetic_ds(has_members: bool = True, has_levels: bool = True):
    step = np.arange(0, 3)
    lat = np.linspace(-20, 20, 11)
    lon = np.linspace(0, 30, 16)
    member = np.arange(0, 4) if has_members else None
    level = np.array([1000, 850]) if has_levels else None
    base_shape = [s.size for s in [step, lat, lon]]
    rng = np.random.default_rng(0)
    data_var = rng.standard_normal(
        tuple(
            ([member.size] if member is not None else [])
            + ([level.size] if level is not None else [])
            + base_shape
        )
    )
    dims = (
        ([] if member is None else ["member"])
        + ([] if level is None else ["isobaricInhPa"])
        + ["step", "latitude", "longitude"]
    )
    coords = {"step": step, "latitude": lat, "longitude": lon}
    if member is not None:
        coords["member"] = member
    if level is not None:
        coords["isobaricInhPa"] = level
    ds_fc = xr.Dataset({"temperature": (dims, data_var)}, coords=coords)
    # Ground truth without member dim
    gt_data = rng.standard_normal(
        tuple(([] if level is None else [level.size]) + base_shape)
    )
    gt_dims = ([] if level is None else ["isobaricInhPa"]) + [
        "step",
        "latitude",
        "longitude",
    ]
    ds_gt = xr.Dataset(
        {"temperature": (gt_dims, gt_data)},
        coords={k: v for k, v in coords.items() if k != "member"},
    )
    return ds_fc, ds_gt


def test_imports():
    assert hasattr(cli_module, "app"), "Typer app missing"
    assert "plot_vertical_profile_metrics" in p1.__all__


def test_vertical_profile_and_pit_functions(tmp_path: Path):
    fc, gt = _synthetic_ds(has_members=True, has_levels=True)
    # Vertical profiles
    p1.plot_vertical_profile_metrics(
        forecast=fc,
        ground_truth=gt,
        variable="temperature",
        path_out=str(tmp_path / "figs"),
        output_mode="npz",
        artifact_root=str(tmp_path / "artifacts"),
        ensemble="test",
        lead_subset=[0, 2],
    )
    vp_dir = tmp_path / "artifacts" / "vertical_profiles"
    assert vp_dir.exists(), "vertical_profiles artifact directory not created"
    assert any(
        "combined_data" in f.name for f in vp_dir.glob("*.npz")
    ), "Missing combined vertical profile NPZ"

    # PIT histogram global (ensure member dimension present)
    p0.plot_pit_histogram(
        variable="temperature",
        forecast=fc,
        ground_truth=gt,
        path_out=str(tmp_path / "figs_pit"),
        model_name="test",
        output_mode="npz",
        artifact_root=str(tmp_path / "artifacts_pit"),
        ensemble="test",
    )
    pit_dir = tmp_path / "artifacts_pit" / "pit"
    assert pit_dir.exists(), "pit artifact directory not created"
    assert any(
        "pit_hist_temperature" in f.name for f in pit_dir.glob("*.npz")
    ), "Missing PIT histogram NPZ"


def test_basic_artifact_structure(tmp_path: Path):
    fc, gt = _synthetic_ds(has_members=True, has_levels=True)
    # Use vertical profile as representative
    p1.plot_vertical_profile_metrics(
        forecast=fc,
        ground_truth=gt,
        variable="temperature",
        path_out=str(tmp_path / "figs_vp"),
        output_mode="both",
        artifact_root=str(tmp_path / "artifacts_vp"),
        ensemble="test",
    )
    figs = list((tmp_path / "figs_vp").glob("*.png"))
    assert figs, "PNG figure not created"
    npz_files = list((tmp_path / "artifacts_vp" / "vertical_profiles").glob("*.npz"))
    assert npz_files, "NPZ artifacts not created"
