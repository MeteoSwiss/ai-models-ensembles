"""High-level smoke tests for the repository.

Inference logic now lives in `earth2studio` and verification in
`swissclim-evaluations`; the tests here only cover that the CLI loads, the
expected commands exist, and external packages are importable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import ai_models_ensembles.cli as cli_module


def test_imports():
    assert hasattr(cli_module, "app"), "Typer app missing"


def test_swissclim_available():
    pytest.importorskip("swissclim_evaluations")


def test_earth2studio_available():
    pytest.importorskip("earth2studio")


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("infer", "verify", "intercompare", "models"):
        assert cmd in result.stdout
    assert "download-reanalysis" not in result.stdout
    assert "convert" not in result.stdout


def test_models_command_lists_registry():
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["models"])
    assert result.exit_code == 0
    assert "graphcast_operational" in result.stdout
    assert "sfno" in result.stdout


def test_verify_rejects_missing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SWISSCLIM_CONFIG", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["verify", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0


def test_infer_help_has_ic_zarr():
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["infer", "--help"])
    assert result.exit_code == 0
    assert "--ic-zarr" in result.stdout


def test_ifs_ens_member_ic_source(tmp_path: Path):
    """Phase 5: per-member IC source selects member k and serves e2s tokens."""
    import numpy as np
    import xarray as xr

    from ai_models_ensembles.e2s_data import ifs_ens_member_ic_source

    ens, levels, lat, lon = 3, [500, 850], 4, 4
    init = np.array(["2024-02-15T00:00", "2024-02-14T18:00"], dtype="datetime64[ns]")
    # Same base field for every member + a distinct per-member offset, so
    # member k - member 0 == k exactly (proves per-member selection).
    base = (
        np.random.default_rng(0)
        .standard_normal((len(init), len(levels), lat, lon))
        .astype("float32")
    )
    offset = np.arange(ens, dtype="float32").reshape(1, ens, 1, 1, 1)
    field = base[:, None, :, :, :] + offset
    dims3d = ("init_time", "ensemble", "level", "latitude", "longitude")
    ds = xr.Dataset(
        {
            "geopotential": (dims3d, field),
            "temperature": (dims3d, field + 10),
        },
        coords={
            "init_time": init,
            "ensemble": np.arange(ens),
            "level": levels,
            "latitude": np.linspace(-1, 1, lat),
            "longitude": np.linspace(0, 3, lon),
        },
    )
    zpath = tmp_path / "ic.zarr"
    ds.to_zarr(zpath)

    src0, cached = ifs_ens_member_ic_source(str(zpath), 0)
    src1, _ = ifs_ens_member_ic_source(str(zpath), 1, cached_ds=cached)

    # z500 token resolves and members differ by the per-member offset.
    a = src0(init[0], "z500").values
    b = src1(init[0], "z500").values
    assert np.isfinite(a).all()
    assert np.allclose(b - a, 1.0)

    # Multi-time serve (the T / T-6h mechanism Aurora & GraphCast rely on).
    m = src0(init, "t850")
    assert m.sizes["time"] == 2
