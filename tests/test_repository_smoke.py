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
    result = runner.invoke(
        cli_module.app, ["verify", "--config", str(tmp_path / "nope.yaml")]
    )
    assert result.exit_code != 0
