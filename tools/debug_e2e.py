#!/usr/bin/env python3
"""End-to-end debug script for ai-models-ensembles.

Tests every layer of the pipeline: imports, CLI, format conversion, data layer,
perturbation, model registry, GPU, and (optionally) a real 1-step inference.

Designed to run inside the container on a GPU compute node via:
    bash tools/submit_debug.sh

Can also run standalone:
    python tools/debug_e2e.py              # full suite
    python tools/debug_e2e.py --skip-gpu   # skip GPU tests
    python tools/debug_e2e.py --with-infer # include real 1-step inference
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

PASS = 0
FAIL = 0
SKIP = 0
RESULTS: list[tuple[str, str, str]] = []


def _record(name: str, status: str, detail: str = "") -> None:
    global PASS, FAIL, SKIP
    RESULTS.append((name, status, detail))
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    else:
        SKIP += 1
    tag = f"  {status:5s} {name}"
    if detail:
        tag += f"  -- {detail}"
    print(tag, flush=True)


def run_check(name: str, fn: Callable[[], str | None], skip_if: bool = False) -> None:
    if skip_if:
        _record(name, "SKIP", "skipped by flag")
        return
    try:
        detail = fn() or ""
        _record(name, "PASS", detail)
    except Exception as exc:
        _record(name, "FAIL", f"{exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        print(flush=True)


# -- 1. Environment ----------------------------------------------------------
def check_python_version():
    v = sys.version_info
    assert v >= (3, 11), f"Python >= 3.11 required, got {v.major}.{v.minor}.{v.micro}"
    return f"{v.major}.{v.minor}.{v.micro}"


# -- 2. Core imports ----------------------------------------------------------
def check_import_numpy():
    import numpy as np

    return f"numpy {np.__version__}"


def check_import_xarray():
    import xarray

    return f"xarray {xarray.__version__}"


def check_import_zarr():
    import zarr

    return f"zarr {zarr.__version__}"


def check_import_typer():
    import typer

    return f"typer {typer.__version__}"


def check_import_torch():
    import torch

    return f"torch {torch.__version__}"


def check_import_earth2studio():
    import earth2studio

    return f"earth2studio {getattr(earth2studio, '__version__', 'unknown')}"


def check_import_swissclim():
    import swissclim_evaluations

    return f"swissclim_evaluations {getattr(swissclim_evaluations, '__version__', 'unknown')}"


def check_warp_version():
    import warp

    ver = getattr(warp, "__version__", "unknown")
    warp.init()
    if not hasattr(warp, "context"):
        raise RuntimeError(f"warp {ver} missing 'context' submodule (need warp-lang<1.13)")
    return f"warp {ver}"


# -- 3. Package module imports ------------------------------------------------
def check_import_cli():
    from ai_models_ensembles.cli import app

    assert app is not None
    return "cli.app loaded"


def check_import_e2s_models():
    from ai_models_ensembles.e2s_models import REGISTRY

    assert len(REGISTRY) >= 5, f"Expected >= 5 models, got {len(REGISTRY)}"
    return f"{len(REGISTRY)} models registered"


def check_import_e2s_data():
    from ai_models_ensembles import e2s_data

    assert hasattr(e2s_data, "build_data_source")
    return "OK"


def check_import_e2s_inference():
    from ai_models_ensembles import e2s_inference

    assert hasattr(e2s_inference, "run_inference")
    return "OK"


def check_import_e2s_perturbation():
    from ai_models_ensembles import e2s_perturbation

    assert hasattr(e2s_perturbation, "perturb_initial_conditions")
    return "OK"


def check_import_swissclim_format():
    from ai_models_ensembles.swissclim_format import E2S_TO_SWISSCLIM

    assert len(E2S_TO_SWISSCLIM) > 10
    return f"{len(E2S_TO_SWISSCLIM)} variable mappings"


# -- 4. CLI smoke (typer.testing) ---------------------------------------------
def check_cli_help():
    from typer.testing import CliRunner
    from ai_models_ensembles.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"exit {result.exit_code}: {result.stdout}"
    for cmd in ("infer", "verify", "intercompare", "models"):
        assert cmd in result.stdout, f"'{cmd}' missing from --help"
    return "all subcommands present"


def check_cli_models():
    from typer.testing import CliRunner
    from ai_models_ensembles.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, f"exit {result.exit_code}: {result.stdout}"
    for name in ("graphcast_operational", "sfno", "aurora", "fcn3", "atlas"):
        assert name in result.stdout, f"'{name}' missing"
    return "5 models listed"


# -- 5. SwissClim format conversion ------------------------------------------
def check_e2s_to_swissclim_synthetic():
    import numpy as np
    import xarray as xr
    from ai_models_ensembles.swissclim_format import e2s_to_swissclim

    lat = np.linspace(90, -90, 8, dtype="float32")
    lon = np.linspace(0, 359, 16, dtype="float32")
    times = np.array(["2023-01-02T00:00"], dtype="datetime64[ns]")
    lead = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    variables = ["u10m", "v10m", "t2m", "t500", "t850", "z500"]
    rng = np.random.default_rng(42)
    data = rng.standard_normal((1, 3, 6, 8, 16)).astype("float32")
    ds = xr.Dataset(
        {"output": (("time", "lead_time", "variable", "lat", "lon"), data)},
        coords={"time": times, "lead_time": lead, "variable": variables, "lat": lat, "lon": lon},
    )
    result = e2s_to_swissclim(ds, ensemble_id=0)

    for dim in ("init_time", "lead_time", "ensemble", "latitude", "longitude"):
        assert dim in result.dims, f"Missing dim: {dim}"
    assert np.issubdtype(result["lead_time"].dtype, np.timedelta64)
    assert "2m_temperature" in result.data_vars
    assert "temperature" in result.data_vars
    assert "level" in result["temperature"].dims
    return f"vars={sorted(result.data_vars)}"


def check_swissclim_zarr_roundtrip():
    import numpy as np
    import xarray as xr
    from ai_models_ensembles.swissclim_format import e2s_to_swissclim

    rng = np.random.default_rng(99)
    ds = xr.Dataset(
        {
            "output": (
                ("time", "lead_time", "variable", "lat", "lon"),
                rng.standard_normal((1, 2, 2, 4, 8)).astype("float32"),
            )
        },
        coords={
            "time": np.array(["2023-01-02T00:00"], dtype="datetime64[ns]"),
            "lead_time": np.array([0, 6], dtype="timedelta64[h]").astype("timedelta64[ns]"),
            "variable": ["t2m", "u10m"],
            "lat": np.linspace(90, -90, 4, dtype="float32"),
            "lon": np.linspace(0, 359, 8, dtype="float32"),
        },
    )
    sc = e2s_to_swissclim(ds, ensemble_id=0)

    with tempfile.TemporaryDirectory(prefix="debug_e2e_") as tmp:
        zp = Path(tmp) / "test.zarr"
        sc.to_zarr(zp, mode="w", consolidated=True)
        loaded = xr.open_zarr(zp, consolidated=True)
        assert set(sc.data_vars) == set(loaded.data_vars)
        for var in sc.data_vars:
            np.testing.assert_allclose(sc[var].values, loaded[var].values, atol=1e-6)
    return "write + read OK"


def check_multi_member_zarr_append():
    import numpy as np
    import xarray as xr
    from ai_models_ensembles.swissclim_format import e2s_to_swissclim

    rng = np.random.default_rng(11)
    n_members = 3
    with tempfile.TemporaryDirectory(prefix="debug_e2e_ens_") as tmp:
        zp = Path(tmp) / "ensemble.zarr"
        for m in range(n_members):
            ds = xr.Dataset(
                {
                    "output": (
                        ("time", "lead_time", "variable", "lat", "lon"),
                        rng.standard_normal((1, 2, 2, 4, 8)).astype("float32"),
                    )
                },
                coords={
                    "time": np.array(["2023-01-02T00:00"], dtype="datetime64[ns]"),
                    "lead_time": np.array([0, 6], dtype="timedelta64[h]").astype("timedelta64[ns]"),
                    "variable": ["t2m", "u10m"],
                    "lat": np.linspace(90, -90, 4, dtype="float32"),
                    "lon": np.linspace(0, 359, 8, dtype="float32"),
                },
            )
            sc = e2s_to_swissclim(ds, ensemble_id=m)
            if m == 0:
                sc.to_zarr(zp, mode="w", consolidated=True)
            else:
                sc.to_zarr(zp, mode="a", append_dim="ensemble", consolidated=True)
        loaded = xr.open_zarr(zp, consolidated=True)
        assert loaded.sizes["ensemble"] == n_members
    return f"{n_members} members OK"


def check_filter_levels():
    import numpy as np
    import xarray as xr
    from ai_models_ensembles.e2s_inference import _filter_levels

    ds = xr.Dataset(
        {
            "temperature": (
                ("init_time", "lead_time", "ensemble", "level", "latitude", "longitude"),
                np.zeros((1, 2, 1, 5, 4, 8)),
            ),
            "2m_temperature": (
                ("init_time", "lead_time", "ensemble", "latitude", "longitude"),
                np.zeros((1, 2, 1, 4, 8)),
            ),
        },
        coords={
            "init_time": np.array(["2023-01-02T00:00"], dtype="datetime64[ns]"),
            "lead_time": np.array([0, 6], dtype="timedelta64[h]").astype("timedelta64[ns]"),
            "ensemble": [0],
            "level": [100, 200, 500, 700, 850],
            "latitude": np.linspace(90, -90, 4),
            "longitude": np.linspace(0, 359, 8),
        },
    )
    filtered = _filter_levels(ds, [500, 850])
    assert list(filtered["level"].values) == [500, 850]
    assert "2m_temperature" in filtered.data_vars
    unfiltered = _filter_levels(ds, None)
    assert list(unfiltered["level"].values) == [100, 200, 500, 700, 850]
    return "OK"


# -- 6. Data layer ------------------------------------------------------------
def check_ic_perturbation():
    import numpy as np
    import xarray as xr
    from ai_models_ensembles.e2s_perturbation import perturb_initial_conditions

    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"t2m": (("time", "lat", "lon"), rng.standard_normal((1, 4, 8)).astype("float32"))},
        coords={
            "time": np.array(["2023-01-02T00:00"], dtype="datetime64[ns]"),
            "lat": np.linspace(90, -90, 4),
            "lon": np.linspace(0, 359, 8),
        },
    )
    perturbed = perturb_initial_conditions(ds, magnitude=0.01, seed=42)
    diff = np.abs(perturbed["t2m"].values - ds["t2m"].values).max()
    assert diff > 0, "No perturbation applied"
    unchanged = perturb_initial_conditions(ds, magnitude=0.0, seed=42)
    np.testing.assert_array_equal(unchanged["t2m"].values, ds["t2m"].values)
    return f"max_diff={diff:.6f}"


def check_xarray_datasource():
    import numpy as np
    import xarray as xr
    from ai_models_ensembles.e2s_data import XarrayDataSource

    ds = xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), np.ones((1, 4, 8), dtype="float32")),
            "u10m": (("time", "lat", "lon"), np.full((1, 4, 8), 2.0, dtype="float32")),
        },
        coords={
            "time": np.array(["2023-01-02T00:00"], dtype="datetime64[ns]"),
            "lat": np.linspace(90, -90, 4),
            "lon": np.linspace(0, 359, 8),
        },
    )
    src = XarrayDataSource(ds)
    result = src(datetime(2023, 1, 2), ["t2m", "u10m"])
    assert "variable" in result.dims or result.sizes.get("variable", 0) == 2
    assert result.sizes["lat"] == 4
    assert result.sizes["lon"] == 8
    return f"shape={result.shape}"


def check_data_source_factory_file():
    import numpy as np
    import xarray as xr
    from ai_models_ensembles.e2s_data import build_data_source

    with tempfile.TemporaryDirectory(prefix="debug_e2e_ds_") as tmp:
        zp = Path(tmp) / "ic.zarr"
        ds = xr.Dataset(
            {"t2m": (("time", "lat", "lon"), np.ones((1, 4, 8), dtype="float32"))},
            coords={
                "time": np.array(["2023-01-02T00:00"], dtype="datetime64[ns]"),
                "lat": np.linspace(90, -90, 4),
                "lon": np.linspace(0, 359, 8),
            },
        )
        ds.to_zarr(zp, mode="w")
        src = build_data_source(f"file:{zp}")
        result = src(datetime(2023, 1, 2), "t2m")
        assert result is not None
    return "file: source OK"


def check_data_source_arco_class():
    from earth2studio import data as e2s_data

    src = e2s_data.ARCO()
    assert src is not None
    return "ARCO instantiated (no fetch)"


# -- 7. Model registry -------------------------------------------------------
def check_model_registry():
    from ai_models_ensembles.e2s_models import REGISTRY, get_spec, steps_from_hours

    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert ":" in spec.e2s_class
    spec = get_spec("graphcast_operational")
    assert steps_from_hours(spec, 336) == 56
    return f"{len(REGISTRY)} models OK"


def check_model_class_import():
    from ai_models_ensembles.e2s_models import REGISTRY, import_class

    results = []
    failures = []
    for name, spec in REGISTRY.items():
        try:
            import_class(spec)
            results.append(f"{name}=OK")
        except Exception as e:
            results.append(f"{name}=FAIL({e})")
            failures.append(name)
    summary = ", ".join(results)
    if failures:
        raise RuntimeError(f"Failed to import: {failures}. {summary}")
    return summary


# -- 8. Config ---------------------------------------------------------------
def check_config_files():
    required = [
        "scripts/config.sh",
        "config/swissclim_eval.yaml.template",
        "config/swissclim_ifs_ens.yaml.template",
        "pyproject.toml",
    ]
    missing = [f for f in required if not (REPO_ROOT / f).exists()]
    assert not missing, f"Missing: {missing}"
    return f"{len(required)} files present"


def check_yaml_template():
    content = (REPO_ROOT / "config" / "swissclim_eval.yaml.template").read_text()
    for ph in ["${PERTURBATION_DIR}", "${TARGET_PATH}", "${REGION_DIR}", "${MODEL_NAME}"]:
        assert ph in content, f"Missing placeholder {ph}"
    envsubst = "available" if shutil.which("envsubst") else "missing"
    return f"placeholders OK, envsubst {envsubst}"


# -- 9. GPU -------------------------------------------------------------------
def check_gpu_torch():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.cuda.get_device_name(0)
    n = torch.cuda.device_count()
    x = torch.randn(256, 256, device="cuda")
    y = torch.mm(x, x)
    assert y.shape == (256, 256)
    return f"{n} GPU(s): {dev}"


def check_gpu_jax():
    import jax

    return f"backend={jax.default_backend()}, devices={len(jax.devices())}"


# -- 10. Real inference (opt-in, needs GPU + network) -------------------------
def check_real_inference():
    """Single-member, 1-step graphcast_operational inference via ARCO."""
    from ai_models_ensembles.e2s_inference import run_inference

    with tempfile.TemporaryDirectory(prefix="debug_e2e_infer_") as tmp:
        out = Path(tmp) / "forecast.zarr"
        run_inference(
            model_name="graphcast_operational",
            init_time=datetime(2023, 1, 2),
            lead_hours=6,
            output=out,
            n_members=1,
            data_source="arco",
            seed=0,
            output_levels=[500, 850],
        )
        import xarray as xr

        ds = xr.open_zarr(out, consolidated=True)
        assert "init_time" in ds.dims
        assert "lead_time" in ds.dims
        assert "ensemble" in ds.dims
        nvars = len(ds.data_vars)
    return f"{nvars} variables, dims={list(ds.dims)}"


# -- Main ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="E2E debug for ai-models-ensembles")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU tests")
    parser.add_argument(
        "--with-infer", action="store_true", help="Run real 1-step inference (needs GPU + network)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ai-models-ensembles -- end-to-end debug")
    print(f"  python: {sys.executable} ({sys.version.split()[0]})")
    print(f"  cwd:    {Path.cwd()}")
    print(f"  time:   {datetime.now().isoformat()}")
    print("=" * 70, flush=True)

    sections = [
        (
            "[Environment]",
            [
                ("python_version", check_python_version, False),
            ],
        ),
        (
            "[Core imports]",
            [
                ("numpy", check_import_numpy, False),
                ("xarray", check_import_xarray, False),
                ("zarr", check_import_zarr, False),
                ("typer", check_import_typer, False),
                ("torch", check_import_torch, False),
                ("earth2studio", check_import_earth2studio, False),
                ("swissclim_evaluations", check_import_swissclim, False),
                ("warp", check_warp_version, False),
            ],
        ),
        (
            "[Package modules]",
            [
                ("cli", check_import_cli, False),
                ("e2s_models", check_import_e2s_models, False),
                ("e2s_data", check_import_e2s_data, False),
                ("e2s_inference", check_import_e2s_inference, False),
                ("e2s_perturbation", check_import_e2s_perturbation, False),
                ("swissclim_format", check_import_swissclim_format, False),
            ],
        ),
        (
            "[CLI smoke]",
            [
                ("cli_help", check_cli_help, False),
                ("cli_models", check_cli_models, False),
            ],
        ),
        (
            "[Format conversion]",
            [
                ("e2s_to_swissclim", check_e2s_to_swissclim_synthetic, False),
                ("zarr_roundtrip", check_swissclim_zarr_roundtrip, False),
                ("multi_member_append", check_multi_member_zarr_append, False),
                ("filter_levels", check_filter_levels, False),
            ],
        ),
        (
            "[Data layer]",
            [
                ("ic_perturbation", check_ic_perturbation, False),
                ("xarray_datasource", check_xarray_datasource, False),
                ("datasource_file", check_data_source_factory_file, False),
                ("datasource_arco_class", check_data_source_arco_class, False),
            ],
        ),
        (
            "[Model registry]",
            [
                ("model_registry", check_model_registry, False),
                ("model_class_import", check_model_class_import, False),
            ],
        ),
        (
            "[Config]",
            [
                ("config_files", check_config_files, False),
                ("yaml_template", check_yaml_template, False),
            ],
        ),
        (
            "[GPU]",
            [
                ("gpu_torch", check_gpu_torch, args.skip_gpu),
                ("gpu_jax", check_gpu_jax, args.skip_gpu),
            ],
        ),
        (
            "[Inference (real)]",
            [
                ("real_1step_infer", check_real_inference, not args.with_infer),
            ],
        ),
    ]

    for header, checks in sections:
        print(f"\n{header}")
        for name, fn, skip in checks:
            run_check(name, fn, skip_if=skip)

    print("\n" + "=" * 70)
    print(f"  PASS: {PASS}   FAIL: {FAIL}   SKIP: {SKIP}   TOTAL: {len(RESULTS)}")
    print("=" * 70)

    if FAIL > 0:
        print("\nFailed:")
        for name, status, detail in RESULTS:
            if status == "FAIL":
                print(f"  {name}: {detail}")

    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
