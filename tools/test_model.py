#!/usr/bin/env python3
"""Test a single model's 1-step inference.

Usage:
    python tools/test_model.py graphcast_operational
    python tools/test_model.py sfno
    python tools/test_model.py aurora
    python tools/test_model.py fcn3
    python tools/test_model.py atlas
"""

import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_model(model_name: str) -> dict:
    """Run 1-step inference for a single model, return results dict."""
    result = {
        "model": model_name,
        "status": "PASS",
        "error": None,
        "traceback": None,
    }

    try:
        from ai_models_ensembles.e2s_inference import run_inference

        with tempfile.TemporaryDirectory(prefix=f"test_{model_name}_") as tmp:
            out = Path(tmp) / "forecast.zarr"
            # AIFS-ENS needs variables (tcw, stl1, ...) not in ARCO ERA5
            source = "cds" if model_name == "aifsens" else "arco"
            init = datetime(2023, 1, 2)
            run_inference(
                model_name=model_name,
                init_time=init,
                lead_hours=6,
                output=out,
                n_members=1,
                data_source=source,
                seed=0,
                output_levels=[500, 850],
            )
            import xarray as xr

            ds = xr.open_zarr(out, consolidated=True)
            nvars = len(ds.data_vars)
            result["detail"] = f"{nvars} vars, dims={list(ds.dims)}"

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{e.__class__.__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/test_model.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    result = test_model(model_name)

    print(f"\n{'=' * 70}")
    print(f"Model: {result['model']}")
    print(f"Status: {result['status']}")
    if result.get("detail"):
        print(f"Detail: {result['detail']}")
    if result["error"]:
        print(f"Error: {result['error']}")
        if result["traceback"]:
            print(f"\nTraceback:\n{result['traceback']}")
    print(f"{'=' * 70}\n")

    sys.exit(0 if result["status"] == "PASS" else 1)
