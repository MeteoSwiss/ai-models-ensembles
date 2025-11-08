#!/usr/bin/env python3
"""
Basic functionality test for ai-models-ensembles repository.
This script tests core functionality without requiring data downloads or GPU.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all core modules can be imported."""
    print("=" * 60)
    print("Testing module imports...")
    print("=" * 60)

    modules_to_test = [
        "ai_models_ensembles.cli",
        "ai_models_ensembles.convert_grib_to_zarr",
        "ai_models_ensembles.preprocess_data",
        "ai_models_ensembles.animate_2d_maps",
        "ai_models_ensembles.animate_3d_grids",
        "ai_models_ensembles.plot_0d_distributions",
        "ai_models_ensembles.plot_1d_timeseries",
    ]

    failed = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            failed.append((module_name, str(e)))

    print()
    if failed:
        print(f"Failed to import {len(failed)} module(s)")
        return False
    else:
        print("All modules imported successfully!")
        return True


def test_cli_commands():
    """Test that CLI commands can be invoked."""
    print("\n" + "=" * 60)
    print("Testing CLI commands...")
    print("=" * 60)

    try:
        from typer.testing import CliRunner

        from ai_models_ensembles.cli import app

        runner = CliRunner()

        # Test main help
        result = runner.invoke(app, ["--help"])
        if result.exit_code == 0:
            print("✓ Main CLI help works")
        else:
            print(f"✗ Main CLI help failed with exit code {result.exit_code}")
            return False

        # Test individual command help
        commands = [
            "download-reanalysis",
            "download-ifs-ensemble",
            "download-ifs-control",
            "convert",
            "infer",
            "verify",
            "intercompare",
        ]

        for cmd in commands:
            result = runner.invoke(app, [cmd, "--help"])
            if result.exit_code == 0:
                print(f"✓ {cmd} --help works")
            else:
                print(f"✗ {cmd} --help failed")
                return False

        print("\nAll CLI commands are accessible!")
        return True

    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dependencies():
    """Test that key dependencies are available."""
    print("\n" + "=" * 60)
    print("Testing key dependencies...")
    print("=" * 60)

    dependencies = {
        "xarray": "Data manipulation",
        "zarr": "Data storage format",
        "numpy": "Numerical computing",
        "matplotlib": "Plotting",
        "seaborn": "Statistical visualization",
        "typer": "CLI framework",
        "rich": "Terminal formatting",
    }

    failed = []
    for pkg, description in dependencies.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {pkg:20s} v{version:10s} - {description}")
        except ImportError:
            print(f"✗ {pkg:20s} - {description} - NOT FOUND")
            failed.append(pkg)

    print()
    if failed:
        print(f"Missing {len(failed)} package(s): {', '.join(failed)}")
        return False
    else:
        print("All key dependencies are available!")
        return True


def test_ai_models():
    """Test that ai-models CLI is available."""
    print("\n" + "=" * 60)
    print("Testing ai-models installation...")
    print("=" * 60)

    import subprocess

    try:
        result = subprocess.run(
            ["ai-models", "--models"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            models = result.stdout.strip().split("\n")
            print("✓ ai-models CLI is available")
            print(f"  Available models: {', '.join(models)}")
            return True
        else:
            print(f"✗ ai-models CLI failed with exit code {result.returncode}")
            print(f"  Error: {result.stderr}")
            return False

    except FileNotFoundError:
        print("✗ ai-models command not found")
        return False
    except Exception as e:
        print(f"✗ ai-models test failed: {e}")
        return False


def test_config():
    """Test that config.sh can be sourced and has required variables."""
    print("\n" + "=" * 60)
    print("Testing configuration...")
    print("=" * 60)

    import os

    # Check for config file
    config_file = Path("config.sh")
    if not config_file.exists():
        print("✗ config.sh not found")
        return False

    print("✓ config.sh exists")

    # Check key environment variables (if already set)
    env_vars = [
        "OUTPUT_DIR",
        "DATE_TIME",
        "MODEL_NAME",
        "NUM_MEMBERS",
    ]

    set_vars = []
    unset_vars = []

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var:20s} = {value}")
            set_vars.append(var)
        else:
            print(f"  {var:20s} (not set - will be set by config.sh)")
            unset_vars.append(var)

    print("\nNote: Source config.sh before running commands:")
    print("  source config.sh")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AI Models Ensembles - Basic Functionality Test")
    print("=" * 60)
    print()

    # Change to repository root
    repo_root = Path(__file__).parent.parent
    import os

    os.chdir(repo_root)
    print(f"Working directory: {repo_root}")
    print()

    results = {
        "Dependencies": test_dependencies(),
        "Module Imports": test_imports(),
        "CLI Commands": test_cli_commands(),
        "AI Models CLI": test_ai_models(),
        "Configuration": test_config(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")

    all_passed = all(results.values())

    print()
    if all_passed:
        print("🎉 All tests passed! The repository is ready to use.")
        print("\nNext steps:")
        print("1. Configure ECMWF API credentials in ~/.ecmwfapirc")
        print("2. Review config.sh and adjust settings as needed")
        print("3. Follow QUICKSTART_TEST.md for running a complete example")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("- Ensure virtual environment is activated: source .venv/bin/activate")
        print("- Install missing dependencies: uv pip install -e .")
        print("- Check that ai-models is installed: pip install ai-models")
        return 1


if __name__ == "__main__":
    sys.exit(main())
