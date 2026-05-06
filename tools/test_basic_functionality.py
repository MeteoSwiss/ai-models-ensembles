#!/usr/bin/env python3
"""Smoke test for the ai-models-ensembles repository (no GPU, no network).

Checks:
- Core modules import (cli + e2s_* layers + swissclim_format).
- earth2studio and swissclim_evaluations are importable.
- CLI subcommands `--help` succeeds for each registered command.
- scripts/config.sh exists.
"""

import sys
from pathlib import Path


def test_imports() -> bool:
    print("=" * 60)
    print("Testing module imports...")
    print("=" * 60)

    modules = [
        "ai_models_ensembles.cli",
        "ai_models_ensembles.e2s_models",
        "ai_models_ensembles.e2s_data",
        "ai_models_ensembles.e2s_perturbation",
        "ai_models_ensembles.e2s_inference",
        "ai_models_ensembles.swissclim_format",
        "earth2studio",
        "swissclim_evaluations",
    ]
    failed = []
    for name in modules:
        try:
            __import__(name)
            print(f"OK  {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failed.append(name)
    return not failed


def test_cli_commands() -> bool:
    print("\n" + "=" * 60)
    print("Testing CLI commands...")
    print("=" * 60)

    try:
        from typer.testing import CliRunner

        from ai_models_ensembles.cli import app

        runner = CliRunner()

        result = runner.invoke(app, ["--help"])
        if result.exit_code != 0:
            print(f"FAIL --help (exit {result.exit_code})")
            return False
        print("OK  ai-ens --help")

        for cmd in ("models", "infer", "verify", "intercompare"):
            r = runner.invoke(app, [cmd, "--help"])
            if r.exit_code != 0:
                print(f"FAIL {cmd} --help (exit {r.exit_code})")
                return False
            print(f"OK  ai-ens {cmd} --help")
        return True
    except Exception as e:
        print(f"FAIL CLI test: {e}")
        return False


def test_dependencies() -> bool:
    print("\n" + "=" * 60)
    print("Testing key dependencies...")
    print("=" * 60)

    deps = {
        "xarray": "Data manipulation",
        "zarr": "Zarr v3 storage",
        "numpy": "Numerical computing (>= 2.0)",
        "typer": "CLI framework",
        "rich": "Terminal formatting",
        "earth2studio": "Inference runtime",
        "swissclim_evaluations": "Verification toolkit",
    }
    failed = []
    for pkg, desc in deps.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"OK  {pkg:25s} v{ver:10s}  {desc}")
        except ImportError:
            print(f"FAIL {pkg:25s} - {desc}")
            failed.append(pkg)
    return not failed


def test_config_file() -> bool:
    print("\n" + "=" * 60)
    print("Testing configuration...")
    print("=" * 60)
    p = Path("scripts/config.sh")
    if not p.exists():
        print(f"FAIL {p} not found")
        return False
    print(f"OK  {p}")
    return True


def main() -> int:
    print("\nai-models-ensembles - basic functionality smoke test\n")
    repo_root = Path(__file__).parent.parent
    import os

    os.chdir(repo_root)

    results = {
        "Dependencies": test_dependencies(),
        "Module imports": test_imports(),
        "CLI commands": test_cli_commands(),
        "Configuration": test_config_file(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL':6s} {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
