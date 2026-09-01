"""Resolved paths for the analysis, table, and figure scripts.

The scripts were developed against a fixed directory layout on the CSCS
Alps box. Every machine-specific location is centralised here and can be
overridden with an environment variable, so the figure/table regeneration
runs on any machine that has the cached intermediates. The defaults
reproduce the original hardcoded paths.

  AIENS_STORE    persistent data root (forecasts, baselines, intercomparison)
  AIENS_SCRATCH  fast scratch for large intermediates (CSV / JSON caches)
  AIENS_WB2_22   WeatherBench2 ERA5 truth zarr, 2022-2023
  AIENS_WB2_24   WeatherBench2 ERA5 truth zarr, 2024-2025
  AIENS_IFS_ENS  IFS-ENS reference zarr

REPO, FIGURES, and DATA are derived from this file's location and never
need an override.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "figures"
DATA = REPO / "tools" / "data"

STORE = Path(
    os.environ.get("AIENS_STORE", "/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles")
)
SCRATCH = Path(os.environ.get("AIENS_SCRATCH", "/iopsstor/scratch/cscs/sadamov"))

BASELINES = STORE / "baselines"
INTERCOMP = BASELINES / "intercomparison"

WB2_2022 = os.environ.get(
    "AIENS_WB2_22",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2022_2023.zarr",
)
WB2_2024 = os.environ.get(
    "AIENS_WB2_24",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_2024_2025.zarr",
)
WB2_ORIGINAL = os.environ.get(
    "AIENS_WB2_ORIGINAL",
    "/capstor/store/cscs/swissai/weatherbench/weatherbench2_original",
)
IFS_ENS = os.environ.get("AIENS_IFS_ENS", "/capstor/store/cscs/swissai/a122/IFS/ifs_ens.zarr")
