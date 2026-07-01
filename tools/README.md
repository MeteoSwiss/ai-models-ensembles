# Development Tools

Inspection, testing, and zarr-maintenance utilities for
`ai-models-ensembles`. Environment setup lives in the main
[README](../README.md) (host venv on iopsstor, plus per-model containers
under [containers/](../containers/)).

## Host venv (one-time)

```bash
VENV_DIR=/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles  # persistent capstor; scratch purges
uv venv --python 3.11 "$VENV_DIR"
ln -s "$VENV_DIR" ../.venv
source ../.venv/bin/activate

uv pip install \
  "typer>=0.12" "rich>=13.0" \
  "zarr>=3.0" "xarray>=2024.10" dask numcodecs \
  "numpy>=2.0" pandas matplotlib \
  "pyyaml>=6.0" netCDF4 cfgrib \
  "pytest>=8.0" "ruff>=0.6" "pre-commit>=3.7" ipython
uv pip install -e .. --no-deps
```

This venv is for editor support, slurm orchestration, and reading zarr
outputs. `earth2studio`, `torch`, `jax`, and model deps are deliberately
absent - they live in the per-model `.sqsh` containers.

## check_gpu.py

Verify CUDA / GPU visibility from inside a container.

```bash
srun --container-image=$STORE/aurora.sqsh python tools/check_gpu.py
```

## inspect_weights.py

Walks a model checkpoint and reports tensor shapes / dtypes. Used to derive
the layer-group indices recorded in `_MODEL_LAYER_GROUPS`
([ai_models_ensembles/e2s_perturbation.py](../ai_models_ensembles/e2s_perturbation.py)).

## reshard_zarr.py + submit_reshard.sh

Rewrite an existing zarr store to the SwissClim-compatible chunking /
sharding layout. See the `_INNER_CHUNKS` constants in
[e2s_inference.py](../ai_models_ensembles/e2s_inference.py) and
[reshard_zarr.py](reshard_zarr.py) for the target shape.

```bash
bash tools/submit_reshard.sh <path/to/forecast.zarr>
```

## run_model_tests.sh / submit_model_tests.sh / test_model.py

Per-model smoke tests that exercise a short rollout inside a container.

```bash
bash tools/run_model_tests.sh <model|all>
```

## debug_e2e.py

End-to-end debug script: short rollout + SwissClim format conversion. Use
when chasing a regression in the inference or schema-bridge code.

## See also

- [Main README](../README.md)
- [scripts/README.md](../scripts/README.md) - slurm submission scripts
- [containers/](../containers/) - per-model Dockerfiles and build helpers
