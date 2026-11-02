# containers/

One container image per model for the GH200 (aarch64) nodes of Clariden. The models need mutually
incompatible stacks (JAX vs PyTorch builds, natten, makani, anemoi), so each gets its own image.

| File | Purpose |
|---|---|
| `<model>.dockerfile` | `aurora`, `graphcast`, `sfno`, `fcn3`, `atlas`, `aifsens` (the `aifs` runs use `aifsens.sqsh`) |
| `locks/<model>.txt` | full package list of the built image actually used for the paper runs (pip-freeze style, git installs with their commit) |
| `freeze_sqsh.py` | regenerates `locks/` by reading the dist-info metadata out of `$AIENS_STORE/<model>.sqsh` with `unsquashfs` (no container runtime; runs on a login node, ~20 s) |
| `build.sh` | podman build (linux/arm64) + `enroot import` -> `.sqsh` |
| `submit_build.sh` | runs `build.sh` on a compute node via `srun` (builds are not allowed on login nodes) |
| `Dockerfile` | legacy all-in-one image (`ai-ens.sqsh`), only used by `tools/submit_model_tests.sh` |

```bash
bash containers/submit_build.sh aurora            # one model, debug partition, -> $AIENS_STORE/aurora.sqsh
bash containers/submit_build.sh all normal        # all six in parallel
python containers/freeze_sqsh.py                  # refresh locks/ after a rebuild
bash tools/run_model_tests.sh <model|all>         # short-rollout smoke test inside the image
```

Image facts (from `locks/`, 2026-11-02): Ubuntu 24.04, Python 3.12, CUDA 13.1, NGC PyTorch 25.12 base (atlas:
26.01, needed for natten 0.21.5), uv 0.9.26. earth2studio commits: aurora `c82844e`; graphcast, sfno, fcn3,
atlas `cbcb0ec`; aifsens `9c93a55`. Per-model stacks and checkpoint sources:
[docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md).

Pinning: the Dockerfiles originally installed `earth2studio@main` and unpinned git dependencies; on 2026-11-02
they were pinned to the commits found inside the images. Transitive PyPI dependencies still resolve at build
time; for a bit-identical environment add `-c containers/locks/<model>.txt` to the `uv pip install` lines or
simply use the existing images in `$AIENS_STORE`. A rebuild from the pinned Dockerfiles has not been tested.

Known build quirks, all handled in the Dockerfiles:

- PyPI `graphcast` is an unrelated package: uninstall it and install DeepMind's from git, plus `chex`.
- `warp-lang < 1.13` for physicsnemo 2.0 (`wp.context.Device` was removed).
- GraphCast: `xarray < 2026.4` (`xr.Dataset(dataset)` became a TypeError).
- torch-harmonics, makani and natten must be built `--no-build-isolation` against the NGC torch, with
  `FORCE_CUDA_EXTENSION=1` / `NATTEN_CUDA_ARCH=9.0`.
- NGC marks its Python as externally managed: `--system --break-system-packages`.

At run time the repo is bind-mounted over the installed package
(`$AIENS_REPO/ai_models_ensembles:/usr/local/lib/python3.12/dist-packages/ai_models_ensembles`), so code
changes do not need a rebuild; call `python -m ai_models_ensembles.cli`, not the baked-in `ai-ens`.
