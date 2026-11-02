# Environment and dependencies

Three separate software environments, each pinned to what actually produced the paper results:

| Environment | Used for | Pinned in |
|---|---|---|
| Host analysis venv (Python 3.11, no GPU stack) | every `tools/*.py` figure, table and analysis job, SwissClim evaluation, the CLI's `verify` / `intercompare` | [env/host-requirements.txt](../env/host-requirements.txt) |
| Six per-model containers (NGC PyTorch, Python 3.12, CUDA 13.1, aarch64) | inference (`ai-ens infer`), anything importing `earth2studio`, `torch` or `jax` | `containers/<model>.dockerfile` + [containers/locks/](../containers/locks/) |
| Conda CLI tools | TempestExtremes (Milton tracking), ffmpeg, tectonic | [env/conda-tools.yml](../env/conda-tools.yml) |

`pyproject.toml` only states abstract requirements (ranges) and the per-model extras; it is not a lock. The old
`uv.lock` was removed on 2026-11-02: it no longer matched `pyproject.toml` and resolved `earth2studio==0.5.0`
from PyPI, while every run used earth2studio 0.15.0a0 from git.

## Host analysis venv

- On Clariden: `/capstor/store/cscs/mch/s83/sadamov/venvs/ai-models-ensembles` (the repo's `.venv` symlink;
  readable by `s83`). Python 3.11.9.
- Fresh install anywhere (Linux or macOS, x86_64 or aarch64):

  ```bash
  uv venv --python 3.11 .venv && source .venv/bin/activate
  uv pip install -r env/host-requirements.txt      # 229 exact pins, incl. SwissClim at a commit
  uv pip install -e . --no-deps                     # this package (CLI + helpers), no GPU deps
  ```

  The lock resolves cleanly (`uv pip install --dry-run`, checked 2026-11-02).
- Key versions: numpy 2.4.6, xarray 2026.4.0, zarr 3.1.6, dask 2026.3.0, pandas 2.3.3, scipy 1.17.1,
  matplotlib 3.10.9, cartopy 0.25.0, huracanpy 1.4.0, typer 0.26.7, weatherbenchX at `3e9cf1f`.
- SwissClim Evaluations: `git+https://github.com/swiss-ai/SwissClim_Evaluations.git@9d3a7fde10361c7136b78d97b91b0e08085fc54d`
  (branch `research`, `v0.2.0-72-g9d3a7fd`, includes the 2026-09-17 member-0 spectra fix). On Clariden it is
  an editable checkout in `SwissClim_Evaluations/` (gitignored); `pyproject.toml` pins the same commit.
- Regenerate the lock after changing the venv: `uv pip freeze --python .venv/bin/python`, then replace the
  editable lines by the git pins (see the header of the lock file).
- Never `uv pip install` a new package into the shared venv without adding it to `pyproject.toml` first (it
  once broke `certifi` for everybody).

## Containers

One image per model, because the models' dependency stacks conflict (JAX vs PyTorch builds, natten, makani,
anemoi). Images on Clariden: `$STORE/<model>.sqsh` (17-22 GB each). The code is bind-mounted at run time, so
the image only provides the environment.

| Image | Base | earth2studio | Model stack |
|---|---|---|---|
| `aurora` | `nvcr.io/nvidia/pytorch:25.12-py3` | `c82844e` | microsoft_aurora 1.8.0, torch_harmonics `a632ca7` |
| `graphcast` | `nvcr.io/nvidia/pytorch:25.12-py3` | `cbcb0ec` | graphcast `08cf736`, jax/jaxlib 0.10.0 (CUDA 13 plugin), xarray pinned < 2026.4 (2026.2.0) |
| `sfno` | `nvcr.io/nvidia/pytorch:25.12-py3` | `cbcb0ec` | makani `b38fcb2`, torch_harmonics `8f08a22`, physicsnemo 2.0.0, warp-lang < 1.13 |
| `fcn3` | `nvcr.io/nvidia/pytorch:25.12-py3` | `cbcb0ec` | as sfno |
| `atlas` | `nvcr.io/nvidia/pytorch:26.01-py3` | `cbcb0ec` | natten 0.21.5 (built for SM 9.0), physicsnemo `369a157`, torch_harmonics `a632ca7` |
| `aifsens` (also used for `aifs`) | `nvcr.io/nvidia/pytorch:25.12-py3` | `9c93a55` | anemoi-models 0.5.1, earthkit-regrid 0.4.0, torch_harmonics `a632ca7` |

Common to all: Ubuntu 24.04, Python 3.12, CUDA 13.1, torch 2.10.0a0 NGC build, numpy 2.1.0, zarr 3.2.1,
xarray 2026.4.0 (graphcast: 2026.2.0), eccodes 2.47.0, uv 0.9.26.

- The Dockerfiles used to install `earth2studio@main`, so the four builds picked up three different
  commits. On 2026-11-02 every git dependency in the Dockerfiles was pinned to the commit found inside the
  image, and `containers/locks/<model>.txt` records the full package list of each image (read straight from
  the squashfs with `python containers/freeze_sqsh.py`, no container runtime needed).
- PyPI dependencies that the Dockerfiles do not name are still resolved at build time. For an exact rebuild,
  add the lock as constraints (`uv pip install -c containers/locks/<model>.txt ...`). Rebuilding from the
  pinned Dockerfiles has not been tested; the existing `.sqsh` images are the reference environment.
- Build: `bash containers/submit_build.sh <model|all> [partition]` (compute node; podman + enroot).
  Smoke test: `bash tools/run_model_tests.sh <model|all>`. Details: [containers/README.md](../containers/README.md).

## Model checkpoints

Downloaded by earth2studio on first use into `EARTH2STUDIO_CACHE` (bind-mounted from
`$AIENS_SCRATCH/e2s_cache`, or `$STORE/e2s_cache_backup` for CDS runs). Sources at the pinned earth2studio
commits:

| Model | Package | Notes |
|---|---|---|
| Aurora | `hf://microsoft/aurora`, file `aurora-0.25-pretrained.ckpt` | our `e2s_models._patch_aurora_package` replaces earth2studio's default (`refs/pr/1`) so a perturbed local copy can be loaded |
| GraphCast operational | `gs://dm_graphcast/graphcast` (operational 0.25 deg, 13 levels, HRES fine-tuned) | anonymous GCS |
| SFNO | `ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0` | copy in `$STORE/earth2studio_cache/sfno` |
| FourCastNet 3 | `hf://nvidia/fourcastnet3@76ef0c60237e458b33196ba027134e27f3fc4538` | |
| Atlas | `hf://nvidia/atlas-era5@fdce0480c5e6f03d409089bf285f4bcc1d84519e` | |
| AIFS single | `hf://ecmwf/aifs-single-1.1` | copy in `$STORE/e2s_cache_backup/aifs-single-1.1` |
| AIFS-ENS | `hf://ecmwf/aifs-ens-1.0` | copy in `$STORE/e2s_cache_backup/aifs-ens-1.0` |

The scratch copies of the other checkpoints were purged; they are re-downloaded automatically. Weight layouts
were audited against these exact files (`.claude/reference/checkpoint_perturbation_audit.md`).

## Conda tools

On Clariden they live in the shared miniforge base `/capstor/store/cscs/mch/s83/sadamov/miniforge3`
(`source .../etc/profile.d/conda.sh && conda activate base`), which also provides `gh`. Elsewhere:
`conda env create -f env/conda-tools.yml` and `export AIENS_MILTON_CONDA=$CONDA_PREFIX`. Pinned:
tempest-extremes 2.4.2, udunits2 2.2.28, libnetcdf 4.10.0, hdf5 2.1.0, ffmpeg 8.1.1, tectonic 0.16.9.
`texcount` (word counts) is the CTAN perl script copied into that base.

## Services and credentials

| Service | Needed for | Credential |
|---|---|---|
| CDS (Copernicus Climate Data Store) | AIFS / AIFS-ENS initial conditions | `~/.cdsapirc` (and `~/.ecmwfapirc`), mounted into the container at `~` and `/root`; personal, not in the repo |
| ARCO-ERA5 (Google Cloud public bucket) | ICs of all other models | none |
| WeatherBench 2 bucket over HTTPS | 1990-2019 climatology jobs | none (`gcsfs` is broken in the venv; plain HTTPS is used) |
| Hugging Face, NGC, GCS | checkpoints | none (public) |
| IBTrACS | Milton best track (`huracanpy`) | none |

On Clariden compute nodes HTTPS needs
`export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')` (no system CA bundle).

## Hardware

Produced on CSCS Alps/Clariden: NVIDIA GH200 (Grace-Hopper, aarch64, SM 9.0), 4 GPUs per node, Slurm with
pyxis/enroot. The host venv and the analysis scripts run on any Linux/macOS machine; the containers are
aarch64 builds and need GH200-class nodes or a rebuild for x86_64.
