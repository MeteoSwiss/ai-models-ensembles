# AI Model Ensembles for Weather Forecasting

Code, pinned environments and documentation for the paper "Stochastically Perturbed Weights: Ensembles
from Deterministic Machine-Learning Weather Models" (Adamov, Fuhrer, Knutti, Schemm; submitted to AMS
Artificial Intelligence for the Earth Systems, preprint on arXiv physics.ao-ph).

Stochastically Perturbed Weights (SPW) turn a frozen deterministic ML weather model into an ensemble: each
member gets multiplicative Gaussian noise on (a subset of) the checkpoint weights, `w' = w (1 + sigma xi)`,
with no retraining. The paper applies it to Aurora, GraphCast, SFNO and AIFS and compares the resulting
ensembles with trained-probabilistic models (AIFS-ENS, Atlas, FourCastNet 3) and IFS-ENS on 112
initialisations, plus a Hurricane Milton case study.

The repository is a thin layer: [earth2studio](https://github.com/NVIDIA/earth2studio) loads the models,
fetches initial conditions and rolls out; [SwissClim Evaluations](https://github.com/swiss-ai/SwissClim_Evaluations)
verifies; this code adds the perturbations, the model registry, the Slurm workflow for CSCS Alps, one
container per model, and the analysis behind every figure and table.

Release `v1.0.0` is the code that produced the submitted results.

## Documentation

| Page | Read it for |
|---|---|
| [docs/HANDOVER.md](docs/HANDOVER.md) | project status, who owns what, where everything is, open items |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | method, ablation phases, production picks, metrics, settled decisions |
| [docs/DATA.md](docs/DATA.md) | every dataset: location, access, format, naming, provenance ([inventory](docs/data/STORE_INVENTORY.md)) |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) | how each figure and table is made, which ones rebuild on any machine |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | dependencies and their pinned versions, checkpoints, credentials |
| [docs/HPC.md](docs/HPC.md) | running on CSCS Alps/Clariden, environment variables, failure modes |
| [.claude/README.md](.claude/README.md) | the project notes kept for Claude Code (logbook of findings and decisions) |

## Layout

| Path | What |
|---|---|
| [ai_models_ensembles/](ai_models_ensembles/README.md) | the package: CLI, model registry, weight/IC perturbation, inference driver, SwissClim schema |
| [scripts/](scripts/README.md) | Slurm submitters for inference and verification |
| [tools/](tools/README.md) | analysis jobs and one script per figure/table; [tools/milton/](tools/milton/README.md) case study; [tools/data/](tools/data/README.md) vendored inputs |
| [figures/](figures/README.md) | generated figures (PDF) and LaTeX tables |
| [containers/](containers/README.md) | one Dockerfile per model, build scripts, package locks of the built images |
| [env/](env/README.md) | pinned host venv and conda tools |
| [config/](config/README.md) | example SwissClim configurations |
| `notebooks/` | end-to-end examples: a generic pip-environment quickstart and a Clariden version |
| [docs/](docs/) | documentation above |
| `tests/` | schema-bridge and CLI tests (`pytest -q`) |

## Quick start

Two notebooks run a small ensemble end to end (GraphCast with the production perturbation, 2 members,
24 h) and score it with SwissClim:

- [notebooks/spw_quickstart.ipynb](notebooks/spw_quickstart.ipynb): any Linux machine with one NVIDIA
  GPU, a pip environment and the public data on Google Cloud.
- [notebooks/spw_clariden.ipynb](notebooks/spw_clariden.ipynb): CSCS Clariden, the production containers
  and local data only, compared with the production run of the same init.

Regenerate a paper figure on any machine from the vendored data:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r env/host-requirements.txt && uv pip install -e . --no-deps
export AIENS_STORE=$PWD/tools/data/store AIENS_SCRATCH=$PWD/tools/data/store/analysis
python tools/plot_headline_crpss_vs_lead_8way.py         # -> figures/headline_crpss_vs_lead_8way.pdf
```

Run inference and verification on Clariden (GH200, Slurm + pyxis/enroot):

```bash
bash containers/submit_build.sh <model|all>                          # or reuse $STORE/<model>.sqsh
bash scripts/submit_all_inference.sh aurora_encoder                  # production grid
bash scripts/evaluate_baselines.sh all aurora_encoder                # SwissClim, one job per module
python -m ai_models_ensembles.cli infer --help                       # inside a container
```

Forecast output follows the SwissClim schema (dims `ensemble, [level,] init_time, lead_time, latitude,
longitude`, ECMWF long variable names, integer hPa levels); see
[swissclim_format.py](ai_models_ensembles/swissclim_format.py) and [docs/DATA.md](docs/DATA.md#3-forecast-format).

## Development

```bash
pre-commit install && pre-commit run --all-files
pytest -q
```

Requirements: Linux or macOS for the analysis; for inference Linux aarch64 with NVIDIA GH200, CUDA 13.1
images, Slurm with pyxis/enroot, outbound HTTPS (ARCO-ERA5, Hugging Face, NGC) and a CDS key for AIFS.

License: BSD 3-Clause ([LICENSE](LICENSE)).
