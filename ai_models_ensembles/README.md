# ai_models_ensembles (the package)

Thin layer between earth2studio (model loading, IC data, rollout) and SwissClim Evaluations (verification).
Inference code runs inside the per-model containers; `swissclim_format.py`, `cli.py verify/intercompare` and
the tests also run in the host venv.

| Module | Role |
|---|---|
| [cli.py](cli.py) | Typer CLI `ai-ens` / `python -m ai_models_ensembles.cli`: `models` (registry), `infer` (one init, M members -> zarr), `verify` (SwissClim on a YAML config), `intercompare` |
| [e2s_models.py](e2s_models.py) | model registry: our names (`aurora`, `graphcast_operational`, `sfno`, `fcn3`, `atlas`, `aifs`, `aifsens`) -> earth2studio class, native step, default data source; patches Aurora's package so a perturbed local checkpoint can be loaded |
| [e2s_data.py](e2s_data.py) | earth2studio data-source adapters: ARCO, CDS, WB2, IFS / IFS-ENS zarr, `file:` paths, per-member IFS-ENS perturbed-analysis ICs (`ifs_ens_member_ic_source`, `--ic-zarr`) |
| [e2s_perturbation.py](e2s_perturbation.py) | IC noise and checkpoint (weight) perturbation for every backend (torch state dicts, GraphCast NPZ, AIFS anemoi checkpoints); `_MODEL_LAYER_GROUPS` (named groups for `--layer`); SFNO coarse-mode masks; caches the default checkpoints once per job |
| [e2s_inference.py](e2s_inference.py) | driver: per-member perturbed packages, multi-GPU member scheduling (spawned workers, per-round retries, `/dev/shm` cleanup), rollout, conversion and sharded zarr write (`_INNER_CHUNKS`) |
| [swissclim_format.py](swissclim_format.py) | conversion of earth2studio output to the SwissClim schema (dims, long variable names, `level` in hPa, `lead_time` timedelta) |
| [graphcast_coarse_perturbation.py](graphcast_coarse_perturbation.py) | Phase 3 GraphCast: noise on latent activations of the n coarsest mesh nodes (`--graph-coarse-sigma`, `--graph-coarse-nodes`, `GC_FROZEN`) |
| `{aurora,aifs,sfno,graphcast}_fresh_perturbation.py` | Phase 6: weight noise redrawn during the rollout, switched on by `<MODEL>_FRESH=1`, `<MODEL>_FRESH_SIGMA`, `<MODEL>_FRESH_REFRESH_EVERY` (SFNO also `_MODE_CUT`, `_MODE_SKIP`); seeds mixed with `_mix_seed` (splitmix64) |

## `ai-ens infer`

```bash
python -m ai_models_ensembles.cli infer --model aurora --init 2023-01-02T00:00 --lead-hours 240 \
    --members 10 --weight-magnitude 0.025 --layer encoder --data-source arco \
    --output /path/forecast.zarr
```

| Option | Meaning |
|---|---|
| `--model` | registry name (`ai-ens models` lists them) |
| `--init`, `--lead-hours`, `--members` | ISO init time, rollout length (default 240), ensemble size (default 1) |
| `--weight-magnitude`, `--layer` | weight-noise sigma (0 = off) and the group to perturb: `all`, a named group, an index, a range `10:50` or fractions `0.0:0.33` |
| `--coarse-mode-cut`, `--coarse-mode-skip-first` | SFNO: perturb only spectral modes l < cut, optionally skipping the first k (l = 0 is the global mean) |
| `--graph-coarse-sigma`, `--graph-coarse-nodes` | GraphCast Phase 3 activation noise |
| `--ic-magnitude` | multiplicative IC noise (0 = off) |
| `--ic-zarr` | per-member ICs from a SwissClim-format IFS-ENS analysis zarr (Phase 5) |
| `--data-source` | `arco`, `cds`, `gfs`, `ifs`, `ifs_ens`, `wb2`, `file:PATH` |
| `--output-levels`, `--output-vars` | what to keep in the zarr (defaults: levels `500,850`, `all` variables the model emits; the production submitters pass the 9 paper variables) |
| `--seed` | base seed (default 42); member m uses `seed + m` for weights, `seed + m + 10000` for ICs |

Environment: `E2S_ROUND_RETRIES` (retries of crashed members), `E2S_TO_SWISSCLIM`, the `*_FRESH*` variables
above, `GC_FROZEN`. GPUs are taken from `CUDA_VISIBLE_DEVICES`; GraphCast runs members sequentially on one GPU.

Adding a model: one `ModelSpec` in `e2s_models.py`, its layer groups in `_MODEL_LAYER_GROUPS`, a checkpoint
handler in `e2s_perturbation.py` if the format is new, and a container (`containers/`).

Tests: `pytest -q` from the repo root (`tests/`: schema bridge, CLI loadability, IC-zarr source; earth2studio
tests skip on the host). 13 passed, 1 skipped on 2026-11-02.
