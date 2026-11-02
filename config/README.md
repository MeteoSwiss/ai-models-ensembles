# config/

Example SwissClim Evaluations configurations. The Slurm evaluation drivers do not read these files: they
generate the YAML for every job inline (`generate_main_config` and friends in
`scripts/evaluate_baselines.sh`, `scripts/evaluate_ablation.sh`) and save it next to the outputs
(`$STORE/baselines/<experiment>/eval_*_config.yaml`, `$STORE/ablation/<phase>/<model>/eval/<run>_*config.yaml`).
Those saved files are the authoritative record of how each evaluation ran.

| File | Content |
|---|---|
| `swissclim_eval.yaml.template` | single-forecast evaluation; `envsubst` placeholders `${PERTURBATION_DIR}`, `${TARGET_PATH}`, ... |
| `swissclim_ifs_ens.yaml.template` | the same for the IFS-ENS reference zarr |
| `swissclim_intercompare.yaml` | example cross-model intercomparison |

```bash
envsubst < config/swissclim_eval.yaml.template > my_eval.yaml
python -m ai_models_ensembles.cli verify --config my_eval.yaml
```

Schema reference: the example config in the SwissClim Evaluations repository (branch `research`, commit
pinned in [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md)).
