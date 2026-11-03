Successor access audit (user `olifu`), shared-venv home dependency found, per-user script defaults, and an end-to-end demo verified under a fresh venv.

## Access state of a new user (checked 2026-11-02 for `olifu`)
- In `s83`: reaches `$STORE` (s83 ACL), the container images and the conda base. Read/write user ACL on the
  whole store granted (job 3834319, xfer).
- Not in `ab016`: no IFS reference (`ab016/IFS`), no COSMO/neural-LAM data, no Slurm allocation.
  `/capstor/store/cscs/swissai/ab016` is `root:ab016` mode 770, so user ACLs below it do not help; the
  `ab016` PI has to add the user in the CSCS account portal.
- Not in `weatherbench`: `/capstor/store/cscs/swissai/weatherbench` (ERA5 truth of every evaluation) is
  `root:weatherbench` with ACLs for `a122`, `uba03`, `g34`, NOT `ab016`. The author reads it through `a122`.
  Every new user needs `weatherbench` (or CSCS adds `ab016` to that ACL).

## Shared host venv works only for the author
`venvs/ai-models-ensembles` has its base interpreter in `/users/sadamov/.local/share/uv/python/` and
editable installs of this package and SwissClim pointing into `/users/sadamov/pyprojects/...`. The home is
mode 700 and is deleted with the account. Everyone else builds a venv from `env/host-requirements.txt`
(`scripts/handover_demo.sh venv`, 1.5 min on a debug node from an empty uv cache and Python install dir).

## Changes
- Scratch defaults `/iopsstor/scratch/cscs/sadamov` -> `$USER` in all submit scripts and `tools/_env.py`;
  `#SBATCH --output` paths use `%u`.
- Script comments pointing at "MEMORY.md" / `[[calibration-winners]]` now name the `.claude/` files.
- `scripts/handover_demo.sh` (venv / infer / eval / compare). Reference run in
  `/capstor/store/cscs/mch/s83/sadamov/handover-demo/run/` (clean clone, fresh venv): inference 3834344
  (8:20, debug), SwissClim deterministic + probabilistic 3834409 (7:04). Demo vs production
  `graphcast_all/20230102_0000`, +24 h: 2m temperature RMS diff 0.037 K (member spread 0.29 K), MSL 3.6 Pa
  (spread 42 Pa), i.e. GPU nondeterminism only.
- Private notes and the pre-handover agent memory copied to `/capstor/store/cscs/mch/s83/sadamov/handover-private/`
  (owner + `olifu` only; `group::---` had to be set explicitly, the parent's default ACL gives s83 rwx).
- Cross-paper overview (status, code versions, checkpoints, data, reproduction per paper) merged into the
  MeteoSwiss Confluence page of the PhD; its source is the local-only `scripts/confluence.md` (in
  `.git/info/exclude`). A first version in the thesis repo and on the store was removed again the same day.
- Only this repo has a shared `.claude/`. The local-only Claude notes of neural-lam-dev (CLAUDE.md, memory),
  building_neural_lam_models_rev2 and SwissClim_Evaluations (auto-memory) were copied to
  `agent-memory-other-repos/` in `handover-private/` and in `/capstor/store/cscs/swissai/ab016/sadamov/`.
  The thesis notes stay private (personal, archived in `sadamov/clariden-archive`).
- Demo switched to Clariden-only inputs: `DATA_SOURCE=swissclim:<WB2 ERA5 zarr>` (new spec in
  `build_data_source`) and `E2S_CACHE=$STORE/e2s_cache_backup`, where the GraphCast package (gs://dm_graphcast,
  6 files) was copied. Found on the way: `XarrayDataSource` returned (variable, time, lat, lon) and broke the
  earth2studio IO write ("got 2 and 83"); only `MergedDataSource` transposed, so it never surfaced in Phase 5.
  Fixed in `XarrayDataSource.__call__`. Rerun 3835133 (3:42, no new files in the cache, no downloads), eval
  3835163 (7:29): demo vs production 0.037 K at +24 h, same as with ARCO ICs, so the WB2 and ARCO ERA5 agree.
  The ARCO-IC reference run of the morning is in `handover-demo/_arco-ic-2026-11-02/`.
- Notebooks: `notebooks/spw_quickstart.ipynb` (pip env, one GPU, public data) and `notebooks/spw_clariden.ipynb`
  (wraps `handover_demo.sh`, new `plot` step). The generic one was executed on a GH200 node outside the
  containers (jobs 3835530, 3837594): GraphCast 2 members 24 h from ARCO, SwissClim vs ARCO ERA5,
  T2m RMSE 0.92 K at +24 h, the same as the Clariden run. Three install traps on the way, all now in
  `env/graphcast-overrides.txt` (`uv pip install -e ".[graphcast]" --override ...`): (1) the unpinned
  `graphcast` git dependency now resolves to DeepMind's renamed `weathernext` package, and earth2studio
  `cbcb0ec` pins its own older graphcast commit, so the container's override (`08cf736`) must be explicit;
  (2) an override `jax==...` drops the `[cuda13]` extra (CPU-only JAX), write `jax[cuda13]==0.10.0`;
  (3) pandas 3 breaks SwissClim's by-lead pivot (`astype(str)` no longer yields "<NA>", all surface
  variables get the same key: "Index contains duplicate entries"), so pandas is pinned to 2.3.3 like the
  container and the host venv. `pyproject.toml` now pins earth2studio to `cbcb0ec` and no longer lists
  graphcast directly (only the Phase 3 module imports it).
- SwissClim against the ARCO zarr over the network took > 2 h per module from Clariden; the notebook
  first copies the 5 scored valid times, 5 variables, 2 levels (150 MB) and evaluates against that.

## 2026-11-03 follow-up (neural-LAM side, all in neural-lam-dev `research`)
- WeatherBench 2 downloads from Clariden: `xr.open_zarr("gs://weatherbench2/...")` without
  `storage_options={"token": "anon"}` makes gcsfs look for Google credentials, which hangs on compute
  nodes (the IFS sample reached 1% in 4 h; one init wrote no chunk in 10 min, CPU idle). Anonymous reads
  run at ~300 MB/s; the IFS sample (133 inits x 41 leads, 65k chunks of full global fields, ~880 GB read,
  15 GB written) then took 26 min (job 3840447). Staged to `ab016/COSMO/cosmo_ifs_sample.zarr`; the Clariden runner and
  notebook now evaluate with IFS boundaries as in Step 9. Full-study IFS (2019-08-30 to 2020-11-02) is
  still not on Clariden; the three missing steps for a full reproduction are now in the paper repo's
  `HANDOVER.md` and on the Confluence page.
- neural-lam eval zarr: `time` (left by `swap_dims`, dim elapsed_forecast_duration only) was overwritten
  by every append along start_time, so it held the last forecast's valid times. Same in the paper's
  `ab016/sadamov/lam_model_forecasts` (README added there). The verification scripts drop it and use
  start_time + lead, so no paper figure or table is affected. Writer fixed; mllam/neural-lam main has no
  writer (open PR #289 drops `time` too).
- GitHub's notebook renderer typesets `$...$` inside one line of a code cell as math
  (`"$SRCDIR/scripts/$f"` was torn apart): keep at most one `$` per line in notebook shell cells.
- Both neural-LAM notebooks dropped again: the guides stay generic and the single source, the Clariden
  specifics are a short `docs/reproduce_paper_sample_clariden.md` next to the runner (the notebook was
  770 KB of JSON for five commands and a figure). SPW notebooks committed with outputs (ad6ffbb).
- Clariden sample with the published checkpoint (training stats, IFS boundaries, jobs 3840519/45/89):
  T_2M drifts cold domain-wide, mean bias -0.24/-1.54/-2.68/-3.68 K at +1/6/12/24 h (RMSE 0.45/1.80/2.97/
  3.98 K), all four start times alike and the same with ERA5 boundaries (+24 h RMSE 3.78 K). Cause: the
  boundary was normalised with ERA5 stats of the sample window (the paper normalises ERA5 and IFS
  boundaries with full-period ERA5 stats, train 2015-11-28..2019-09-30). With those stats (now shipped as
  `scripts/artifacts/cosmo_era5_train_stats.zarr` in neural-lam-dev, 698f368): bias -0.01/-0.14/-0.08/
  -0.05 K, RMSE 0.35/0.64/0.86/0.97 K. The interior stats were already right (cosmo_dataset_reference.md).
  Test job 3840851 (full `cosmo_era5_config.yaml` datastore built from `era_subset.zarr`); runner verified from a
  fresh WORKDIR at 698f368 (prep 3841004, eval 3841058), bit-identical. Clariden walkthrough is now
  `docs/reproduce_paper_sample_clariden.md` + `figures/cosmo_sample_clariden_t2m.png` (c421c37). Scratch
  workdirs `/iopsstor/scratch/cscs/sadamov/neural_lam_sample*` are throwaway.
