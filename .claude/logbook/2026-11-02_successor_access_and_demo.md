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
