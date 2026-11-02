# 2026-11-02 Handover digest: technical facts that otherwise live only in the local-only paper notes

At handover the memory was split: technical notes are public in this repo, paper-process notes
(review rounds, writing, submission mechanics) stay local-only (`.claude/*/paper/`, gitignored; full copy
on the ab016 store, see `docs/HANDOVER.md`). These technical facts were found only in the local half,
so they are restated here.

- IFS-ENS (WB2 archive, `ifs_ens.zarr`) NaN gaps are whole (init, lead) map slices, level-invariant
  within a variable, with a per-variable pattern. Per-variable NaN fraction (24-init sample, all leads):
  2t 18.4%, msl 29.3%, z 17.8%, t 19.7%, u 18.9%, v 18.9%, q 19.7%. Not "surface only". This is why
  IFS-ENS is excluded from ES/VS/SIGK/W1 (see also `2026-10-07_ifs_ens_spatial_mean_ssr_fairness.md`).
- IFS-ENS common-sample CRPSS check (`tools/crpss_common_sample.py`, inputs
  `$STORE/analysis/per_init_crps_production_v2.csv` + `tools/data/crps_clim_eval_1990_2019_per_init.json`):
  the 8-way ranking is identical on the full 112 inits and on the IFS-ENS-valid common sample at 120 h
  (n=70, max delta +0.0020) and 240 h (n=83, max delta -0.0022, aurora_encoder).
- SFNO spatial-mean over-dispersion containment is a property of the coarse-mode (Phase 3) restriction,
  not of SFNO itself. 4-init ablation grid, 7-variable mean, 240 h spatial-mean SSR: Phase 1 all weights
  3.02, Phase 2b encoder 1.78, Phase 2b decoder 2.44, Phase 2 processor 0.56, Phase 3 coarse-mode
  (sigma 0.25) 0.83; coarse-mode never exceeds 1 across sigma 0.05-1.9. The 112-init sfno_modes10 anchor
  is 0.66. Processor-only (the SHT spectral filters) at low sigma also stays below 1, so the general lever
  is "perturb a small spectral/low-dimensional subset at small magnitude".
- `tools/spatial_mean_ssr.py` reopens each zarr per variable/level/lead: run it as a CPU sbatch
  (~12 min on a compute node), it dies on a login node.
- SFNO refresh production grid (`baselines/sfno_p6c`): `tools/submit_sfno_p6c_production.sh`
  (env `SFNO_FRESH=1 SIGMA=0.35 MODE_CUT=10`), one sbatch per init (SIGSEGV avoidance) plus a `/dev/shm`
  sweep.
- 1990-2019 climatology (`tools/compute_climatology_1990_2019.py`): WB2 ARCO-ERA5 is read over anonymous
  HTTPS because `gcsfs` was broken in the host venv (missing `_cffi_backend.so`); the full 30-year x
  7-variable run takes 13-16 h instead of ~6 h. Output cached in `tools/data/crps_clim_eval_*.json` and
  `tools/data/channel_scale_1990_2019.json`, so it never needs rerunning for the paper numbers.
- Open code-review items (never fixed): `evaluate_ablation.sh` has no guard for the "fewer than 4 inits ->
  stale eval config" trap; `e2s_data.py` `MergedDataSource` silently falls back to CDS on `KeyError`
  (Phase 5 IC), which could collapse a member's IC spread unnoticed (add a log + assert the fallback set
  is boundary-only).
- Inference cost: Atlas needs ~38 min/member on GH200 (paper: 63 min on A100), structural to its
  stochastic-interpolant sampler, not a Clariden artefact. The paper's cost claim is "zero marginal
  training cost" (SPW reuses the deterministic checkpoint), not cheaper inference per member.
- LaTeX: tectonic only writes `main.bbl` to disk with `--keep-intermediates`. An arXiv tarball that ships
  `main.bbl` without `paper_refs.bib` + `ametsocV6.bst` compiles with exit 0 and an EMPTY bibliography.
  Always assert `\bibitem` count == cited-key count, never trust the exit status.
- `pkill -f <pattern>` run over ssh matches the remote shell's own command line and kills it; use
  `pgrep -f` + `kill <pid>` instead.
