---
date: 2026-06-13
summary: "Pre-submission deletion candidates (obsolete one-shot scripts, one-off milton bib/dev scripts, stale doc/redundant test, ephemeral logs). NOT deleted yet -- sweep them right before code submission."
---

# Pre-submission cleanup candidates (defer deletion until code submission)

UPDATE 2026-06-15: the "defer" decision below is SUPERSEDED. The project is now in submission-prep; the
cleanup sweep is an ACTIVE default task (see `.claude/reference/presubmission_checklist.md` section A).
The drop list below is still the authoritative list of what to delete.

Decision 2026-06-13 (superseded): do NOT delete these now (keep them available while the paper
is in flight); clean them up in one pass right before the code repo is submitted /
made public. History retains everything after deletion, so reproducibility is safe.

## Ephemeral on-disk only (gitignored, no git/remote impact)
- `logs/` -- ~69 MB, 204 dask/run logs. `rm -rf logs/*` anytime.
- regenerable caches (already cleaned this session): `.pytest_cache`, `.ruff_cache`,
  `ai_models_ensembles.egg-info`, repo `__pycache__/`.

## Obsolete one-shot scripts (tracked; job done, 0 references)
- `tools/rescale_ssr_fortinfix.py` -- one-shot Fortin-factor SSR remediation, applied 2026-06-13.
- `tools/diff_climatology_recompute.py` -- one-off old-vs-new CRPS-denominator compare; Gaussian->WB2 migration complete.
- `scripts/relaunch_phase1_sfno_graphcast.sh` + `scripts/supervise_phase1_relaunch.sh` -- one-time relaunch after the 2026-05-30 complex-skip/hyperparam-perturb bugfix.

## One-off Milton bib/dev scripts (tracked; "already executed", moved in 2026-06-13)
- `tools/milton/lookup_walsh_refs.py`, `add_milton_refs.py`, `zotero_add_walsh_refs.py` -- one-shot Crossref/Zotero/bib appenders.
- `tools/milton/test_single_track.py` -- single-init tracker dev-verification scaffold.

## Borderline (stale/redundant, harmless to keep)
- `tools/QUICKSTART_TEST.md` -- walkthrough doc, no external refs, last touched 2026-05-15; may have drifted from README.
- `tools/test_basic_functionality.py` -- overlaps the pytest `tests/test_repository_smoke.py`.

## Keep (referenced/active -- NOT candidates)
`tools/test_model.py` (run_model_tests.sh / submit_model_tests.sh), `tools/debug_e2e.py`,
`tools/dump_*_keys.py` + `submit_dump_*`, `tools/zotero_*` audit tools, `tools/reshard_cosmo_zarr.py`.

## Bigger lever (separate, not done): .git is ~7.1 GB
Untracking figures (commit 8f5fff2) shrinks the working tree, not history. Real repo-size
reduction needs `git filter-repo` + force-push on MeteoSwiss/ai-models-ensembles -- riskier,
do only on explicit request, ideally coordinated (shared repo).
