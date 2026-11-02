# Handover

State of the project on 2026-11-02, written for whoever continues it (revision of the paper, new
experiments, or reuse of the data). The original author is not reachable from December 2026 and loses
CSCS access then.

## 1. Status

- Paper: "Stochastically Perturbed Weights: Ensembles from Deterministic Machine-Learning Weather Models"
  (Adamov, Fuhrer, Knutti, Schemm). Submitted to AMS Artificial Intelligence for the Earth Systems (AIES);
  the technical-check return was fixed and resubmitted 2026-09-18. No reviewer reports yet at handover.
  Preprint on arXiv (physics.ao-ph, cross-list cs.LG).
- Reviews: a co-author takes over the review and all correspondence and becomes corresponding author with
  the journal; they get access to the Overleaf project and the private paper repo. Journal-process details,
  the weaknesses already known for the revision and the expected reviewer questions are in the private
  handover note (`HANDOVER.md` in the paper repo, copy on Clariden, see section 2).
- When the paper is accepted, please open a GitHub issue in the paper repo with the citation and DOI; the
  original author needs them to cite the final version.
- Code used for the submitted results: tag `v1.0.0` (= `69e3f9f`, named in the paper's App. E). Do not move
  that tag; tag revision code as `v1.1.0` or later.
- After `v1.0.0`, `main` gained only post-submission material: an IFS-ENS line in Fig. 4, a weight-only AIFS
  column in the Milton track figure, talk figures (`figures/talk/`, `figures/slides/`), member-distribution
  diagnostics (`figures/ens_anom/`, 50-member runs), a GraphCast refresh run, and this handover
  (pinned environments, vendored data, docs, agent memory).
- Nothing is running. The last jobs (ACL grant, rank-histogram and fixed-scale ES/VS/SIGK regeneration,
  wind-frame PDFs) were submitted on 2026-11-02; their outputs are in `$STORE/analysis/` and `tools/data/`.

## 2. Where everything is

| What | Where | Access |
|---|---|---|
| Code, docs, vendored figure inputs, technical agent memory | github.com/MeteoSwiss/ai-models-ensembles (`main`) | public |
| Verification package | github.com/swiss-ai/SwissClim_Evaluations, branch `research`, pinned commit `9d3a7fd` | public |
| Paper LaTeX (Overleaf mirror) | github.com/sadamov/ai-models-ensembles (`main`) | private: ask the author or a co-author |
| Forecasts, evaluations, analysis caches, container images | Clariden `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles` (92 TB) | Unix group `s83` |
| IFS-ENS reference and perturbed analyses | Clariden `/capstor/store/cscs/swissai/ab016/IFS/` | group `ab016` |
| Private handover note (journal process, reviewer-question material; also `HANDOVER.md` in the paper repo) + local-only agent memory | Clariden `/capstor/store/cscs/swissai/ab016/sadamov/ai-models-ensembles/` (`HANDOVER.md`, `agent-memory-local-2026-11-02/`) | group `ab016` |
| Copy of that note and memory, plus the pre-handover agent memory (full `MEMORY.md` versions of 2026-06 and 2026-10, the per-fact notes they replaced) | Clariden `/capstor/store/cscs/mch/s83/sadamov/handover-private/` | the author and the co-author taking over (`olifu`) |
| Overview of all of the author's papers and the thesis (status, code versions, data, checkpoints, how to reproduce) | Clariden `/capstor/store/cscs/mch/s83/sadamov/HANDOVER.md`, also `HANDOVER.md` in the private thesis repo `sadamov/thesis` | group `s83` |
| Host venv, conda tools | Clariden `/capstor/store/cscs/mch/s83/sadamov/{venvs/ai-models-ensembles,miniforge3}` | group `s83`; the venv works only for the author ([ENVIRONMENT.md](ENVIRONMENT.md)) |

Everything on Clariden stays owned by the author's (soon deactivated) user. Group members can read it. To
reorganise or delete anything, ask CSCS (or the `s83` project PI) to transfer ownership.

## 3. Getting started

1. Read [EXPERIMENTS.md](EXPERIMENTS.md) (what was done), then [DATA.md](DATA.md) (where it is).
2. On any machine: install the host environment and regenerate a figure from the vendored data
   ([REPRODUCING.md](REPRODUCING.md), section 1). This needs nothing but the repo.
3. On Clariden: get the groups and the Slurm account listed in [HPC.md](HPC.md#accounts-partitions-limits)
   (`s83` for the store, `ab016` for the IFS data and the allocation, `weatherbench` for the ERA5 truth).
   Clone the repo, build your own venv (the shared one works only for the author) and run the small
   end-to-end check, which goes through the production submitter and SwissClim:

   ```bash
   git clone https://github.com/MeteoSwiss/ai-models-ensembles.git && cd ai-models-ensembles
   export AIENS_REPO=$PWD AIENS_PY=$PWD/.venv/bin/python AIENS_ACCOUNT=ab016
   bash scripts/handover_demo.sh venv      # once, ~2 min
   bash scripts/handover_demo.sh infer     # GraphCast graphcast_all, 2023-01-02 00 UTC, 2 members, 24 h (~8 min)
   bash scripts/handover_demo.sh eval      # SwissClim deterministic + probabilistic vs ERA5 (~7 min)
   bash scripts/handover_demo.sh compare   # demo vs production run of the same init
   ```

   Outputs go to `$AIENS_SCRATCH/aiens_demo` (`/iopsstor/scratch/cscs/$USER/aiens_demo`). A reference run
   of the same commands is in `/capstor/store/cscs/mch/s83/sadamov/handover-demo/run/`. Write new results to
   a store location you own (`AIENS_STORE`) or into the existing store (write ACL for `olifu` since
   2026-11-02).
4. For AIFS/AIFS-ENS reruns create your own CDS key (`~/.cdsapirc`).
5. If you use Claude Code: the repo's `CLAUDE.md` and `.claude/` load automatically ([.claude/README.md](../.claude/README.md)).

## 4. Open items

For the revision (details in [EXPERIMENTS.md](EXPERIMENTS.md#known-open-issues-for-the-revision)):

- Related-work sentence on Whitaker et al. is inaccurate; fix it.
- GraphCast "architecture excludes the weight-space refresh" is overstated; `baselines/graphcast_p6c_reseed`
  gives the number (spatial-mean SSR 3.20 -> 2.97 at 240 h).
- Milton F3 and F4 rasters are below 300 dpi; regenerate with `savefig(dpi>=300)` (needs Clariden).
- Replace the Milton F9 in the paper with the corrected one in `figures/`: the submitted F9 draws its IFS-ENS
  reference curve from 2 of 10 members (stale double member filter; text and caption unaffected).
  [REPRODUCING.md](REPRODUCING.md) note 3.
- The paper link in the top-level README still needs the arXiv/DOI URL; Zenodo DOIs for code and data after
  acceptance.

Housekeeping, optional ([data/LARGE_DATASETS.md](data/LARGE_DATASETS.md) lists every dataset of 1 TB or more
with owner and verdict; the copy on Clariden, `$STORE/LARGE_DATASETS.md`, also covers the author's other projects):

- Deleted on 2026-11-02 after verifying the kept copies: the a122 originals of the IFS zarrs, the pre-fix
  `baselines/{aurora,aifs,sfno}_p6c` runs, and the a122/a01 duplicates of the COSMO and neural-LAM data.
- Superseded runs that could still be deleted (~60 GB) once nobody needs them: the four `*_SMOKE` dirs,
  `baselines/aifs_ic_only_truncated_2026-08-28`, `_archived/`, `scratch/`. Never delete anything with status
  `paper` in [data/STORE_INVENTORY.md](data/STORE_INVENTORY.md).
- Two code-level guards are missing (fewer-than-4-inits eval trap, silent CDS fallback in
  `MergedDataSource`).

## 5. What was changed for the handover (2026-11-02)

- Docs restructured: this folder ([DATA](DATA.md), [EXPERIMENTS](EXPERIMENTS.md), [REPRODUCING](REPRODUCING.md),
  [ENVIRONMENT](ENVIRONMENT.md), [HPC](HPC.md)), a README in every code and data directory, and a generated
  inventory of all 120+ datasets ([data/STORE_INVENTORY.md](data/STORE_INVENTORY.md), `tools/inventory_store.py`).
  The same pages were copied into the store (`$STORE/README.md` etc.).
- Dependencies pinned: exact host lock ([env/host-requirements.txt](../env/host-requirements.txt)), full
  package freezes of the six images ([containers/locks/](../containers/locks/)), git dependencies in the
  Dockerfiles pinned to the commits inside the images, SwissClim pinned to a commit, conda tools pinned
  ([env/conda-tools.yml](../env/conda-tools.yml)). The stale `uv.lock` was removed.
- Portability: 60 MB of figure/table inputs vendored under `tools/data/store/` (same layout as the store);
  regenerated outputs verified byte-identical. Analysis caches moved from purgeable scratch to
  `$STORE/analysis/`. IFS paths moved from the frozen a122 tree to the verified ab016 copies. Slurm account
  overridable via `AIENS_ACCOUNT`. `tools/milton/era5_control.py`, which an over-broad `.gitignore` pattern had
  kept out of git, is now tracked.
- Access: read ACLs for `s83` (and `ab016`) on the whole store (`tools/grant_group_read.sh`); read/write
  ACL for `olifu` on the whole store.
- Runs under another user: scratch paths default to `/iopsstor/scratch/cscs/$USER` (`#SBATCH` logs use
  `%u`), comments point to the `.claude/` files instead of the author's private memory, and
  `scripts/handover_demo.sh` checks the whole chain (venv, inference, evaluation) end to end.
- Agent memory split: technical notes public in `.claude/` (sanitised), paper-process notes local-only.
