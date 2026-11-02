# .claude/ - project memory for Claude Code

This folder plus the top-level `CLAUDE.md` is the working memory that Claude Code sessions accumulated
over the project (May to November 2026): conventions, audits, and a dated logbook of every finding, bug
and decision. Claude Code loads `CLAUDE.md` and `rules/` automatically when started in a checkout; the rest
is read on demand. It is also plain Markdown, useful to humans for the "why" behind the code and the data
(the polished documentation is in `docs/`).

Made public on 2026-11-02 for the handover. Everything here was sanitised: no credentials, private
contact details, personal information or journal-process details. The paper-process half of the notes
(manuscript review rounds, writing, word counts, bibliography workflow, submission mechanics) is local-only:
it lives in the gitignored `*/paper/` subfolders of the original author's checkout, and a full copy is on
Clariden in `/capstor/store/cscs/swissai/ab016/sadamov/ai-models-ensembles/agent-memory-local-2026-11-02/`
(group `ab016`).

## Layout

| Path | Content | Loaded |
|---|---|---|
| `../CLAUDE.md` | condensed project conventions: paths, accounts, containers, run/eval commands, data rules, reproducibility | always |
| `rules/` | short rules, some scoped to files via `paths:` frontmatter (`plot-units.md`, `zarr-v3-fill-value.md`) | always / when touching matching files |
| `reference/` | stable reference notes: model cards (`model_*.md`), checkpoint perturbation audit, metric audit, sigma-sweep design, noise-injection taxonomy, scoring-rule jargon, project conventions (SIGSEGV history, intercomparison design, CDS/JAX specifics), reference list | on demand |
| `reference/hpc/` | generic CSCS Alps/Clariden lessons (scratch purge, SSL on compute nodes, login-node jobs, eccodes on aarch64, the a122 -> ab016 migration, ...) | on demand |
| `logbook/` | ~85 dated entries `YYYY-MM-DD_<slug>.md` plus undated topic notes (`phase*_*.md`, `calibration_winners.md`, ...); first line = one-line summary | on demand |
| `memory/MEMORY.md` | snapshot of the per-user auto-memory index at handover (state, guards, session log) | on demand |

Start with `logbook/2026-11-02_technical_facts_from_local_paper_notes.md` (facts recovered from the local
half), `logbook/calibration_winners.md` (why the production picks), and the most recent dated entries.

## Conventions

- Logbook entries are history: they describe the state at their date and are not rewritten later. When a
  later entry corrects an earlier one, the earlier one gets a short "SUPERSEDED" banner (example:
  `2026-09-01_milton_pipeline_regen_ifsens_detection_fix.md`).
- `[[slug]]` links point to other notes by their `name:` slug, usually the file name with `-` for `_`
  (`[[calibration-winners]]` = `logbook/calibration_winners.md`; find any with
  `grep -rl "name: <slug>" .claude/`). Some targets are in the local-only paper notes or were never
  written; treat those as references, not broken files.
- Older entries cite "MEMORY.md" sections: that is the pre-handover per-user memory, archived with the
  local-only notes in `/capstor/store/cscs/mch/s83/sadamov/handover-private/ai-models-ensembles/`.
- Entries mention paths on Clariden (`/capstor/...`, `/iopsstor/...`) and Slurm job IDs; job logs are in
  `$STORE/*_logs/`.
- One internal-review logbook entry was renamed for the public copy
  (`2026-08-25_internal_review_response_runs.md`).
- New notes: findings and decisions as a new dated logbook entry, durable conventions in `CLAUDE.md` or
  `rules/`. This folder is public: keep anything personal or journal-related in `*/paper/` (gitignored).
