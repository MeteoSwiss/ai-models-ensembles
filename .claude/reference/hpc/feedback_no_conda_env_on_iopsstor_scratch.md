---
name: feedback-no-conda-env-on-iopsstor-scratch
description: Live conda base is /capstor/store/cscs/mch/s83/sadamov/miniforge3 (off by default; activate with `conda activate base`; tectonic + gh live in it). NEVER host a persistent install (conda/miniforge env, long-lived venv) on /iopsstor/scratch -- the scratch purge silently deletes unaccessed files and rots the install. Persistent tooling goes on /capstor/store.
metadata:
  node_type: memory
  type: feedback
---

**Live install (use this).** Conda base is `/capstor/store/cscs/mch/s83/sadamov/miniforge3` (on `/capstor/store`, persistent). It is NOT auto-activated at login: `~/.bashrc` sets `CONDA_ROOT`/`MAMBA_ROOT_PREFIX` to it and defines lazy `conda`/`mamba` shell functions (`CONDA_CHANGEPS1=no`), so base is off until you activate it. Activate:
- interactive: `conda activate base` (the lazy function inits conda on first call).
- script / non-interactive (e.g. the Bash tool): `source /capstor/store/cscs/mch/s83/sadamov/miniforge3/etc/profile.d/conda.sh && conda activate base`.
`tectonic` (0.16.9, for `./build.sh`) and `gh` (git credential helper) both live in this base env, so LaTeX builds and gh-backed pushes need it activated / its bin on PATH. `~/.condarc` `pkgs_dirs` and the `~/.gitconfig` credential helpers point here (repointed off the purged iopsstor path 2026-07-07).

`/iopsstor/scratch` has an access-time-based purge: files not read for a while get deleted. That is FINE for transient spill (dask, TMPDIR, NumPy intermediates -- see [[feedback-never-use-tmp-on-hpc]], which correctly redirects spill TO iopsstor). It is NOT fine for a persistent install you depend on: the purge eats individual files out from under it and leaves a half-gutted env.

Why: 2026-06-13, miniforge3 lived at `/iopsstor/scratch/cscs/sadamov/miniforge3` (moved there from `/users` home to escape the home quota). git HTTPS broke (`git: 'remote-https' is not a git command`) because `git-remote-http` had been purged. An audit found **~1400 dangling symlinks (452 in `lib/`) across 219 packages** -- the symlinks survived but their real-file targets were deleted. Signature is selective-by-access: `python3.12` (used often) was present, `python3.12-config` (rarely run) was purged; shared-library version links (`libvulkan.so -> .so.1`, `libevent`, `libabsl`) dangled. So tools that work today can break tomorrow when a lib target ages out. NOT a relocation/prefix bug -- 0 stale `/users` prefixes; it is missing files. Reconfirmed 2026-07-07: a stray second miniforge at `/iopsstor/scratch/cscs/sadamov/miniforge3` was again gutted to just `pkgs/` (conda, gh, tectonic all gone), which is why a build+push that session couldn't find tectonic or gh. The live conda on `/capstor/store` was untouched; the only damage was stale `~/.condarc`/`~/.gitconfig` pointers still aimed at the dead iopsstor path (now fixed).

How to apply:
- Host conda/miniforge envs (and any long-lived venv) on `/capstor/store/cscs/<...>/` (persistent), NOT `/iopsstor/scratch`. For this user: `/capstor/store/cscs/mch/s83/sadamov/miniforge3`.
- Detect purge rot: `find $PREFIX -xtype l -not -path '*/pkgs/*' | wc -l` (dangling symlinks with MISSING relative targets = purge, not a bad copy -- a bad copy wouldn't selectively delete only the symlink targets).
- Repair = rebuild, do NOT copy the rotted tree (you'd propagate the holes). Capture intent first while conda still runs: `conda env export --from-history`, `conda list --explicit`, `pip freeze`. Then `mamba env create -p <new-capstor-prefix> -f from_history.yml` with `CONDA_PKGS_DIRS` pointed at a FRESH capstor cache (never reuse the purged iopsstor pkgs cache). Verify: 0 dangling + git HTTPS works. Then repoint `~/.bashrc` conda/mamba init at the new prefix (back it up first).
- Same risk hits uv venvs on iopsstor scratch (e.g. a project `.venv -> /iopsstor/scratch/.../venvs/...`). They are pyproject-regenerable so lower stakes, but if one starts failing imports, this is why.
