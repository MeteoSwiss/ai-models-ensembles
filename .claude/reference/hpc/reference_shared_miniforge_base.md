---
name: reference-shared-miniforge-base
description: "On CSCS (Clariden/Alps), sadamov's system-wide CLI tools live in a shared miniforge base env at /capstor/store/cscs/mch/s83/sadamov/miniforge3. `conda activate base` first to get gh, etc. gh is pre-authenticated as sadamov."
metadata:
  node_type: memory
  type: reference
---

sadamov keeps a shared miniforge install on the persistent `/capstor/store` (mch/s83) project space, NOT in `$HOME` and NOT on the purged iopsstor scratch:

- Miniforge root: `/capstor/store/cscs/mch/s83/sadamov/miniforge3`
- The `base` env holds system-wide CLIs. Access them by activating base first: `conda activate base` (or call the absolute path directly).
- **GitHub CLI** is here: `/capstor/store/cscs/mch/s83/sadamov/miniforge3/bin/gh` (gh 2.94.0 as of 2026-07-11). It is already authenticated as user `sadamov` via the token in `~/.config/gh/hosts.yml` - no `gh auth login` needed. So GitHub ops = `conda activate base && gh ...` (create repos, PRs, `gh api`, etc.).

How to apply:
- Before assuming a CLI (gh, and likely others) is missing because `command -v` fails on a bare login shell, check the miniforge base env: `conda activate base` then retry, or call `/capstor/store/cscs/mch/s83/sadamov/miniforge3/bin/<tool>` directly.
- Do NOT download/install gh (or other base-env tools) separately on Clariden - reuse the shared base env. The only exception is the x86_64 `gh` for Bristen (below).
- SSH to GitHub also works from CSCS login nodes (verified), so `git push` to an existing repo works without gh.
- Keep conda envs off iopsstor scratch (see [[feedback-no-conda-env-on-iopsstor-scratch]]); this base lives on capstor store, which is correct.
- This base is an aarch64 build and is UNUSABLE on Bristen (x86_64, e.g. `bristen-ln001`; check `uname -m`). There, do not source `conda.sh`: `conda activate` falls through to bash running conda's Python script, and `import sys` invokes ImageMagick `import`, which hangs waiting for an X display. Since git's credential helper is this aarch64 `gh`, HTTPS push/fetch auth fails there: use SSH URLs (`git push git@github.com:OWNER/REPO.git ...` works). (Hit 2026-10-20.)
- Bristen `gh`: the official x86_64 release gh 2.94.0 (same version as the base, so the shared `~/.config/gh/hosts.yml` token works and neither migrates the config) lives at `/capstor/store/cscs/mch/s83/sadamov/opt/x86_64/gh-2.94.0/bin/gh` (checksum-verified, installed 2026-10-20). `~/.bashrc` prepends it to PATH only when `uname -m` is `x86_64`, so plain `gh` works in a fresh bash on Bristen (not in `~/.zshrc`). Never put it in `~/.local/bin` (shared with Clariden). `curl`/`wget` are denied in Claude settings, so the user runs downloads.
