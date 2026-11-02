# a122 -> ab016 project migration (CSCS Alps / Clariden)

**Status as of 2026-08-24:** old allocation `a122` is being replaced by the new funded
allocation `ab016`. Both are real Unix groups (`groups` shows both: `s83 ab016 s83c s83d
msclim a122`) - they don't resolve via `getent group <name>` on this node (SSSD doesn't
answer for them directly), so don't rely on `getent`/`id` lookups to confirm they exist.

## Compute (Slurm)

- `sacctmgr show assoc user=sadamov format=account,user,partition,maxsubmit,maxjobs,qos`
  shows account `a122` has QOS `stop` (frozen), account `ab016` has QOS `normal`.
- Default account on `sbatch` resolves to `a122` and silently fails with
  `AssocMaxSubmitJobLimit` / "Job violates accounting/QOS policy" - always pass
  `--account=ab016` explicitly.
- For pure data-movement jobs (rsync, tar, no compute), use the dedicated `xfer` partition
  (2 nodes, 24h time limit) instead of burning a GH200 compute node - `sinfo` lists it
  alongside `debug`/`normal`/`low`.

## Storage

- All project data lives on **capstor store**: `/capstor/store/cscs/swissai/{a122,ab016}/`.
  No other org tree (`mch`, `ethz`, etc.) has an `a122`/`ab016` dir. `/iopsstor/store/cscs/
  swissai` exists but is `d---------` root:root - dead/unreadable, not in use.
- `ab016` subdirectories are created `setgid` (`rws` on group) - new files/dirs written
  under them inherit group `ab016` automatically. When copying from `a122`, drop `-o -g`
  from rsync (`-rlptD` not `-a`) rather than trying to preserve the old `a122` group, which
  a non-root user can't set anyway.
- Migration is copy-only, never move (`do not move data, always copy` was explicit user
  instruction) - old `a122` data stays in place until the copy is verified.
- Full per-item audit (what's only in a122, what's already migrated, size estimates,
  flagged conflicts like the `ch17`-group `cerra-rr...zarr` sitting inside the a122 tree,
  and the `era5_coarse` vs `ERA5_coarse` naming drift) was a one-off report, not repeated
  here - regenerate via `comm -23 <(ls a122) <(ls ab016)` if needed again, don't assume the
  2026-08-24 snapshot is still accurate for anyone else's data.
- Personal scratch (`/capstor/scratch/cscs/sadamov`, `/iopsstor/scratch/cscs/sadamov`) is
  entirely group `s83`, unrelated to this migration - nothing to check there for this user.

## sadamov's own copy

`/capstor/store/cscs/swissai/a122/sadamov/` (5.5TB) -> `.../ab016/sadamov/`, submitted as
Slurm job on the `xfer` partition, account `ab016`:

```bash
sbatch --account=ab016 <<'EOF'
#!/bin/bash
#SBATCH --job-name=a122-to-ab016-sadamov
#SBATCH --partition=xfer
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=/capstor/scratch/cscs/sadamov/a122_ab016_copy_%j.log

rsync -rlptD --info=progress2 --stats \
  /capstor/store/cscs/swissai/a122/sadamov/ \
  /capstor/store/cscs/swissai/ab016/sadamov/
EOF
```

Verify completeness after the job finishes with a dry run (`rsync -avn` between the same
two paths should show nothing left to transfer) rather than trusting a clean exit code
alone.

**Incident (2026-08-25):** first attempt used `--mem=4G` and got OOM-killed at 99% (item
~50k of 57k files) once rsync's incremental recursion hit the zarr chunk stores (tens of
thousands of tiny files inflate rsync's in-memory file list well past a few GB). `xfer`
partition nodes have 250GB RAM - use `--mem=64G` (or more) for any rsync job over a zarr
tree, not 4-8G. Resuming is safe: rerun the identical rsync command, it skips
already-matching files by size+mtime and continues from where it stopped.
