---
name: clim1990-crps-perf-fix
description: "30-year climatology CRPS bottleneck -- replaced per-bin diff-matrix loop (95 min/key) with per-bin Gini-identity sort + streaming weighted sum (~2 min/key). All-bins-at-once numpy sort variant OOM'd, kept the OOM lesson in code."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# clim1990 CRPS perf fix

The `compute_climatology_1990_2019.py` per-(var, level) job has two cost phases:

1. **Year pull** (~120 min): year-at-a-time `.load()` of `(1460, 721, 1440)` slabs from the WB2 ERA5 zarr. I/O-bound; left untouched.
2. **LOO fair-CRPS over (DOY, hour) bins**: originally **~95 min** because it allocated the `(30, 30, 721, 1440) = 3.7 GB` diff matrix 1464 times. This was the bottleneck that pushed per-key wall to 216 min and forced the previous TIMEOUT chain.

## Fix shape (in `fair_crps_loo_vectorized`)

Use the Gini sort identity:
- For sorted `x_(1) <= ... <= x_(n)` per pixel, mean over k of LOO fair-CRPS equals `sum_{i,j} |x_i - x_j| / (2 n (n-1))`.
- The pairwise sum equals `2 sum_i (2i - n - 1) x_(i)` (1-indexed).
- So each bin becomes ONE sort + ONE streaming weighted sum -> O(N log N) per pixel instead of O(N^2). With N=30 that's about a 60x speedup.

Per-bin transient is ~250 MB scratch (one sorted `(30, 721, 1440)` slice). 1464 bins -> ~1-2 min CRPS phase per key.

Verified against brute-force pairwise sum at 4e-8 max abs err across all-present, partial (5 years missing), and skipped (n<3) bins.

## What I tried first that OOM'd (kept for next-time-me)

I first wrote an **all-bins-at-once** variant: `stage.sort(axis=0)` on the full `(30, 1464, 721, 1440)` cube, then `np.einsum("ib,ibyx->byx", weight_grid, stage)`. With 2 workers this OOM'd job `2488607` mid-CRPS-phase at 800 GB.

Numpy's `ndarray.sort(axis=0)` on a C-order array where axis 0 is the slowest-varying dim isn't actually in-place at the C level: numpy's axis-non-contiguous sort path allocates a transposed scratch (~181 GB at this shape). With 2 workers concurrent the combined transient exceeded the cap.

The `optimize=True` einsum may also have built an `(i, b, y, x)` intermediate; either way the all-bins version had a hidden ~2x memory blowup that the per-bin version doesn't.

**Rule:** for sort-along-axis-0 on very large cubes, sort per-slice in a Python loop. The per-iteration GC keeps the scratch transient short-lived. Don't trust numpy's "in-place" sort for non-contiguous axis-0 on huge arrays.

## Memory budget at --parallel 2

Per worker (peak during pull): `stage` (181 GB) + dask buffers (~30 GB) + accumulators (~24 GB) + numpy temps (~12 GB) -> ~250 GB. Two workers -> ~500 GB. Fits the 800 GB cap with headroom. **`--parallel 3` OOM'd (job `2487050`)** because the three workers' pull-phase peaks overlapped; even with the per-bin CRPS, 3 staging cubes + dask = ~750 GB and the dask overhead variance tipped it past 800.

## What's parametric, what's load-bearing

- `--workers 16` (dask thread pool per process): no longer the bottleneck once year-pull is bulk-loaded. Don't bother tuning.
- `--parallel 2`: hard ceiling on `--mem=800G` partition. Don't try 3.
- `chunks={"time": 200, "level": 1}` in `open_wb2`: load-bearing. With `{"time": 1460}` (one calendar year) each 3D-var chunk read becomes 222 GB which OOM-killed `2476374` at 2 min. With `level=37` packed in one chunk and no `level=1` hint, `.sel(level=lvl).load()` materialises the full 37-level slab before slicing.
- Per-(var, level) JSONL checkpoint at `/iopsstor/scratch/cscs/sadamov/climatology_1990_2019_checkpoint.jsonl`: per-key atomic append + fsync. Resume skips committed keys and merges into the same output JSONs on the fly.

## Related

- [[paper-add-climatology-persistence]] (downstream consumer: Tab. 3 / Fig. 2 CRPSS denominator)
- [[feedback-unbuffered-stdout]] (the script is `sys.stdout.reconfigure(line_buffering=True)` so live progress lands in the SLURM log)
- [[feedback-cluster-max-resources]] (--mem=800G --time=12h is the cluster max)
