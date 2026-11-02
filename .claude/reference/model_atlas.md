---
name: model-atlas
description: "NVIDIA Atlas-ERA5 (Chen et al. 2026, arXiv:2601.18111). Latent diffusion Transformer (DiT) sampled via stochastic interpolants. 4.3B params, 17 GB fp32, ~38 min/member on GH200 (structural, not setup-related). Training cost not documented."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# Atlas-ERA5 (NVIDIA, Chen et al. 2026)

## Reference (CORRECTED 2026-06-01)
Atlas DOES have a peer-review paper -- earlier memory note (`model_ifs_ens.md` style) said "no preprint available" which was wrong.

**Paper**: "Demystifying Data-Driven Probabilistic Medium-Range Weather Forecasting", [arXiv:2601.18111](https://arxiv.org/html/2601.18111v1) (2026).

**Model card**: [huggingface.co/nvidia/atlas-era5](https://huggingface.co/nvidia/atlas-era5).

**Implementation**: [earth2studio/models/px/atlas.py](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/px/atlas.py), [earth2studio/models/nn/atlas.py](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/models/nn/atlas.py).

## Architecture
- **Latent diffusion Transformer (DiT)** + decoder DiT, total ~4.3 B params (2.5 B encoder/latent + 1.8 B decoder).
- Stochastic interpolant sampling (Albergo & Vanden-Eijnden 2023 framework; Chen et al. 2024 arXiv:2403.13724 wrapper).
- Default sampler: `sinterpolant_sample_steps=60` (atlas.py:180), i.e. 60 denoising iterations per forecast call.
- Uses **NATTEN** (Neighborhood Attention Extension) for windowed attention. `try/except` import in `nn/atlas.py` -- if NATTEN missing, model instantiation FAILS with `OptionalDependencyFailure("atlas")`. NO silent slow-path fallback.

## Atlas variants (per paper)
- **ATLAS-SI** -- stochastic-interpolant variant (the default we use in the 7-way intercomp).
- **ATLAS-EDM** -- EDM-style denoising variant.
- **ATLAS-CRPS** -- separately trained with CRPS-loss head. **28x faster than ATLAS-SI** at inference. Conceptually similar to AIFS-CRPS for AIFS.

## Inference cost (verified)
Per the paper's Table 1 (single A100):
- ATLAS-SI: **94 s per 6 h step** -> 10-day forecast = 40 steps = **63 min/member**.
- ATLAS-EDM: 88 s/step -> 59 min/member.
- ATLAS-CRPS: 3.3 s/step -> 2.2 min/member.

Our measured cost on Clariden GH200: **38 min/member** for ATLAS-SI (10 members in production-baseline runs, 1120 forecasts total, parsed from tqdm `61/61 [37:53<...]` markers).

**38 min on GH200 vs 63 min on A100 = ~1.65x speedup**, consistent with GH200 being ~1.5x faster than A100 on this workload. **Our setup is NOT pathologically slow** -- the cost is structural to the multi-step DiT sampling. Verified 2026-06-01 ([[feedback-verify-paper-claims]] discipline).

## Training cost
**Not documented** on the model card or in arXiv:2601.18111. The paper covers training methodology but does not state GPU type/count/wall-clock or aggregated GPU-hours. Worth asking NVIDIA Earth-2 team if quoting in the paper.

## Diagnostics if you ever need to push lower
1. `python -c "import natten; print(natten.has_fna(), natten.has_fused_na())"` -- confirm fused Hopper kernels.
2. `nvidia-smi dmon -s u` during a run -- if SM util < 60%, kernels are not saturating.
3. `model = model.to(torch.bfloat16)` or `torch.autocast("cuda", dtype=torch.bfloat16)` -- atlas.py never sets a dtype.
4. Try `sinterpolant_sample_steps=30` (half default) -- the SI sampler is the dominant cost.
5. Switch to ATLAS-CRPS variant (28x faster) if a CRPS-trained model is acceptable for the use case.

## Strengths / limitations (for our paper context)
- **Headline limitation**: 38 min/mbr makes large-ensemble case studies expensive. A 50-member case-study run takes ~8 h wall-clock on a single 4-GH200 node, dictating the practical SLURM-walltime ceiling for any fair Atlas comparison.
- **Stochastic-interpolant framework** is the most sophisticated of the trained-probabilistic baselines (AIFS-CRPS uses conditional-LN noise, FCN3 uses learned spherical-diffusion). Worth flagging in related-work as the most "diffusion-like" of the three.
- **No transparent training cost**: paper's reproducibility for the cost-of-training story is weaker than AIFS-CRPS or GraphCast which give explicit numbers.

## Related
- [[paper-phase-structure]] (overall narrative)
- [[model-aifs-deterministic]] (compare aifs det vs aifsens vs atlas training-cost gap)
- [[sigma-sweep-design]] (perturbation sigma vs trained noise)
- [[feedback-verify-paper-claims]] (don't cite "no preprint" without checking -- atlas paper exists since 2026 Jan)
