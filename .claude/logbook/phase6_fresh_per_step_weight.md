---
name: phase6-fresh-per-step-weight
description: "Phase 6 candidate: re-perturb weights at every AR step (fresh noise per step) instead of one realisation per member persisted across rollout. Theoretical fix for the spatial-mean coherence issue ([[spatial-mean-vs-pointwise-ssr]]). Expected ~7-8x reduction in spatial-mean spread at T=60 steps; pointwise SSR/CRPS roughly unchanged."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

## Premise

Current weight perturbation: each member gets ONE noise realisation `δW = σ·N` sampled at member init, applied to ALL T=60 rollout steps. The per-step output bias `b_kt = b_k` is the SAME across t, so cumulative spatial-mean bias = T·b_k after T steps. Std across members of spatial-mean ~ T·σ_b. This is why we see 6-9x larger spatial-mean spread than trained-probabilistic baselines at long leads (see [[spatial-mean-vs-pointwise-ssr]]).

Proposal: re-sample `δW_t = σ·N_t` per (member, step), apply only at step t, then reset. Per-step bias becomes independent across t. Cumulative spatial mean = sum of T independent random draws = random walk → std ~ √T·σ_b.

## Math

For T=60 steps:

```
spatial-mean-spread ratio  fresh / persistent  =  √T / T  =  1/√T  =  1/√60  ≈  1/7.7
```

Graphcast_all spatial-mean spread @ 360h drops from ~9x aifsens to ~1.2x aifsens. Aurora_encoder from ~6x to ~0.8x. Pointwise variance roughly preserved.

## Sigma scaling

Per-step contribution per cell stays the same, but the cumulative effect grows as √T instead of T. To maintain the calibrated pointwise spread you currently have, the per-step sigma needs to be `σ_fresh ≈ σ_persistent · √T ≈ 7.7·σ_persistent`:

| model | current σ (persistent) | predicted σ (fresh) | risk |
|---|---|---|---|
| graphcast_all | 0.01 | ~0.08 | safe |
| aurora_encoder | 0.025 | ~0.19 | safe |
| sfno_modes10 | 0.25 | ~1.9 | likely blows up the spectral conv weights -- may need Phase 3 mode-cut to confine |
| aifs (TBD) | ~0.01-0.03 (Phase 1 winner) | ~0.08-0.23 | should be safe given transformer architecture |

SFNO is the one to watch. The other three should accept the scaled sigma without obvious instability.

## Implementation

PyTorch models (Aurora, SFNO, AIFS): register `module.register_forward_pre_hook` that adds `σ·randn_like(param)` to each parameter immediately before the forward call. The hook is module-level, the noise is fresh per call, no checkpoint re-dump or disk I/O. ~30 LOC including a per-member RNG seed plumb-through.

JAX/Haiku models (GraphCast): the apply_fn(params, rng, ...) signature already accepts per-call params. Replace the closure variable `params` with `params + σ·jax.random.normal(rng, ...)` at each forward call. Same per-member RNG plumbing as the existing GC Phase 3 perturbation hook.

## Connection to existing GraphCast Phase 3 result

GC Phase 3 activation perturbation already does fresh-per-step noise on the FIRST n=42 mesh-node latent activations. The frozen-noise pilot just ran (2026-05-29) provides the matched control. Once the eval + intercomp from job 2426062 + 2426063 lands:

- **Strong hypothesis test for free**: compare the spatial-mean spread between gcsigma_1.0_gcnodes42 (fresh) and gcsigma_1.0_gcnodes42_frozen. If fresh has dramatically lower spatial-mean spread, the hypothesis is confirmed for the ACTIVATION mechanism. By analogy it should extend to weight perturbation, since both inject coherent noise that decorrelates by random walk over steps when fresh.
- This means **Phase 6a costs ZERO new forecasts** -- just a spatial-mean SSR diagnostic added to the existing intercomp data.

## Phasing

**Phase 6a (analysis-only, no new runs):**
1. Wait for GC frozen pilot eval + intercomp (2026-05-30).
2. Run the spatial-mean SSR diagnostic across gcsigma_1.0_gcnodes42 (fresh) vs gcsigma_1.0_gcnodes42_frozen.
3. If fresh shows dramatically lower spatial-mean spread → hypothesis confirmed for activation, by analogy expected to hold for weight. Proceed to 6b.
4. If fresh is similar to frozen → hypothesis weaker. Investigate why before committing to a full weight ablation.

**Phase 6a-prime (SSR diagnostic probe, ~120 forecasts) -- ADDED 2026-05-31**:
Before committing to a full bracket sweep, run ONE cell per model at the temporal-budget anchor (`σ_persistent · √60`):
- aurora_encoder @ 0.19 (= 0.025 · √60) -- 4 inits × 10 members = 40 forecasts
- sfno_modes10 @ ~1.0 (capped below the √60 prediction of 1.93 because of spectral conv blowup risk) -- 40 forecasts
- aifs_<winner> @ winner · √60 -- 40 forecasts

Read SSR at lead 240 h per model. Determines whether the temporal-budget rule is approximately right and whether the Phase 6b bracket should sit symmetrically around the anchor or shifted.

**Bracket multiplier scaling (using verified Phase 2b slope ~0.3, see [[sigma-sweep-design]])**:

Per-group SSR slope in log-sigma is ~0.3 (Phase 2b data). The same shallow slope likely applies to fresh-per-step weight perturbation since both target the same parameter subset. To cross SSR=1 from an anchor at SSR_anchor:

```
log-multiplier_target = (1 - SSR_anchor) / 0.3
sigma_multiplier_target = exp((1 - SSR_anchor) / 0.3)
```

Examples:
- SSR_anchor = 1.2 → mult = exp(-0.67) ≈ 0.51x  → "below" multiplier ~0.5x is right.
- SSR_anchor = 1.5 → mult = exp(-1.67) ≈ 0.19x  → "below" multiplier needs to be ~0.2x, MUCH lower than the symmetric {0.5x, 1x, 1.5x} bracket would suggest.
- SSR_anchor = 0.8 → mult = exp(0.67) ≈ 1.95x  → "above" multiplier needs to be ~2x.

So the {0.5x, 1x, 1.5x} bracket is only correct when `SSR_anchor` is within ~0.15 of 1.0. Outside that, the bracket must be widened (toward whichever side crosses SSR=1).

**Phase 6b (if 6a + 6a-prime confirm, ~360 new forecasts for ablation):**
Per (aurora_encoder, sfno_modes10, aifs_<winner>) -- graphcast_all is already done via existing Phase 3 fresh-noise data:
- Implement fresh-per-step weight hook (PyTorch / JAX -- see Implementation section above).
- 3-sigma bracket around `σ_anchor`, **designed asymmetrically based on the Phase 6a-prime SSR observation** (see [[sigma-sweep-design]] for the bracket rule):
  - If SSR_anchor ≈ 1: bracket symmetrically `{0.6x, 1x, 1.5x}` of σ_anchor.
  - If SSR_anchor > 1 (overdispersive): bracket below `{0.4x, 0.7x, 1x}`.
  - If SSR_anchor < 1 (underdispersive): bracket above `{1x, 1.5x, 2.5x}` (SFNO-decoder pattern).
- 4 inits × 10 members per cell = 40 forecasts × 3 cells × 3 models = 360 forecasts total.
- Pick sigma per model: argmin-CRPS at lead 240 h, with multi-metric inspection (CRPS + SSR + LSD + FSS) as tiebreaker (same Stage 2 rule as [[calibration-winners]]).

**Phase 6c (production baseline replacement, ~8960 forecasts):**
- For each model: full 224-init x 10-member baseline at the chosen sigma.
- Replaces or supplements the current 7-way baseline intercomp.
- Cost: same as current baseline run cost.

## Theoretical framing for the paper

This is the weight-domain analog of SKEB (stochastic kinetic energy backscatter) or SPP (stochastic perturbed parameter tendencies), used operationally in NWP. Trained probabilistic models implicitly do this -- their stochastic noise injection is per-forward-pass, fresh each step. Phase 6 matches that mechanism but in weight space rather than activation/feature space.

Interpretive shift: each member is no longer a deterministic forecast of a single perturbed model. Each member is a sample from a stochastic process where the parameters evolve as a random walk. Closer to the trained-probabilistic generative interpretation, and arguably more honest -- the post-hoc perturbation isn't really about uncertainty in which weights are correct.

## Open question

Whether the spatial-mean reduction is purely due to random-walk decorrelation (the √T argument) or whether the dynamics couple the per-step noises in ways that change the optimal sigma. The 3-sigma sweep in Phase 6b will reveal this empirically.
