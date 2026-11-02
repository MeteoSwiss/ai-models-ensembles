# Variance-budget rescaling: paper said scalar-weight counts, code uses tensor counts. Corrected paper.

2026-07-03, attended. Triggered by a reviewer comment on the Frobenius/notation paragraph
(sec.~methods-framework) and a follow-up user probe on the variance-budget derivation.

## What was wrong
The paper defined the rescaling count N as *scalar-weight* counts in two places, but the figure
and all code use *tensor* counts:
- appendix A (variance budget): `N_group = sum_{W in S} |W|`, `N_total = sum_{W in W} |W|` (scalar entries).
- Fig. 1 (Phase 2 schematic) caption: "N_total and N_group are the numbers of learnable weights".

The implementation counts TENSORS, verified three ways:
- `scripts/submit_ablation.sh:76-88`: "N_GROUP_<model>[<group>] = number of perturbable tensors";
  N_GROUP_aurora=(encoder 33, backbone 594, decoder 17), etc.
- `figures/draw_perturbation_schematic.py`: n_total=644/87/264/242 (tensor counts), "box width by tensor count".
- Anchor arithmetic: Aurora encoder 0.044 = 0.01*sqrt(644/33); AIFS decoder 0.0275 = 0.01*sqrt(242/32);
  Aurora P3 0.0259 = 0.01*sqrt(644/96). All tensor counts.

## Phase 3 uses different count units (this is deliberate, not a bug)
The Phase 3 denominator counts the unit that is actually perturbed, which differs because the
mechanism differs per model:
- Aurora P3: whole-tensor weight perturbation (unet_bottom = 96 tensors) -> tensor count, same as Phase 2.
- SFNO P3: sub-axis slice of the 8 spectral filters (coarse-mode-cut) -> spectral DOF (23.6M of 572.5M).
  Tensor count is meaningless when you take a fraction of a tensor.
- GraphCast P3: runtime ACTIVATION perturbation on 42 of 40962 mesh nodes (mesh weights shared, no tensor
  subset selectable) -> mesh-node count. Code notes "the scaling formula breaks down for activation
  perturbation on a tiny spatial subset"; all three P3 models sweep several sigmas around the anchor.

## Fix (paper repo, NOT committed)
- `main.tex` Fig. 1 caption: "learnable weights" -> "learnable weight tensors".
- `main.tex` appendix A: rewrote so N counts perturbed tensors (`N_total=|W|`, `N_group=|S|`), assumption
  stated honestly ("each unit contributes, on average, a comparable share"), plus the three-granularity
  Phase 3 clause (tensors / spectral DOF / mesh nodes) and the anchor-then-sweep caveat.
- Symbol audit clean paper-wide: `\mathcal{W}` always a set, `\mathbf{W}` always a tensor, `\mathcal{S}`
  always the selection set. Also clarified the methods definition earlier the same session
  (`W = {W_1..W_L}`, scalar entries w, `sum_{w in W}` for the per-tensor Frobenius sum).
- tectonic compile PASS (main.pdf, only pre-existing layout warnings).
- Word cost: appendix grew ~65 words; paper at ~7,499/7,500 AMS limit -> needs a small offset
  (deferred Disc 5.1 ~33-word claw-back + one more trim). Not yet applied.

## How different is tensor- vs scalar-count scaling?
Exact identity: sigma_tensor / sigma_scalar = sqrt( mean(params/tensor in group) / mean(params/tensor in network) ).
Same-sized-tensor group -> identical; divergence is purely tensor-size heterogeneity.

AIFS, exact from the state-dict dump (encoder 34 tensors/26.9M, processor 176/201M, decoder 32/27M,
total 242/254.9M, sigma_full=0.01):
| group     | Ntens | Mparam | sig_tensor | sig_scalar | ratio |
| encoder   |    34 |   26.9 |     0.0267 |     0.0308 | 0.867 |
| processor |   176 |  201.0 |     0.0117 |     0.0113 | 1.041 |
| decoder*  |    32 |   27.0 |     0.0275 |     0.0307 | 0.895 |
(* = aifs_perturbed production baseline.) AIFS tensors are near-uniform in size (group means 0.79-1.14M
vs 1.05M network mean), so the two scalings agree within ~15%, well inside the Phase 2b sigma-sweep
granularity. SFNO would diverge more (8 giant spectral-filter tensors among 87); exact per-group SFNO/
Aurora param counts not on host (checkpoints in-container), would need a short container dump to tabulate.

## Appendix reframed: energy-target -> impact-proxy (2nd pass, same session)
The "equal weight-noise energy" framing was itself imprecise: tensor-counting does NOT hold energy fixed
for heterogeneous-tensor models. SFNO's Frobenius energy is concentrated in the 8 giant spectral filters;
the encoder (3 pointwise-lift tensors, ~1e4-1e5 params of 573M) holds almost none. Exact energy-matching
(= scalar/parameter-count scaling) would set the SFNO-encoder sigma to ~0.24-1.4 (vs the swept-optimal
0.054), which at sigma>~1 flips weight signs and destroys the layer. So exact energy-matching is the
pathological rule; tensor-counting is the gentler, empirically-better proxy.

Verified premise: SFNO Phase 2 winner IS the encoder at the tensor anchor 0.053852
(`evaluate_ablation.sh:342`, `calibration_winners.md:143`); the Phase 2b refinements (0.035, 0.080) both
underperformed, so the anchor is the swept optimum. AIFS decoder (production baseline) tensor 0.0275 vs
scalar 0.0307 = within 12% (AIFS tensors are size-uniform, so the two agree there).

Principled justification for counting tensors (not scalars): perturbation impact ~ sigma^2 sum_w F_w |w|^2
(Fisher/curvature-weighted), NOT sum_w |w|^2. Scalar-count assumes F_w=const; small bottleneck tensors
have high per-weight sensitivity + big matrices are low-rank/redundant, so per-weight impact is
anti-correlated with tensor size -> per-tensor impact is more uniform than per-scalar. Tensor-count is a
coarse impact proxy; the exact Fisher weighting is the paper's already-stated Future-work item
(`main.tex:1122`, Kirkpatrick2017ewc).

Edits (paper repo, NOT committed): rewrote appendix A spatial-anchor para around "scale sigma by unit
count because exact variance-matching over-perturbs small important tensors; coarse impact proxy refined
by the sweep; Fisher = future work"; SOFTENED the four flat "equal weight-noise energy" claims for
consistency (appendix intro, Fig.1 caption L~407, methods L376 "hold fixed"->"balance across
configurations", methods L387 + Phase3b L456 "holds energy constant"->"scales the perturbation up").
Temporal-anchor (refresh) energy-matching claim LEFT as-is (that derivation is exact). Also note: text now
uses |w|^2 (complex-safe modulus) - user edit, preserved. tectonic PASS.

## Takeaway
Tensor-count scaling is justifiable and simpler; keep it. Reframed the paper's justification from "equal
injected energy" (false for SFNO) to "coarse per-tensor impact proxy, refined by the sweep, exact Fisher
weighting deferred" - honest and robust to a sharp reviewer. The paper now matches the code, and the
Phase 2 vs Phase 3 count-unit difference is stated as a deliberate consequence of three perturbation
mechanisms. Word cost still outstanding (~7,499/7,500; needs the deferred Disc 5.1 claw-back).
