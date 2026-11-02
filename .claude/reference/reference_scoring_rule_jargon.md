---
name: reference-scoring-rule-jargon
description: "Demystified taxonomy of proper scoring rule terminology - proper, strictly proper, fair CRPS, almost-fair CRPS, naive CRPS. Used in CRPS, ES, VS literature."
metadata:
  node_type: memory
  type: reference
  originSessionId: c0399c1b-ed43-4160-9f18-321f47b9dad5
---

# Proper scoring rule terminology (demystified)

The jargon ("proper", "fair", "almost-fair") overloads everyday English with very narrow technical meanings. Each maps to a precise statistical statement.

## CRPS itself

For a forecast distribution F and a scalar verification y, CRPS has two equivalent forms:

```
CRPS(F, y) = integral (F(x) - 1{y <= x})^2 dx
           = E|X - y|  -  0.5 * E|X - X'|       (X, X' iid from F)
```

The second form is what everyone implements: a "skill" term (E|X - y|, closeness to truth) minus half a "spread" term (E|X - X'|, member-to-member spread). Reward closeness, penalise being too spread out.

## "Proper" - no incentive to lie

A scoring rule S(F, y) is PROPER if reporting the true distribution G minimises the expected score under G:

```
E_G[S(F, y)] >= E_G[S(G, y)]   for all F
```

STRICTLY PROPER if G is the unique minimiser. CRPS is strictly proper for any distribution on R with finite mean. That is the only content. If the loss is not proper, the model has a gradient incentive to lie (report inflated or sharpened distribution because it scores better than truth). Non-proper loss -> systematic miscalibration at training time.

## "Fair" - unbiased finite-M estimator

With only M ensemble members you replace E|X - y| and E|X - X'| with sample averages:

```
naive: spread term = (1 / M^2) * sum_{i,j} |x_i - x_j|
fair:  spread term = (1 / (M*(M-1))) * sum_{i != j} |x_i - x_j|
```

Naive divides by M^2 and includes i=j terms (which contribute zero), so it SYSTEMATICALLY UNDERESTIMATES E|X - X'|. Fair drops the diagonal, rescales, becomes the standard U-statistic bias correction. In expectation:

```
E[fCRPS(x_1:M, y)] = CRPS(F, y)             for all M >= 2
E[naive(x_1:M, y)] = CRPS(F, y) - bias(M),   bias > 0
```

Consequence: naive CRPS at finite M scores OVER-DISPERSIVE forecasts better than they deserve. Training with naive at small M and deploying at large M trains for the wrong target. Fair removes that.

The word "fair" means "fair across ensemble sizes" - M=2 fair-CRPS and M=50 fair-CRPS estimate the same quantity. Not anything moral.

## Why M=2 in training is OK (with fair)

Dense GPU activations forbid M=50 during backprop. Fair CRPS is unbiased at any M >= 2, so M=2 training is mathematically the same target as M=infinity training in expectation. That is the whole reason fair-CRPS matters for ML.

## "Almost-fair" CRPS (afCRPS)

ECMWF AIFS-CRPS paper: at training start, members are nearly identical, fair-CRPS gradient is dominated by spread term pushing members apart in essentially arbitrary directions. Destabilises early training.

Fix: interpolate naive and fair with parameter alpha in (0, 1]:

```
afCRPS = E|X - y|_hat - alpha * 0.5 * (1 / (M*(M-1))) * sum_{i != j} |x_i - x_j|
```

alpha = 1 recovers full fair CRPS. alpha < 1 leaves a small over-dispersion bias but tames early-training gradients. "Almost" = "we backed off slightly from fair to make Adam happy". Not deep.

## Who uses what

- [[model-aifs-ens]]: afCRPS (alpha < 1)
- [[model-fgn]]: full fair CRPS (alpha = 1, N=2 samples per location)
- [[model-fcn3]]: composite spectral CRPS
- [[model-atlas]]: stochastic-interpolant / score-matching loss (NOT CRPS-based)
- Your eval: CRPS for univariate, ES + VS for multivariate ([[energy-variogram-score]]). All same family. ES has the same naive/fair distinction; you compute ES at deploy time with M=10 fixed so you don't yet need fair-ES, but flag it if you ever train with ES.

## Common confusions

- CRPS vs CRPSS: CRPS = score (lower better). CRPSS = climatology-normalised skill score = 1 - CRPS / CRPS_clim (higher better). See [[crpss-migration]].
- "Score" is overloaded: loss vs verification metric vs skill score. All three coexist in the same paper.
- Multivariate analogues (Energy Score, Variogram Score) have their own fair/naive estimators with the same logic, but the literature renames them per paper. Treat any "fair X" as "U-statistic bias correction for finite M".
