---
name: references
description: "Running bibliography of papers cited in design discussions for this project. Topics include weight perturbation theory, NN ensemble methods, AI weather forecasting, and Bayesian deep learning."
metadata:
  node_type: memory
  type: reference
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# References

Bibliography of papers cited in methodology discussions for this project.
Organized by topic. Each entry: short tag, full citation, arxiv/doi, and a
one-line note on why it's relevant here.

## Weight perturbation for ML ensembles

- **HENS** -- Mahesh, A. et al. (2024). *Huge Ensembles Part I: Design of Ensemble
  Weather Forecasts using Spherical Fourier Neural Operators.* arxiv 2408.03100.
  Most directly relevant: SFNO + weight perturbation for probabilistic weather
  ensembles. Tunes magnitude empirically; ranges consistent with our 0.5–3%
  finding. Uses bred vectors + weight perturbation together.

- **Model Soups** -- Wortsman, M. et al. (2022). *Model Soups: averaging weights
  of multiple fine-tuned models improves accuracy without increasing inference
  time.* ICML, arxiv 2203.05482. Weight-space variance and combining
  perturbations. Implicit scaling arguments.

## Theory of perturbation magnitudes and flat minima

- **Flat minima (early)** -- Hochreiter, S. & Schmidhuber, J. (1997). *Flat
  Minima.* Neural Computation 9(1). Foundational: argues flat regions of the
  loss landscape generalize better.

- **Large-batch / flat minima** -- Keskar, N.S. et al. (2017). *On Large-Batch
  Training for Deep Learning: Generalization Gap and Sharp Minima.* ICLR,
  arxiv 1609.04836. Sharp vs flat minima and their generalization properties.

- **Loss landscape visualization** -- Li, H. et al. (2018). *Visualizing the
  Loss Landscape of Neural Nets.* NeurIPS, arxiv 1712.09913. Empirically shows
  flat basins in modern NN training.

- **NTK** -- Jacot, A., Gabriel, F. & Hongler, C. (2018). *Neural Tangent
  Kernel: Convergence and Generalization in Infinite Width Networks.* NeurIPS,
  arxiv 1806.07572. Linearization regime: δy ≈ J·δw. Used to justify √N
  scaling of σ across weight subsets.

## Bayesian deep learning / approximate posteriors

- **SWAG** -- Maddox, W.J. et al. (2019). *A Simple Baseline for Bayesian
  Uncertainty in Deep Learning.* NeurIPS, arxiv 1902.02476. Gaussian posterior
  approximation fitted to SGD trajectory. Posterior std scales 1/√N for
  parameter count N; supports the empirical 0.5–3% perturbation range.

- **SGLD** -- Welling, M. & Teh, Y.W. (2011). *Bayesian Learning via Stochastic
  Gradient Langevin Dynamics.* ICML. Foundational: noise added to SGD
  approximates posterior sampling. Conceptual basis for treating weight
  perturbation as approximate Bayesian inference.

- **SGD as approximate Bayes** -- Mandt, S., Hoffman, M.D. & Blei, D.M. (2017).
  *Stochastic Gradient Descent as Approximate Bayesian Inference.* JMLR 18.
  Formalizes the connection between SGD dynamics and posterior sampling.

## AI weather forecasting / probabilistic models

- **GenCast** -- Price, I. et al. (2024). *Probabilistic weather forecasting
  with machine learning.* Nature (online preprint earlier on arxiv). Diffusion
  model for global ensemble forecasting. Cited as the canonical purpose-trained
  probabilistic AI weather model.

## Statistical / methodological

- (Add as discussed.)

## Project-internal classics

- ECMWF IFS ENS documentation (50-member ensemble, bred vectors + SPPT
  perturbation strategy). Used as the classical-NWP benchmark and as the
  reference for stratified subsampling (members evenly spread in perturbation
  index).
