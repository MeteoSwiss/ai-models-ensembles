---
name: model-ifs-ens
description: "ECMWF IFS Ensemble Prediction System (IFS-ENS / EPS). The classical physical-NWP ensemble reference baseline. Configuration, IC perturbation scheme, model-error perturbation scheme, and historical context."
metadata:
  node_type: memory
  type: project
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

# IFS-ENS (ECMWF Ensemble Prediction System)

## Overview
- Operational since 1992 (Palmer et al. 1993, Molteni et al. 1996); current configuration is IFS Cy49r1 (operational 12 November 2024).
- Medium-range ensemble: **50+1 members** (1 control + 50 perturbed).
- Resolution: **TCo1279L137** (~9 km horizontal, 137 vertical levels, model top 1 Pa). Time step 7.5 min.
- Initialised twice daily at 00 and 12 UTC.
- Lead time: **day 0 to day 15**.
- Output archived in MARS as `stream=enfo`, `type=cf` (control) / `type=pf` (perturbed).
- All non-linear integrations run in single precision since 11 May 2021. Coupled to NEMO (ORCA025, 75 levels) every hour from t=0.
- Separate sub-seasonal ensemble (100+1 members, TCo319, to day 46) supersedes the old VAREPS since 27 June 2023.

## Initial condition perturbations
- Hybrid scheme since June 2010 (Buizza et al. 2008): **EDA perturbations + singular vector perturbations**.
- **EDA component**: 50-member Ensemble of Data Assimilations at outer-loop resolution TCo639. Member k perturbation = `x_k - mean(x)` from 6 h EDA forecasts. Observations perturbed with stochastic noise; model uncertainty in EDA represented by SPP. One unique EDA perturbation per EPS member (Lang et al. 2019, Isaksen et al. 2010).
- **Singular vector component**: leading 50 SVs per extra-tropical hemisphere + leading 5 SVs in up to 12 tropical-cyclone target regions. Computed at **T42L91** by tangent-linear/adjoint integrations of an adiabatic IFS (diabatic for TC targets) over a **48 h optimisation interval** using the **total energy norm** at initial and final time. Solved via Lanczos algorithm. Members are independent Gaussian linear combinations (sigma `beta = gamma/kappa`, gamma = 0.0175, clipped to ±3 sigma).
- EDA samples errors that have grown; SVs sample structures expected to dominate forecast error from day 2 onwards. Perturbations centred so ensemble mean = control analysis.

## Model error perturbations
- Since **12 November 2024 (Cy49r1)**: **Stochastically Perturbed Parametrizations (SPP)** scheme, replacing SPPT in all ensemble applications.
- SPP perturbs **27 uncertain elements** inside the parametrizations of convection, large-scale cloud, radiation, surface exchanges, vertical turbulent mixing, and orographic gravity wave drag.
- Each element has an independent random field combining two AR(1) components with spatial decorrelation scales **1000 km and 3000 km** and temporal decorrelation scales **3 days and 30 days** (Ollinaho et al. 2017, Lang et al. 2021b, Leutbecher et al. 2024).
- Pre-2024 the equivalent role was played by **SPPT** (multiplicative noise on total physics tendencies, Buizza et al. 1999), introduced 1998, revised 2009/2010/2018. SKEB is no longer mentioned in Cy49r1 docs.

## What "perturbed analyses" subsampling means for our work
- For the IFS-ENS-IC baseline we drop the control and take the 50 type=pf perturbed analyses at t=0, then subsample 10 of 50 deterministically (stratified).
- For the IFS-ENS forecast baseline we use the same 10/50 stratified subsample applied to the operational ENS forecast.

## Differences from AI-based ensembles
- Physical model: solves primitive equations on TCo1279 grid coupled to NEMO; no learned weights.
- Ensemble spread comes from (a) hybrid EDA + SV initial perturbations, and (b) SPP stochastic physics during integration.

## Strengths / limitations (relevant to our paper)
- Gold-standard medium-range ensemble; reference benchmark for AIFS-CRPS, AIFS-ENS and our post-hoc perturbation baselines.
- Decades of operational tuning; spread-skill calibration is a design target.
- Computationally expensive (~50 perturbed TCo1279 forecasts coupled to NEMO twice daily).

## Sources
- ECMWF, IFS Documentation Cy49r1, Part V: Ensemble Prediction System (Nov 2024). https://www.ecmwf.int/en/elibrary/81627-ifs-documentation-cy49r1-part-v-ensemble-prediction-system
- Leutbecher, M. and Palmer, T. N. (2008). "Ensemble forecasting." *J. Comput. Phys.* 227, 3515-3539. https://doi.org/10.1016/j.jcp.2007.02.014
- Buizza, R., Miller, M. and Palmer, T. N. (1999). "Stochastic representation of model uncertainties in the ECMWF ensemble prediction system." *QJRMS* 125, 2887-2908.
- Leutbecher, M. et al. (2017). "Stochastic representations of model uncertainties at ECMWF." *QJRMS* 143, 2315-2339. https://doi.org/10.1002/qj.3094
- Lang, S. T. K. et al. (2021b). "Revision of the SPPT model-uncertainty scheme." *QJRMS* 147, 1364-1381. https://doi.org/10.1002/qj.3978
- Buizza, R., Leutbecher, M., Isaksen, L. (2008). "Potential use of an ensemble of analyses in the ECMWF EPS." *QJRMS* 134, 2051-2066.
