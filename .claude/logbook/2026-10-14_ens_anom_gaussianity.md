Per-gridpoint SPW ensemble marginals are near-Gaussian (like the trained ensembles); the real non-Gaussian feature is a coherent per-member global offset in Aurora/GraphCast/AIFS SPW (not SFNO).

# Ensemble-anomaly Gaussianity check (2026-10-14, talk context, paper untouched)

Question (user): does Gaussian weight noise give Gaussian marginal distributions?

Why not the existing wd_kde output: `$STORE/baselines/<m>/eval/wd_kde/*ridgeline_enspooled.png` pools the raw
standardized FIELD over space/members vs ERA5 = climatology (T2m bimodal land/ocean), says nothing about the
ensemble distribution.

Method: z = (x_m - ensmean) / ensstd(ddof=1) per gridpoint, 2048 cos-lat-weighted random points, 28 inits
(every 4th of 112), leads 24..360 h every 24 h, 14 var-levels, all 7 M=10 baselines. Reference = finite-M
Gaussian null (10 iid N(0,1) standardized identically; |z| <= 2.85, null mean excess kurtosis -0.545,
mean |skew| 0.455). Fed through SwissClim `wd_kde.run` with the null as "Target" (ridgelines + per-lead W1).
- Tools: `tools/compute_ens_anomalies.py` + `submit_ens_anomalies.sh` (job 3660367, 8.7 min; first try
  3660344 OOM'd with 15-lead slabs per worker), `tools/plot_ens_anom_gaussianity.py` +
  `submit_ens_anom_plots.sh` (3660475 module, 3660687 replot). Login-node run gets killed -> sbatch.
- Data: `$STORE/analysis/ens_anom_gaussianity/<model>.npz`, `shape_stats.csv`, module output
  `wd_kde/<model>/wd_kde/`. Figures: `figures/ens_anom/{w1_vs_lead,shape_vs_lead,hist_48h,hist_240h}.png`.

Findings:
- Per-gridpoint shape: all ensembles within ~0.1-0.15 of the null in |skew| / excess kurtosis; SPW not
  systematically worse than AIFS-ENS/Atlas/FCN3. Pooled log-density at 240 h overlays the null for all.
- 24 h: SFNO/AIFS SPW sit on the null (linear response regime). Aurora (Z500, T850, Q850) and GraphCast Z500
  are PLATYKURTIC at 24 h (-0.63..-0.66 vs -0.545, 95% CI over inits +-0.02-0.03). Trained ensembles are
  already skewed in humidity at 24 h (Q |skew| 0.52-0.55); SPW are not (0.46-0.48).
- Humidity skew grows with lead in ALL ensembles (Q500 |skew| ~0.72 @240 h) -> physical (bounded at 0).
- Main SPW signature: global member offset = fraction of per-point var(z) carried by each member's
  globe-wide mean z. GraphCast T2m 0.36-0.41, MSLP 0.16-0.25, T850 0.2; AIFS SPW Z500/T2m ~0.2; Aurora
  Z500 0.2 -> 0.1. SFNO and AIFS-ENS/FCN3 ~0 (<0.03); Atlas only Z500 (0.2-0.3 early). Same mechanism as the
  spatial-mean SSR over-dispersion (tier1b): weight noise = per-member systematic bias.
- Consequence: SPW pooled W1 to the null is jagged and 3-10x the trained ones for T2m/MSLP/Z/T; that is an
  effective-sample-size artefact (each init repeats one 10-member configuration everywhere), so the iid
  null-vs-null floor (0.003) is not a valid floor for those models.
- GraphCast T2m stays platykurtic at all leads (-0.67 @240 h), with the largest offset; not explained yet.

Follow-up (same day): user cares only about the member-dimension shape, not the field shift. Replaced the
pooled histograms with detrended Q-Q plots `figures/ens_anom/qq_{24,120,240}h.png` (pooled z quantile minus
M=10 null quantile; grey references = Laplace/uniform/gamma k=4 members, same M and n). Split-half (inits
0-13 vs 14-27) 0.1%/99.9% quantiles agree to +-0.03, so curves are stable.
- 24 h: SFNO/AIFS SPW on zero everywhere; Aurora lighter tails in Z500/T500/T850; trained ensembles already
  right-skewed in Q500 (+0.2 at the upper end), SPW not.
- 240 h: winds/T/Z/MSLP within ~0.1 of zero for all models (Laplace is ~0.24); Q500 strongly right-skewed in
  ALL models, beyond the gamma k=4 reference; AIFS SPW T500/Z500 and GraphCast Z850 have a heavier lower tail
  (bump near z = -1.8); GraphCast T2m lighter left tail.
- Diagnostic power is limited at M=10: t5 members give W1 0.044, exponential 0.239 (calibration table in chat).

Follow-up 2026-10-15: added the ERA5 probabilistic climatology (WB2 definition as in the headline figure:
1990-2019, years as members, calendar month/day/hour matched, no smoothing) to the Q-Q and hist figures as
the black dashed line. 30 years split into 3 disjoint 10-member groups (year index mod 3) so M=10 matches the
null; `compute_ens_anomalies.py --baselines era5_clim` (job 3666068, 80 calendar days, 3.3 min) ->
`$STORE/analysis/ens_anom_gaussianity/era5_clim.npz` (ninit axis = 28 x 3). Removed the Laplace/uniform/gamma
reference curves and the reading guide from the Q-Q; the z equation + pooling sentence is on both figures.
- ERA5 climatology tracks the trained ensembles (and most SPW curves) almost exactly, including the heavier
  wind tails and the Q skew (Q500 climatology is even more skewed). So those departures are properties of the
  atmosphere's own variability. SPW-specific departures absent from climatology: AIFS SPW Z500/T500 lower-tail
  bump, GraphCast T2m/Z850.

Follow-up 2026-10-15 (physical units): npz now also store per-gridpoint ensemble `mean` and `std` (native
units; x_m = mean + z*std), re-extracted all 8 (job 3666102, 9.9 min; z bit-identical, old figures unchanged).
New `qq_phys_{24,120,240}h.png` (member anomaly a = x_m - mean in K/hPa/m-s/gpm/g-kg; detrended Q-Q against
Gaussian members with each dataset's OWN per-gridpoint std, x = standard normal quantile) and
`hist_phys_{48,240}h.png`. The pooled physical histograms are tent-shaped (log y) by construction: a mix of
gridpoints with very different spreads. Tail departures in physical units are dominated by the high-spread
points and follow ERA5 climatology in all models (e.g. MSLP heavier deep-low tail, ~2 hPa at the 0.05%
quantile). 48 h hist shows spread magnitude vs climatology (GraphCast narrowest in T/Q).
- `hist_full_{48,240}h.png`: full member values x_m (all 14 var-levels, linear y) vs ERA5 climatology. All
  models match the climatological distribution closely at 240 h; visible model-climate differences: Q850
  moist tropical peak (~12.5 g/kg) highest in GraphCast SPW, lowest in Aurora SPW; V10/V500/V850 SPW (AIFS,
  GraphCast) peakier at 0 than climatology; GraphCast T2m warm edge extends furthest.

Follow-up 2026-10-15 (single gridpoint): `tools/compute_point_ensemble.py` + `submit_point_ensemble.sh`
(job 3666144, 4.4 min) -> `$STORE/analysis/ens_anom_gaussianity/point_zurich.npz` (47.5N 8.5E, all 112 inits,
raw member values, all 14 var-levels, 240 h; ERA5 clim 30 years at the same calendar day+hour, 00/12 UTC).
`tools/plot_point_ensemble.py` -> `figures/ens_anom/point_zurich_{values,anomaly_qq}_240h.{png,pdf}` (KDE
values; detrended Q-Q over 0.6-99.4% with a 90% Gaussian sampling band at the RMS forecast spread; band
assumes 1120 independent values, but 12 h-apart inits are correlated, so it is somewhat too narrow).
- Values: forecasts warmer/moister/higher Z500 than the 1990-2019 climatology (2023-24 were warm years;
  model bias vs trend not separated).
- Q-Q: tail departures shared by all ensembles and ERA5 clim (T2m lighter cold tail ~+0.5 K at -2.5 sigma;
  Z500 lighter tails). SPW-specific: AIFS SPW heavier warm T2m tail (+0.85 K), Aurora SPW heavier low Z500
  tail (-35 gpm), both outside the band.
- Added Ny-Alesund (79N 12E), McMurdo (77.75S 166.75E), equatorial Pacific (0N 180E); all sea-level so 850 hPa
  is above ground. `compute_point_ensemble.py --points name,lat,lon ...` now extracts several points per pass
  (job 3666187, 4.5 min for 3). KDE 240 h: polar points close to climatology (Ny-Alesund SPW GraphCast/AIFS/
  Aurora have an extra cold shoulder ~250-255 K; McMurdo SFNO SPW lacks the moist Q850 tail >1 g/kg).
  Equator: Aurora SPW T2m ~1 K cold, GraphCast SPW ~1.5 K warm and broad, also visible in Z500 (Aurora low,
  GraphCast broad) = model drift at 240 h; Aurora/GraphCast SPW Q850 too peaked at ~12 g/kg, missing the dry
  tail. Trained ensembles cluster together, slightly warm vs 1990-2019 (El Nino 2023-24).
- Sensitivity: same 4 latitudes at longitude + 180 deg (job 3666204, 4.6 min): Arctic Ocean 79N 168W, North
  Pacific 47.5N 171.5W, Gulf of Guinea 0N 0E, Antarctica 77.75S 13.25W (1880 m, 850 hPa below ground, Q850
  extrapolated). `point_grid_opposite_values_240h.{png,pdf}` via `plot_point_ensemble.py --grid --grid-tag
  opposite --grid-points ...`. Robust across both equatorial points: Aurora SPW cold + low Z500, GraphCast SPW
  warm/broad T2m and broad Z500, AIFS SPW broad Z500. Location-dependent: polar cold tail (GraphCast at
  Ny-Alesund, Aurora at 79N 168W); North Pacific SPW washes out the bimodal T2m that trained ensembles and
  climatology show. Trained ensembles track climatology at all 8 points.

Follow-up 2026-10-15 (single ensemble, colleague's question): is the weight-perturbation distribution at
ONE init and ONE gridpoint non-Gaussian? (The pooled-over-init plots are mixtures of each init's predictive
distribution.) With M=10 one ensemble has almost no test power: Shapiro-Wilk at 5% rejects 10 Gaussian
members 4.9%, Laplace 15%, gamma k=4 14%, two clusters at +-2 std 18%, one +4 std outlier 39% (MC, 200k).
- `point_member_qq_zurich_20230102_0000_240h` (`plot_point_ensemble.py --members-only`): 3 vars x 7 models
  normal Q-Q of the 10 members, 95% SIMULTANEOUS envelope of 10 Gaussian members (own mean/std), SW p and
  Benjamini-Hochberg over the 21 panels (red if BH < 0.05: none; Aurora Z500 p = 0.01 -> BH 0.17), plus the
  raw share of the 112 inits at that point with p < 0.05 (4-12%, chance level 5%).
- `member_shapiro_rate`: share of single ensembles non-Gaussian at BH FDR 5%, family = all 112 inits x 8192
  cos-lat points per (dataset, var, lead), symlog y. Sample `<name>_sw.npz` (p only) from
  `compute_ens_anomalies.py --baselines ifs_ens era5_clim all --init-stride 1 --n-points 8192 --vars ... --tag
  sw --p-only` (job 3667091, 12 min; now also stores SW p and reads IFS-ENS: `ifs_ens` = all 50 perturbed
  members, no control on disk; `ifs_ens10` = members 0,5,..,45, one read). Plot job ~3 min.
- Sample size: old 28 inits x 2048 points gave BH Q850 CIs +-0.2-0.3 pp on shares of 0.15-1.7% (marginal);
  design effect (between-init var / binomial) T2m 1-2, Q850 2-4, Z500 3-25 -> space helps T2m/Q, time helps Z.
  New 112 x 8192 (16x tests): CIs +-0.02-0.1 pp, model differences resolved. More would not add detections:
  the BH share is set by the per-test power (M), not by m.
- Results @240 h (BH share): all M=10 forecasts and ERA5 clim 0% in T2m and Z500; Q850 0.2% (GraphCast/AIFS
  SPW), 0.7-1.2% (trained, Aurora SPW, ERA5 clim 0.9%), 1.8% (SFNO SPW). IFS-ENS 50 members: T2m 20%, Q850
  58%, Z500 7%; the SAME system at 10 members: 0 / 4.4 / 0%. -> M=10 is the limit: real departures exist
  (at least in IFS-ENS) but are invisible at 10 members. IFS-ENS10 Q850 (4.4%) > every other M=10 dataset.
- Atlas 336-360 h: a few inits with 41-44% raw rejection (Z500) -> nonzero BH share and wide band; not
  investigated (likely a degenerate member at late lead).
- Point files now also hold IFS-ENS (`compute_point_ensemble.py`, job 3667767, 5.6 min; 20.7% NaN = WB2 gaps).
  `point_member_qq_zurich_*` has 10 columns: + IFS-ENS 50 / 10 members + ERA5 clim (30 years), envelope per
  sample size, BH over the 28 non-missing panels (IFS-ENS Z500 is a WB2 gap at 2023-01-02 00 UTC +240 h).
  No panel significant; share of the 112 inits at Zurich with p < 0.05: IFS-ENS 50 T2m 21%, Q850 27% vs 3%
  at 10 members, ERA5 clim 2-16%.
- Example init switched to 2023-10-05 00 UTC (default `--member-init`; old 20230102 Q-Q removed): the only
  Zurich init at +240 h where IFS-ENS 50 has SW p < 0.05 in all three (T2m 1e-5, Q850 3e-4, Z500 0.037; 56/112
  inits have all three IFS fields). After BH over the 30 panels T2m/Q850 significant, Z500 not (BH 0.28). The
  same forecast's 10-member subset: p 0.66 / 0.52 / 0.20.
- Q-Q panels now use the global envelope test (Myllymaki et al. 2017; statistic = most extreme two-sided
  pointwise tail prob over the order statistics, 200k Gaussian sims per sample size, size 0.050 at M=10)
  instead of Shapiro-Wilk, so the band IS the test: BH over the panels rejects exactly p <= 0.05 k/m, and the
  band is drawn at that global level (here 1 - 0.0033, k=2 of 30). Member outside the band (red ring) <=>
  panel significant. Panel text = BH-adjusted p only ("all inits" share dropped). Example: IFS-ENS 50 T2m
  adj. p 1e-4, Q850 0.04, Z500 0.08; everything else >= 0.4. Rate figure stays Shapiro-Wilk (more power).
- Added 10 m wind speed (WS10, derived hypot(u10, v10) in every source incl. IFS-ENS/ERA5; `field()` in
  compute_ens_anomalies, VARS now 15) to both figures (`TEST_FEATURE`). `_sw` re-extracted with 4 vars (job
  3668054, 17.5 min; IFS-ENS WS10 has more gaps since u10 or v10 missing -> NaN, overall p NaN 24%), point
  files re-extracted (job 3668056). Rate @240 h: WS10 BH share 0 for every M=10 dataset and ERA5 clim (raw
  6.7-8.2%), IFS-ENS 50 members 9.5% (raw 23%), 5-11% over leads. Not strongly non-Gaussian globally (mean
  wind >> spread at most points); skewed only where weak, e.g. Zurich 2023-10-05: IFS-ENS 50 WS10 adj. p 1e-4,
  Aurora SPW p 0.02 (adj. 0.17). The Q-Q now has 40 panels (BH level 1 - 0.0038, k = 3).
- `member_shapiro_rate` zeros are real, not a bug: SW p at M=10 cannot go below ~1e-7 (observed min ~1e-6)
  and BH needs a >=20x excess of small p over chance; T2m/Z500 have ~5x at p < 1e-4 (0.04% vs 0.008%), Q850
  ~50x. y floor moved to -0.1 so the overlapping zero lines are visible.
- Not done: large-M SPW run at one init (est. AIFS SPW 200 members x 240 h ~2 node-h, ~170 GB T2m/Q/Z).
