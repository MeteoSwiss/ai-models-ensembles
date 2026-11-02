# 2026-07-07 ATTENDED - AIES blind peer review + fix implementation (S3 geostrophic regen in flight)

One-line: ran a full AMS AIES blind referee review (9-lens multi-agent workflow + adversarial verify -> 30 surviving findings, 0 surviving majors), then implemented the user-selected fixes; S3 (geostrophic f bug) triggered a Milton-box pooled regen job 2701897 still running at commit time.

## The review
(paper-process detail kept in the local notes)

## Fixes IMPLEMENTED this session (paper repo main.tex, all word-neutralised, compiles clean, 7,495 conservative)
- S1 (user-selected word-adder): new `\textbf{No within-family baseline.}` Limitations sentence (MC-dropout/SWAG need training access the frozen checkpoints lack). Also fixed "let the held-out grid arbitrate" -> "validate the selected picks" (methods minor: tied rivals never run on the 112-grid; all 4 picks are just CRPSS@240h argmax).
- S2: refresh variance-budget narrative (sec 4.3 + App A). The matched-variance identity ALONE would leave the spatial-mean SSR unchanged (verified algebraically: sigma_N=sigma_frozen*sqrt(T/N) => refreshed accumulated std sqrt(TN)*sigma_N = T*sigma_frozen = frozen). The spatial-mean fix is NONLINEAR (trajectory-divergence decorrelation), not the variance budget. App A now says this explicitly; budget preserves per-pixel/CRPS only.
- S4: App B "over the same initialisations as the numerator" reconciled with sec 5.2 - denominator uses ALL 112, the gappy IFS-ENS numerator is the skipna-reduced subset (mismatch pointed to sec 5.2).
- Technical: 0.036-gap caption footnote (0.256-0.221=0.035 rounded, 0.036 unrounded); SFNO spatial-mean SSR "sits within it (TP band 0.62 to 1.42)" [data: AIFS-ENS 0.618, FCN3 0.747, Atlas 1.415, sfno 0.663 => within]; det-RMSE "deterministic intercomparison" -> "(not shown)" + RMSE-not-one-of-ten note; graphcast coarse-node SSR tagged "ablation-grid"; 229M reattributed to AIFS-ENS not v0.2.1 (caption + body L897); abstract "operational ECMWF ensemble (IFS-ENS)".
- Physics (word-neutral): Milton MSL native-TCo1279 causal claim softened to "~8 hPa spread not separable by native resolution (a 0.25deg ML baseline nearly matches it)"; spectral-excess attribution hedged ("appears to track ... more than the perturbation").
- Trims (~95 words, to offset additions): lead-resolved RMSE para, sec 5.1 SFNO-probe + interpretability-pattern paras, GraphCast-boundary restatement, per-variable pattern, Pu2026cnop echo, refresh 240h-horizon aside, cost "measured on identical hardware" clause.
- S5 (repro: tag/DOI, Earth2Studio pin, CSV-on-request, seed, JAX note, TP checkpoint ids) SKIPPED per user ("not ready, need more time").

## S3 = the real find (geostrophic Coriolis bug), IN FLIGHT
- Fig C2 (`bivariate_geostrophic_500hPa.pdf`) used a FIXED f=1.0e-4 s^-1 over the 5-40N Milton box. That is f at ~43.3N (Europe) - WRONG for a subtropical box. Correct box-centre f (22.5N) = 2*Omega*sin(22.5) = 5.58e-5.
- Discovered C2 was a THREE-WAY inconsistency: figure = Milton-box ENSMEAN (byte-identical in commit 1b0e5dc, `multivariate: mean` in submit_geostrophic_milton.sh), caption text = "pooled 1M", quoted "3-6% vs 2-3%" numbers = actually GLOBAL pooled with f=1e-4 (reproduced exactly from `eval/` enspooled NPZ + |wind-Ug|>15 m/s). So the 07-06 note "eval_milton still ensmean, fine" was NOT fine for C2.
- Ageostrophic-% method VALIDATED: fraction of joint-density mass with |wind - (g/f)|gradH|| > 15 m/s. On GLOBAL enspooled f=1e-4 reproduces the caption (aurora 3.4/gc 4.9/sfno 5.8/aifs 3.8/atlas 5.7; TP+truth 2.0-2.9). On Milton-box ENSMEAN f=1e-4 gives ~0.05-0.47% (ensmean has ~10x less scatter than pooled).
- KEY PHYSICS: with the corrected f=5.58e-5 on the Milton box, ERA5 TRUTH itself is ~8% ageostrophic (deep tropics 5-20N + cyclostrophic storm core genuinely ageostrophic), and all baselines are comparable to truth. So the old "post-hoc + Atlas carry MORE scatter" was an artifact of the wrong f on global data. Applying f=5.58e-5 to GLOBAL data gives a pathological 33-41% (global is mostly extratropical where f~1e-4 is right) - CONFIRMS f=5.58e-5 is only valid on the Milton box.
- User decision (AskUserQuestion 2026-07-07): "Milton box + avg f, reframe" - regenerate C2 as Milton-box pooled-1M with f=5.58e-5, recompute %, reframe caption to "all baselines comparable to ERA5, dominated by tropical/cyclonic parts of the domain where geostrophy is not expected" (drop the post-hoc-worse claim). Also chose "Only S1" among word-adders.
- `tools/submit_geostrophic_milton.sh` edited: `multivariate: mean`->`pooled`, added `bivariate_max_samples: auto`, `coriolis_parameter 1.0e-4`->`5.58e-5`, + header comment. COMMITTED (code repo) this session. NB the header `python3` date-loop needs the venv (login-node python3 lacks `datetime.fromisoformat`) - `source .venv/bin/activate` before running.
- Job 2701897 (partition normal, a122, 1 node, 6h cap) submitted ~09:31, ~48 min elapsed at umtrac / 4-of-8 enspooled NPZs written to `$STORE/baselines/<b>/eval_milton/multivariate/*gradient_wind_speed_level500_enspooled.npz`.

## AFTER-BREAK CHECK (next session, to FINISH S3)
1. Confirm job 2701897 finished with 8/8 enspooled Milton geostrophic NPZs (State COMPLETED). Log: `$STORE/baseline_eval_logs/eval_geostrophic_milton_*.out`.
2. Recompute the ageostrophic % from the 8 new Milton-box POOLED NPZs with f=5.58e-5 (|wind-Ug|>15 m/s). Expect truth ~8%, forecasts comparable (pooled adds member scatter over the ensmean preview of 6-10%).
3. REFRAME the Fig C2 caption (main.tex, bivariate-geo caption ~L1370-1392, EXEMPT from word count): fix "f fixed at 1e-4" -> "f at 22.5N = 5.6e-5", the "3-6% vs 2-3%" numbers -> the recomputed pooled values + "all baselines comparable to ERA5, dominated by the tropical/cyclonic parts of the domain where geostrophy is not expected"; drop "synoptic region where the geostrophic relation is more cleanly defined". Caption still (correctly now) says pooled 1M.
4. Re-render: `source .venv/bin/activate && python tools/plot_bivariate.py` (now picks enspooled). Then `bash ai-models-ensembles-paper/sync_figures.sh` to vendor the new PDF into the paper repo `figures/`.
5. Recompile (tectonic) + re-verify word count still <7,500 (caption change is exempt, so should stay 7,495).
6. Commit the C2 caption + figure as the S3-finalization follow-up (paper repo).

## Commits this session (attended, to main, NOT pushed)
- code (MeteoSwiss): tools/submit_geostrophic_milton.sh (pooled+f fix).
- paper (sadamov): main.tex (S1/S2/S4/technical/physics/trims; geostrophic caption NOT yet reframed - pending job).
- UNTRACKED (left as-is): PAPER_PEER_REVIEW_AIES_2026-07-07.md.
- Word count 7,495 conservative (was 7,396 on 07-06; +~99 review fixes offset by ~95 trims). ams-wordcount.md baseline still 7,470 (07-03) - stale.

## S3 RESOLVED (later same session) - final f = 35N, not 22.5N
- Job 2701897 finished 8/8 (COMPLETED). Regenerated C2 as Milton-box POOLED-1M.
- Figure review at f=5.58e-5 (22.5N) showed the reference line sits ABOVE the |grad Z|-wind cloud (reads
  subgeostrophic - wrong for 500 hPa). Diagnosed: the cloud's density-weighted effective f is ~9e-5 (LS,
  dominated by the extratropical jet at ~38N) / ~6.5e-5 (median, ~30N). The 500 hPa geostrophic balance in the
  box is realised in its EXTRATROPICAL part, so the box-centre 22.5N is too low. USER chose 35N (8.37e-5),
  in-box, line runs through the cloud, addresses the reviewer's "43N is outside the box".
- f is DISPLAY-ONLY -> patched the coriolis_parameter scalar in the 8 cached enspooled NPZs to 8.37e-5 (no
  re-run). submit_geostrophic_milton.sh updated to 8.37e-5 for future runs.
- Final ageostrophic % (|wind-Ug|>15 m/s, f=8.37e-5): ERA5 truth 1.0%; aurora 0.6, FCN3 0.7, IFS-ENS 0.7,
  graphcast 0.8, AIFS-ENS 0.8, sfno 0.9, aifs 1.3, Atlas 1.4. All 0.6-1.4%, bracketing ERA5, NO post-hoc excess.
- Caption reframed (EXEMPT); figure re-rendered + synced; compiles; word count 7,475 conservative (25 under).
- COMMITS: code `aad4e30` (35N submit script + geostrophic PDF/PNG); paper `a93f5fd`
  (main.tex caption + figure). Both UNPUSHED, awaiting user push decision.
- S3 COMPLETE. Nothing left for a next session on S3.
