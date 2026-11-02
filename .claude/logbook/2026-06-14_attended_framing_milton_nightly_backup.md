# 2026-06-14 (attended) - headline weight-only, Milton red thread, nightly 4am backup

One-line: cleared the nightly review's "Next up" list with the user - headline made all-weight-only, Milton
gained a wt-vs-weight+IC dispersion figure, abstract/funding/Zotero closed, and the nightly got a 4am backup.

## Decisions
- HEADLINE = WEIGHT-ONLY. Dropped the aifs_perturbed_ic row from the headline table + vs-lead figure so all
  four post-hoc baselines are pure weight perturbation. Rationale: clean attribution of the novel
  weight-perturbation contribution, preserves the 360h story, and the over-dispersion "gap" is the narrative
  hook that Phase 5 (IC) and Phase 6 (refresh) resolve. IC results are NOT lost - they live in Phase 5 and
  the new Milton Fig. 9. Generators: tools/headline_8way_table.py + tools/plot_headline_crpss_vs_lead_8way.py
  (removed aifs_perturbed_ic from the model lists / load calls). The prose "three deterministic backbones"
  (aurora/graphcast/sfno) is intentional even though the table now has four post-hoc rows.
- MILTON RED THREAD. Added Fig. 9 (tools/milton/plot_aifs_wt_vs_ic_spread.py): two cascade maps (wt-only |
  weight+IC, both already on disk) over a new cone-width + MSLP-spread panel. Verified from
  milton_master_tracks.csv: ensemble MEAN is near-identical wt-only vs weight+IC, but IC ~doubles the
  short-lead track cone (48->96 km @0-24h; 314->376 @120-144h) and widens intensity spread at every lead
  (1.1->1.9 hPa @0-24h); detection 64->67%, mean position error 236 vs 262 km. This is why the case study
  reports the Phase 5 IC ensembles - same dispersion mechanism as the headline SSR gap, on one named storm.
- (paper-process detail kept in the local notes)

## Commits (all on main, NOT pushed)
- code: 23a792a (cost table), 226ef25 (headline weight-only), 4082888 (Milton F9 + generator).
- paper: 164d5d8 (GraphCast reframe), 2fbf81d (CRPSS scoping), 8400665 (Milton Fig. 9),
  27d8956 (abstract <250), 0ef78c8 (funding).
- (paper-process detail kept in the local notes)
