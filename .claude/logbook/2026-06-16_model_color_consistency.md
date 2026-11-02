# 2026-06-16 ATTENDED: unify per-model colours across all paper figures + new single source of truth

One-line: made every model render in one canonical colour across plots/tables/captions; introduced
`tools/model_colors.py` as the single source of truth, fixed the tier1b gradient bug, unified AIFS to
purple (pink kept only in the F9/Phase-5 contrast figures), and shifted the Milton F1 IBTrACS truth
track red->black.

## Why
User asked: "same model always gets the exact same colour throughout the full paper". Audit found the
8-model hex palette was already consistent across the LaTeX tables, headline CRPSS figure and most
Milton figures, BUT values were copy-pasted into ~5 scripts with no enforcement, plus two real
divergences.

## Decisions (user-confirmed via AskUserQuestion)
- AIFS: UNIFY to purple (#8E44AD) everywhere. Pink (#D81B60) survives ONLY in the weight-only vs
  weight+IC contrast figures (Milton F9 + Phase-5 SSR diagnostic), where telling the two AIFS runs
  apart on one axis is the whole point. Exposed as `AIFS_IC_CONTRAST`.
- Scope: shared module + rewire all hand-rolled scripts + regenerate, not a minimal in-place fix.
- F1 IBTrACS truth track red->black (user said "yes, shift") to avoid the red-on-red clash with the
  Atlas baseline colour (#C0392B). Pre-existing clash, not introduced this session.

## Canonical palette (mirrors figures/experiments_tables.tex \definecolor block, verified hex-for-hex)
aurora_encoder #E67E22 orange | graphcast_all #27AE60 green | sfno_modes10 #2980B9 blue |
aifs_perturbed #8E44AD purple | aifsens #8B5A2B brown | atlas #C0392B red | fcn3 #D4A017 gold |
ifs_ens #7F8C8D grey. Role styling: post-hoc solid/circle, trained-prob dashed/square, classical
dotted/diamond. `_ic` Milton variants alias to their base model's colour via `model_colors.canon()`.

## Files
- NEW `tools/model_colors.py` - MODEL_COLORS + LINESTYLE + MARKER + `_IC_ALIASES` + `AIFS_IC_CONTRAST`
  + helpers `canon/color_for/style_for/marker_for`. Docstring tells future-me to keep the LaTeX
  \definecolor block in experiments_tables.tex in sync.
- Rewired to import it: `tools/plot_headline_crpss_vs_lead_8way.py`, `tools/plot_headline_crpss_vs_lead.py`
  (7-way, not the paper figure but cleaned), `tools/plot_7way_spatial_mean_ssr.py`,
  `tools/milton/figures_milton.py`, `tools/plot_ssr_phase5_aifs.py`,
  `tools/milton/plot_aifs_wt_vs_ic_spread.py`. Import shim: tools/ scripts insert their own dir,
  tools/milton/ scripts insert the parent (tools/ is NOT a package, no __init__.py).
- THE real visual fix: `plot_7way_spatial_mean_ssr.py` (Figure 5 / tier1b) was using
  `plt.cm.Blues`/`plt.cm.Oranges` GRADIENTS, so Atlas showed orange and Aurora/SFNO were both blue
  shades. Now per-model palette: Atlas=red, SFNO=blue, etc. (numpy import dropped, now unused).

## Regenerated (host venv, no GPU, from cached CSVs)
- headline_crpss_vs_lead_8way.pdf (no colour change - no-break check)
- tier1b_7way_spatial_mean_ssr.pdf (palette fix, verified visually)
- milton_F1_track_spaghetti.{pdf,png} (AIFS purple + truth track black)
- milton_F5_verification_vs_lead.{pdf,png} (AIFS purple)
- milton_F9_aifs_wt_vs_ic_spread.pdf (pink kept; regen byte-changed only)
- F2/F3/F4 NOT regenerated: they don't colour lines by model (field colormaps + red IBTrACS markers).
  Only F1 (line 88) and F5 (lines 186,194) use BASELINE_COLORS; F2 uses steelblue for all members.

## Caption audit
All six colour-word refs in main.tex are field decorations (IBTrACS truth, Clausius-Clapeyron curve,
forbidden regions, ERA5 ceiling), none names a model colour I changed. No body text referenced the old
tier1b blue/orange gradient scheme. Updated F1 caption "red curve"->"black curve" (main.tex:1128) - that
paper edit was already committed+pushed by a concurrent/nightly session (HEAD 8fa8430, working tree
clean, up to date with origin), so nothing for this session to commit in the paper repo.

## State
- main repo: committed (code + 7 regenerated figure PDFs/PNGs). NOT pushed (awaiting user).
- paper repo: clean, caption change already in origin via concurrent actor.
- main.pdf NOT recompiled - figure PDFs on disk are updated; next tectonic compile picks them up.
