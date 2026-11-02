Fig. C5's caption was silently clipped off page 40; fixing it exposed that the default
spread-error CSV is missing IFS-ENS.

## The clip (user-reported)

`spread_error.pdf` was 709.9 x 774.8 pt (aspect 1.09, unusually tall for a one-column
float). At `width=0.95\linewidth` (444.6 pt) it rendered 485 pt tall; with its 250-word
caption the float exceeded `\textheight` (~670 pt of usable block) and TeX dropped the
overflow off the bottom of the physical page.

Two things made this hard to catch:
- **TeX emitted no `Overfull \vbox` warning.** A clean tectonic run is not evidence that
  nothing is clipped. Do not trust the log for this class of bug.
- The clipped text is still absent from the page, but `pdftotext`-style extraction of a
  *correct* page looks identical, so text presence is not the test either.

The reliable test is geometric. With PyMuPDF (`fitz`, available in the capstor miniforge
base; NOT in the project venv, which has `pypdf` instead):

```python
p = fitz.open("main.pdf")[pageno]
ymax = max(b[3] for b in p.get_text("blocks"))   # page_h 756; healthy pages end ~729.6
```

Page 40 measured `ymax = 755.0` on a 756 pt page and had no folio, while every neighbour
ended at 729.6 with its page number as the lowest element. That signature - text past
~740 and a missing folio - is the clip. Sweeping all 57 pages for `ymax > 740` found no
others. Worth re-running that sweep after any figure or float change.

## Why not the obvious fixes

- **Shrink in LaTeX** (the user's first instinct was to shorten the caption; mine was to
  scale the graphic): rejected. At 0.95 the figure's tick labels already render at only
  6.3-6.9 pt; the ~0.78 needed to fit would take them to 5.1-5.7 pt, under AMS legibility.
- **Shorten the caption**: would work but needs ~110 of its 250 words cut, i.e. real
  content. Captions are AMS word-count exempt so it costs nothing in budget, but it costs
  substance.
- **Rendered height depends only on the aspect ratio**, not on figsize, once the width is
  pinned to `\linewidth`. Scaling figsize uniformly changes nothing. Only a relayout helps.

## The fix

Transposed the panel grid in `tools/plot_spread_error.py`: leads index columns, variables
rows (`nrow, ncol = len(PANEL_VARS), len(args.leads)`, `axes[j][i]`, figsize
`2.95*ncol x 3.25*nrow`). Aspect 1.09 -> 0.745, rendered height 485 -> 331 pt. Because the
natural width also shrinks (624.6 pt), the LaTeX scale rises 0.626 -> 0.712 and the tick
labels render *larger* than before: 7.1/7.8 pt vs 6.3/6.9. Caption kept in full; only
"(rows)"/"(columns)" swapped in `main.tex`.

Narrower panels then made the x-axis labels collide, so the label dropped its variable
prefix ("ensemble spread [K]"); the variable is already in each panel title.

## The real find: the default CSV has only 7 of 8 baselines

The first regeneration silently dropped the IFS-ENS curve. Caught by diffing the legend
against the committed PDF, not by looking at the plot:

```
OLD 8 [... 'fcn3', 'ifs_ens']      NEW 7 [... 'fcn3']
```

`tools/plot_spread_error.py` defaulted to `$AIENS_SCRATCH/spread_error_binned.csv`, which
is both purged and incomplete. In `$STORE/analysis/`:

| file | rows | baselines |
|---|---|---|
| `spread_error_binned.csv` | 25,244 | 7 (no ifs_ens) |
| `spread_error_ifsens.csv` | 3,033 | ifs_ens only |
| `spread_error_all.csv` | 28,276 | **all 8** - this is what the published figure used |

Default repointed to `spread_error_all.csv`. **Lesson: after regenerating any figure,
diff its legend/series against the committed PDF before vendoring.** A figure that looks
plausible can be quietly missing a whole baseline, and the caption will still describe it
("dotted: the classical IFS-ENS reference"). Same class as the Milton double-filter bug -
a stale/partial input that produces a believable but wrong artifact.

Also confirmed the regenerated PDF embeds the same font as the original
(`BMQQDV+DejaVuSans`); the `findfont: cmss10 not found` warning is pre-existing, not a
change introduced by re-running the script.

## Commit-message gotcha recurred, now in the CODE repo

The stale `.git/COMMIT_EDITMSG` trap previously logged for the paper repo hit the main
code repo: `git commit -F -` left a trailing `r reproducibility.` line from a prior
message. Fix is the same - `rm .git/COMMIT_EDITMSG` first, then commit. Always
`git log -1 --format='%B' | tail` to verify after committing in either repo.

## State

- code `dc06f46` pushed to `MeteoSwiss/ai-models-ensembles` main
- paper `d0b859d` pushed to `sadamov/ai-models-ensembles` main
- 57 pages, tectonic clean, no body words changed (captions AMS-exempt)
