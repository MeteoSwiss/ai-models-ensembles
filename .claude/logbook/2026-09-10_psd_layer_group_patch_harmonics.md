Summary: Aurora's zonal-PSD spikes are patch-grid harmonics driven by the DECODER group, they exist in the unperturbed model, and none of the four production SPW configs changes the spectrum at all.

Context: a colleague asked, off the back of the Phase-1 PSD panels
(`figures/psd_lead006h_*_phase1.pdf`, commit `d9ffc1c`), why Aurora shows spectral peaks and
the others do not, and hypothesised (a) noise entering "deeper" in the architecture smooths
out, (b) the peaks are patch-size related, (c) AIFS differs because of N320 gridding, (d) it
shows in z500 but not 10m wind. Answering it properly needed the Phase-2/3 spectra, which did
not exist. Tooling generalised and run: `tools/compute_psd_lead.py` now takes `--phases`
(keys are `<phase>:<run>`, Phase-6 refresh/frozen/fresh/skip variants filtered out),
`tools/plot_psd_lead.py` takes `--tag`, `--models`, `--harmonics`, several `--npz` files and
`--runs <model>=<run>`. Jobs 3351467 (phase2+2b) and 3351468 (phase3+3b), ~6 min each on
`normal`, output `tools/data/psd_phase{2,3}_lead006h.npz`.

## Aurora's peaks are patch-grid harmonics, and they predate the perturbation

Spikes sit at zonal wavenumber m = 180, 360, 540 (222/111/74 km), i.e. harmonics of a 2 deg
period with the strongest at 1 deg = `patch_size=4` on the 0.25 deg grid, the first
PatchMerging3D giving the 2 deg token grid. At **sigma=0** the m=360 line is already +3.24 dex
above ERA5 in z500 (+2.2 dex above the local background). SPW amplifies a pre-existing Aurora
artefact, it does not create one.

## Phase 2: the target module decides, and it is not "depth"

Variance-matched layer groups, z500 log10(forecast/ERA5) at +6 h:

| run | m=180 | m=360 | broadband m100-340 |
|---|---|---|---|
| unperturbed | +0.04 | +3.24 | -0.16 |
| backbone 0.0104 | +0.53 | +3.24 | -0.16 |
| encoder 0.025 (production) | +0.09 | +3.24 | -0.15 |
| encoder 0.044 | +0.25 | +3.24 | +0.09 |
| encoder 0.060 | +0.66 | +3.24 | +0.56 |
| decoder 0.040 | +0.15 | +5.39 | -0.07 |
| decoder 0.0615 | +0.24 | +5.75 | +0.00 |
| decoder 0.085 | +0.35 | +6.03 | +0.09 |

Only the decoder (the patch reconstruction back onto the 0.25 deg grid) excites the line
spectrum, up to +2.8 dex over the unperturbed line at a *lower* sigma than the encoder runs.
The encoder instead lifts the broadband mesoscale floor. Same pattern in 10m u
(decoder 0.040: +1.84 at m=360 vs +0.01 unperturbed). So the colleague's "encoder acts more
directly on the grid" intuition has the right mechanism but the wrong end of the network.

AIFS is the mirror image: only the **encoder** group moves the spectrum (broadband -0.15 ->
+0.84, m=180 -0.15 -> +1.07), while processor and decoder are flat to 2 dp. Plausible cause,
not verified: the AIFS encoder group includes `node_attributes`, i.e. per-node learned fields,
so noise there is spatially structured rather than a pure weight jitter.

GraphCast and SFNO show no line spectrum for any group; they cannot, having no regular lat-lon
lattice (icosahedral mesh / SH basis). N320 has a latitude-dependent point count per circle,
so a fixed mesh-stride artefact does not land on one zonal wavenumber and the latitude average
in the zonal PSD smears it.

## Phase 3: coarse-scale targeting is spectrally clean, except in the U-Net

GraphCast `gcnodes42` up to sigma=1.8 and SFNO `modes10` up to sigma=0.6: zero detected peaks,
spectrum indistinguishable from unperturbed. Aurora `unet_bottom` stays clean to sigma=0.15,
then grows a full ladder at m = 90, 180, 270, 360, 450, 540, 630 from sigma=0.4 upward, i.e.
harmonics of a **4 deg** fundamental = the coarsest U-Net token grid (1 -> 2 -> 4 deg over two
PatchMerging stages). The harmonic fundamental therefore tracks the token grid at the
injection depth, which is the cleanest evidence that these lines are architectural.

## All four production configs are spectrally silent

`figures/psd_lead006h_geopotential_500hPa_production.pdf`: aurora_encoder 0.025,
graphcast_all 0.01, sfno_modes10 0.25, aifs_perturbed 0.0275 all lie on their unperturbed
curve at +6 h, Aurora's patch line included (+3.24 both). The calibration was done on CRPS and
spread, never on spectra, and it happened to land on the spectrally benign target for both
Aurora (encoder, not decoder) and AIFS (decoder, not encoder).

## Why z500 and not 10m wind

Not absence, contrast. At sigma=0.03 (Phase 1, all weights) u10 still has +1.51 dex excess at
m=360 and +1.47 at m=540. ERA5 z500 has a spectral slope of -3.63 over m=30-300 with 2.4e-5 of
its variance above m=100; u10 has -2.59 with 7.9e-3, ~300x more small-scale background for the
same artefact to hide in. Ratio-to-truth panels systematically flatter steep-spectrum fields.

## Overdone noise does break the fields (banded LSD, z500, +120 h)

aurora upper-mesoscale LSD 0.39 / 0.30 / 0.82 at sigma = 0 / 0.03 / 0.1: a moderate
perturbation slightly *improves* the mesoscale spectrum (it counteracts the deterministic
blurring) before over-injection wrecks it. AIFS breaks an order of magnitude earlier in sigma
(0.24 -> 1.25 at sigma=0.01), consistent with its smaller calibrated sigma.

Not done: nothing checked beyond +6 h, and no map-space check that the m=360 line is a visible
checkerboard rather than a low-amplitude ripple.
