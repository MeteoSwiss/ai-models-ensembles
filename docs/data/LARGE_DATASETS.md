# Datasets of 1 TB or more

Generated 2026-11-02 13:05 UTC by `python tools/inventory_store.py --sizes ... --external ...` (sizes from `du`, disk usage). Status and descriptions come from [catalog.csv](catalog.csv) and `EXTERNAL` in the script; details per dataset in [STORE_INVENTORY.md](STORE_INVENTORY.md) and [../DATA.md](../DATA.md).

Verdicts: `keep` = needed to reproduce the paper (or the only copy of something expensive); `optional` = not in the paper, delete once nobody needs it for the revision or talks; `delete` = safe to remove (superseded, failed, duplicate). Nothing has been deleted. The files stay owned by the original account after it is closed: deleting needs the owner, or CSCS / the project PI.

The paper's data statement does not promise any forecast output (they are regenerable from public inputs and the code), so `keep` means: keep while the review runs, because reviewers may ask for new evaluations, which then take hours instead of days of GPU time. After acceptance the `keep` rows can go too; regenerating one 112-init production experiment costs about 15-30 GH200 node-hours of inference (warm caches) plus up to 8 x 12 h of evaluation (docs/REPRODUCING.md). Exception: the IFS-ENS perturbed analyses came from ECMWF MARS and need MARS access to rebuild.

Total by verdict: keep 103.8 TB, optional 3.0 TB.

| size (TB) | path | owner | status | verdict | what |
|---|---|---|---|---|---|
| 21.3 | `/capstor/store/cscs/swissai/ab016/IFS/ifs_ens.zarr` | another `ab016` member | paper | keep: data of the paper | IFS-ENS reference forecasts (WB2 archive, 50 members, 10 used); code default since 2026-11-02 |
| 3.4 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/fcn3` | sadamov:a122 | paper | keep: data of the paper | FourCastNet 3 (NVIDIA, trained probabilistic) |
| 3.4 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/atlas` | sadamov:a122 | paper | keep: data of the paper | Atlas (NVIDIA, trained probabilistic, stochastic interpolant) |
| 3.4 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/graphcast_all` | sadamov:a122 | paper | keep: data of the paper | SPW GraphCast (operational): all layers, sigma=0.01 (Phase 1 pick) |
| 3.4 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifsens` | sadamov:a122 | paper | keep: data of the paper | AIFS-ENS 1.0 (ECMWF, trained probabilistic) |
| 3.4 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/sfno_modes10` | sadamov:a122 | paper | keep: data of the paper | SPW SFNO: coarse spherical-harmonic modes l<10 only, sigma=0.25 (Phase 3 pick) |
| 3.4 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aurora_encoder` | sadamov:a122 | paper | keep: data of the paper | SPW Aurora: encoder layer group, sigma=0.025 (Phase 2b pick) |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifs_perturbed` | sadamov:a122 | paper | keep: data of the paper | SPW AIFS (single 1.1): decoder layer group, sigma=0.0275 (Phase 2 pick) |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/graphcast_g2m` | sadamov:a122 | paper | keep: data of the paper | Rival GraphCast g2m sigma=0.014 (the argmin-CRPSS ablation cell) |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/graphcast_m2g` | sadamov:a122 | paper | keep: data of the paper | Rival GraphCast m2g sigma=0.029665 |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/graphcast_ic_only` | sadamov:a122 | paper | keep: data of the paper | IC-only arm: IFS-ENS perturbed analyses as ICs, weight noise off |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/sfno_enc_s054` | sadamov:a122 | paper | keep: data of the paper | Rival SFNO encoder sigma=0.053852 |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/sfno_enc_s035` | sadamov:a122 | paper | keep: data of the paper | Rival SFNO encoder sigma=0.035 |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aurora_enc_s044` | sadamov:a122 | paper | keep: data of the paper | Rival (runner-up ablation cell) Aurora encoder sigma=0.044176 on the held-out production grid |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aurora_ic_only` | sadamov:a122 | paper | keep: data of the paper | IC-only arm: IFS-ENS perturbed analyses as ICs, weight noise off |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/sfno_ic_only` | sadamov:a122 | paper | keep: data of the paper | IC-only arm: IFS-ENS perturbed analyses as ICs, weight noise off |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifs_all_s010` | sadamov:a122 | paper | keep: data of the paper | Rival AIFS all layers sigma=0.01 (Phase 1 cell) |
| 3.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifs_ic_only` | sadamov:a122 | paper | keep: data of the paper | IC-only arm: IFS-ENS perturbed analyses as ICs, weight noise off |
| 3.0 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation/phase3` | sadamov:a122 | paper | keep: data of the paper | Phase 3 coarse-scale-only perturbation: SFNO spectral modes l<10 (mag_<sigma>_modes10[_refresh*/_fresh/_skip1]), Aurora U-Net bottleneck (layer_unet_bottom), GraphCast activation noise on coarse mesh nodes (gcsigma_<s>_gcnodes42[_frozen]) |
| 2.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/graphcast_all_ic` | sadamov:a122 | paper | keep: data of the paper | Phase 5 weight+IC arm: SPW all@0.01 plus IFS-ENS perturbed analyses as per-member ICs (ifs_analysis_perturbed_ic.zarr) |
| 2.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aurora_encoder_ic` | sadamov:a122 | paper | keep: data of the paper | Phase 5 weight+IC arm: SPW encoder@0.025 plus IFS-ENS perturbed analyses as per-member ICs (ifs_analysis_perturbed_ic.zarr) |
| 2.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/sfno_modes10_ic` | sadamov:a122 | paper | keep: data of the paper | Phase 5 weight+IC arm: SPW modes10@0.25 plus IFS-ENS perturbed analyses as per-member ICs (ifs_analysis_perturbed_ic.zarr) |
| 2.3 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifs_perturbed_ic` | sadamov:a122 | paper | keep: data of the paper | Phase 5 weight+IC arm: SPW decoder@0.0275 plus IFS-ENS perturbed analyses as per-member ICs (ifs_analysis_perturbed_ic.zarr) |
| 2.2 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/graphcast_p6c_reseed` | sadamov:a122 | talk only | optional: talk material, not in the paper | GraphCast weight refresh every 20 steps, all@0.01, corrected seeding, 112 inits (2026-10-07) |
| 2.2 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aurora_p6c_reseed` | sadamov:a122 | paper | keep: data of the paper | Phase 6c refresh-every-20 with the corrected _mix_seed seeding, Aurora encoder@0.025, 112 inits, 0-240 h |
| 2.2 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/sfno_p6c_reseed` | sadamov:a122 | paper | keep: data of the paper | Phase 6c refresh-every-20, corrected seeding, SFNO modes10 sigma=0.35, 112 inits |
| 2.2 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/baselines/aifs_p6c_reseed` | sadamov:a122 | paper | keep: data of the paper | Phase 6c refresh-every-20, corrected seeding, AIFS decoder@0.0275, 112 inits |
| 2.2 | `/capstor/store/cscs/swissai/ab016/IFS/ifs_analysis_perturbed_ic.zarr` | another `ab016` member | paper | keep: data of the paper | IFS-ENS perturbed analyses, per-member ICs of the Phase 5 (_ic, _ic_only) runs |
| 2.0 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation/phase1` | sadamov:a122 | paper | keep: data of the paper | Phase 1 magnitude sweep: multiplicative noise on all weights, sigma in {0, 0.001, 0.003, 0.01, 0.03, 0.1} (AIFS: 0, 0.003, 0.01, 0.03) |
| 1.4 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation/phase2` | sadamov:a122 | paper | keep: data of the paper | Phase 2 layer-group sweep: one architectural group at sigma_full*sqrt(N_total/N_group) (Aurora backbone/encoder/decoder, GraphCast g2m/m2g/m2m, SFNO encoder/processor/decoder, AIFS encoder/processor/decoder) |
| 1.2 | `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles/ablation/phase2b` | sadamov:a122 | paper | keep: data of the paper | Phase 2b fine sigma refinement on the top two Phase 2 groups per model (Aurora, GraphCast, SFNO) |
| 0.8 | `/capstor/store/cscs/swissai/ab016/IFS/ifs_control.zarr` | another `ab016` member | unused | optional: not read by any script | IFS control analyses; prepared for an IC experiment, not read by any current script |
