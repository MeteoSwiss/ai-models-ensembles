# Store inventory (generated)

Generated 2026-11-02 12:43 UTC by `python tools/inventory_store.py` from `/capstor/store/cscs/mch/s83/sadamov/ai-models-ensembles`.
Do not edit by hand: change [catalog.csv](catalog.csv) (descriptions) and rerun. Narrative documentation of every dataset: [../DATA.md](../DATA.md).

Columns: `zarrs` = number of `forecast.zarr` stores; `inits` = distinct init dates; `schema` = members x leads (first..last h, step) and grid of one representative store; `products` = evaluation outputs stored next to the forecasts.

## _archived/

Total size: 12 GB

| experiment | status | description | zarrs | inits | schema | size | products |
|---|---|---|---|---|---|---|---|
| `aifs_phase1_20230815_mag_0.003_truncated_forecast.zarr` | archive | see `_archived` | 1 | 0  | 10 mem x 0..0 h/0 h, 8 vars | 10 GB |  |

## ablation/

Total size: 8.8 TB

| experiment | status | description | zarrs | inits | schema | size | products |
|---|---|---|---|---|---|---|---|
| `allphases/aifs` | paper | see `ablation/allphases` | 0 | 0  |  |  | intercomparison |
| `allphases/aurora` | paper | see `ablation/allphases` | 0 | 0  |  |  | intercomparison |
| `allphases/graphcast_operational` | paper | see `ablation/allphases` | 0 | 0  |  |  | intercomparison |
| `allphases/sfno` | paper | see `ablation/allphases` | 0 | 0  |  |  | intercomparison |
| `phase1/aifs` | paper | see `ablation/phase1` | 0 | 0  |  |  | eval/{mag_0.003_layer_all,mag_0.01_layer_all,mag_0.03_layer_all}<br>intercomparison |
| `phase1/aifs/mag_0.003_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 30 vars, lev 500,850 |  |  |
| `phase1/aifs/mag_0.01_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 30 vars, lev 500,850 |  |  |
| `phase1/aifs/mag_0.03_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 30 vars, lev 500,850 |  |  |
| `phase1/aifs/mag_0_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 1 mem x 0..240 h/6 h, 30 vars, lev 500,850 |  |  |
| `phase1/aurora` | paper | see `ablation/phase1` | 0 | 0  |  |  | eval/{mag_0.001_layer_all,mag_0.003_layer_all,mag_0.01_layer_all,mag_0.03_layer_all,mag_0.1_layer_all,mag_0_layer_all}<br>intercomparison |
| `phase1/aurora/mag_0.001_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase1/aurora/mag_0.003_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase1/aurora/mag_0.01_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase1/aurora/mag_0.03_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase1/aurora/mag_0.1_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase1/aurora/mag_0_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 1 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase1/graphcast_operational` | paper | see `ablation/phase1` | 0 | 0  |  |  | eval/{mag_0.001_layer_all,mag_0.003_layer_all,mag_0.01_layer_all,mag_0.03_layer_all,mag_0.1_layer_all,mag_0_layer_all}<br>intercomparison |
| `phase1/graphcast_operational/mag_0.001_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase1/graphcast_operational/mag_0.003_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase1/graphcast_operational/mag_0.01_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase1/graphcast_operational/mag_0.03_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase1/graphcast_operational/mag_0.1_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase1/graphcast_operational/mag_0_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 1 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase1/sfno` | paper | see `ablation/phase1` | 0 | 0  |  |  | eval/{mag_0.001_layer_all,mag_0.003_layer_all,mag_0.01_layer_all,mag_0.03_layer_all,mag_0.1_layer_all,mag_0_layer_all}<br>intercomparison |
| `phase1/sfno/mag_0.001_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase1/sfno/mag_0.003_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase1/sfno/mag_0.01_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase1/sfno/mag_0.03_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase1/sfno/mag_0.1_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase1/sfno/mag_0_layer_all` | paper | see `ablation/phase1` | 4 | 4 20230515..20241115 | 1 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase2/aifs` | paper | see `ablation/phase2` | 0 | 0  |  |  | eval/{mag_0.011726_layer_processor,mag_0.026679_layer_encoder,mag_0.027500_layer_decoder}<br>intercomparison |
| `phase2/aifs/mag_0.011726_layer_processor` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 30 vars, lev 500,850 |  |  |
| `phase2/aifs/mag_0.026679_layer_encoder` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 30 vars, lev 500,850 |  |  |
| `phase2/aifs/mag_0.027500_layer_decoder` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 30 vars, lev 500,850 |  |  |
| `phase2/aurora` | paper | see `ablation/phase2` | 0 | 0  |  |  | eval/{mag_0.010412_layer_backbone,mag_0.044176_layer_encoder,mag_0.061549_layer_decoder}<br>intercomparison_vs_phase1 |
| `phase2/aurora/mag_0.010412_layer_backbone` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase2/aurora/mag_0.044176_layer_encoder` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase2/aurora/mag_0.061549_layer_decoder` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase2/graphcast_operational` | paper | see `ablation/phase2` | 0 | 0  |  |  | eval/{mag_0.011547_layer_m2m,mag_0.027080_layer_g2m,mag_0.029665_layer_m2g}<br>intercomparison |
| `phase2/graphcast_operational/mag_0.011547_layer_m2m` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase2/graphcast_operational/mag_0.027080_layer_g2m` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase2/graphcast_operational/mag_0.029665_layer_m2g` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase2/sfno` | paper | see `ablation/phase2` | 0 | 0  |  |  | eval/{mag_0.010428_layer_processor,mag_0.053852_layer_decoder,mag_0.053852_layer_encoder}<br>intercomparison |
| `phase2/sfno/mag_0.010428_layer_processor` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase2/sfno/mag_0.053852_layer_decoder` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase2/sfno/mag_0.053852_layer_encoder` | paper | see `ablation/phase2` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase2b/aurora` | paper | see `ablation/phase2b` | 0 | 0  |  |  | eval/{mag_0.025_layer_encoder,mag_0.040_layer_decoder,mag_0.060_layer_encoder,mag_0.085_layer_decoder}<br>intercomparison |
| `phase2b/aurora/mag_0.025_layer_encoder` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase2b/aurora/mag_0.040_layer_decoder` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase2b/aurora/mag_0.060_layer_encoder` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase2b/aurora/mag_0.085_layer_decoder` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase2b/graphcast_operational` | paper | see `ablation/phase2b` | 0 | 0  |  |  | eval/{mag_0.014_layer_g2m,mag_0.018_layer_m2g,mag_0.045_layer_g2m,mag_0.050_layer_m2g}<br>intercomparison |
| `phase2b/graphcast_operational/mag_0.014_layer_g2m` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase2b/graphcast_operational/mag_0.018_layer_m2g` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase2b/graphcast_operational/mag_0.045_layer_g2m` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase2b/graphcast_operational/mag_0.050_layer_m2g` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase2b/sfno` | paper | see `ablation/phase2b` | 0 | 0  |  |  | eval/{mag_0.035_layer_encoder,mag_0.075_layer_decoder,mag_0.080_layer_encoder,mag_0.100_layer_decoder}<br>intercomparison |
| `phase2b/sfno/mag_0.035_layer_encoder` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase2b/sfno/mag_0.075_layer_decoder` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase2b/sfno/mag_0.080_layer_encoder` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase2b/sfno/mag_0.100_layer_decoder` | paper | see `ablation/phase2b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase2c/aurora` | superseded | see `ablation/phase2c` | 0 | 0  |  |  | eval/{mag_0.025_layer_encoder} |
| `phase2c/graphcast_operational` | superseded | see `ablation/phase2c` | 0 | 0  |  |  | eval/{mag_0.014_layer_g2m} |
| `phase2c/sfno` | superseded | see `ablation/phase2c` | 0 | 0  |  |  | eval/{mag_0.035_layer_encoder} |
| `phase3/aurora` | paper | see `ablation/phase3` | 0 | 0  |  |  | eval/{mag_0.025900_layer_unet_bottom,mag_0.05_layer_unet_bottom,mag_0.15_layer_unet_bottom,mag_0.40_layer_coarse_encoder,mag_0.40_layer_unet_bottom,mag_0.80_layer_unet_bottom,mag_1.50_layer_unet_bottom}<br>intercomparison |
| `phase3/aurora/mag_0.025900_layer_unet_bottom` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/aurora/mag_0.05_layer_unet_bottom` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/aurora/mag_0.15_layer_unet_bottom` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/aurora/mag_0.40_layer_unet_bottom` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/aurora/mag_0.80_layer_unet_bottom` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/aurora/mag_1.50_layer_unet_bottom` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational` | paper | see `ablation/phase3` | 0 | 0  |  |  | eval/{gcsigma_0.01_gcnodes42,gcsigma_0.03_gcnodes42,gcsigma_0.10_gcnodes42,gcsigma_0.10_gcnodes42_frozen,gcsigma_0.312_gcnodes42,gcsigma_0.312_gcnodes42_frozen,gcsigma_0.60_gcnodes42,gcsigma_0.60_gcnodes42_frozen,gcsigma_1.00_gcnodes42,gcsigma_1.0_gcnodes42_frozen,gcsigma_1.80_gcnodes42}<br>intercomparison<br>intercomparison_fresh_pre_frozen_2026-06-21 |
| `phase3/graphcast_operational/gcsigma_0.01_gcnodes42` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_0.03_gcnodes42` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_0.10_gcnodes42` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_0.10_gcnodes42_frozen` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_0.312_gcnodes42` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_0.312_gcnodes42_frozen` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_0.60_gcnodes42` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_0.60_gcnodes42_frozen` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_1.00_gcnodes42` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_1.0_gcnodes42_frozen` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3/graphcast_operational/gcsigma_1.80_gcnodes42` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3/sfno` | paper | see `ablation/phase3` | 0 | 0  |  |  | eval/{mag_0.049261_modes10,mag_0.10_modes10,mag_0.25_modes10,mag_0.25_modes10_skip1,mag_0.35_modes10_refresh20_0.35,mag_0.35_modes10_refresh20_0.35_skip1,mag_0.50_modes10_refresh10_0.50,mag_0.57_modes10_refresh20_0.57,mag_0.60_modes10,mag_0.60_modes10_fresh,mag_0.71_modes10_refresh20_0.71,mag_0.80_modes10_refresh10_0.80,mag_1.00_modes10_refresh10_1.00,mag_1.93_modes10,mag_1.93_modes10_fresh}<br>intercomparison |
| `phase3/sfno/mag_0.049261_modes10` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.10_modes10` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.25_modes10` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.25_modes10_skip1` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.35_modes10_refresh20_0.35` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.35_modes10_refresh20_0.35_skip1` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.50_modes10_refresh10_0.50` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.57_modes10_refresh20_0.57` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.60_modes10` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.60_modes10_fresh` | paper | see `ablation/phase3` | 3 | 3 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.71_modes10_refresh20_0.71` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_0.80_modes10_refresh10_0.80` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_1.00_modes10_refresh10_1.00` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3/sfno/mag_1.93_modes10_fresh` | paper | see `ablation/phase3` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3b/aurora` | paper | see `ablation/phase3b` | 0 | 0  |  |  | eval/{mag_0.015_layer_enc_012,mag_0.017_layer_enc_12}<br>intercomparison |
| `phase3b/aurora/mag_0.015_layer_enc_012` | paper | see `ablation/phase3b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3b/aurora/mag_0.017_layer_enc_12` | paper | see `ablation/phase3b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase3b/graphcast_operational` | paper | see `ablation/phase3b` | 0 | 0  |  |  | eval/{gcsigma_0.080_gcnodes642,gcsigma_0.080_gcnodes642_frozen,gcsigma_0.159_gcnodes162,gcsigma_0.159_gcnodes162_frozen}<br>intercomparison<br>intercomparison_fresh_pre_frozen_2026-06-21 |
| `phase3b/graphcast_operational/gcsigma_0.080_gcnodes642` | paper | see `ablation/phase3b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3b/graphcast_operational/gcsigma_0.080_gcnodes642_frozen` | paper | see `ablation/phase3b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3b/graphcast_operational/gcsigma_0.159_gcnodes162` | paper | see `ablation/phase3b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3b/graphcast_operational/gcsigma_0.159_gcnodes162_frozen` | paper | see `ablation/phase3b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 11 vars, lev 500,850 |  |  |
| `phase3b/sfno` | paper | see `ablation/phase3b` | 0 | 0  |  |  | eval/{mag_0.025_modes40,mag_0.035_modes20}<br>intercomparison |
| `phase3b/sfno/mag_0.025_modes40` | paper | see `ablation/phase3b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase3b/sfno/mag_0.035_modes20` | paper | see `ablation/phase3b` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 13 vars, lev 500,850 |  |  |
| `phase6_family/graphcast_operational` | paper | see `ablation/phase6_family` | 0 | 0  |  |  | intercomparison |
| `phase6_family/sfno` | paper | see `ablation/phase6_family` | 0 | 0  |  |  | intercomparison |
| `phase6_family_v2/aifs/phase6_aifs_refresh20_sigma0.0275` | paper | see `ablation/phase6_family_v2` | 4 | 4 20230515_0000..20241115_0000 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase6_family_v2/aifs/phase6_aifs_refresh20_sigma0.0385` | paper | see `ablation/phase6_family_v2` | 4 | 4 20230515_0000..20241115_0000 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase6_family_v2/aurora/phase6_aurora_refresh20_sigma0.025` | paper | see `ablation/phase6_family_v2` | 4 | 4 20230515_0000..20241115_0000 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase6_family_v2/aurora/phase6_aurora_refresh20_sigma0.035` | paper | see `ablation/phase6_family_v2` | 4 | 4 20230515_0000..20241115_0000 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `phase6_family_v2/sfno` | paper | see `ablation/phase6_family_v2` | 0 | 0  |  |  | intercomparison |
| `phase6_smoke/aifs` | smoke test | see `ablation/phase6_smoke` | 1 | 1 20230102_0000..20230102_0000 | 2 mem x 0..24 h/6 h, 3 vars, lev 500,850 |  |  |
| `phase6_smoke/aurora` | smoke test | see `ablation/phase6_smoke` | 1 | 1 20230102_0000..20230102_0000 | 2 mem x 0..24 h/6 h, 3 vars, lev 500,850 |  |  |

## ablation_seed43/

Total size: 319 GB

| experiment | status | description | zarrs | inits | schema | size | products |
|---|---|---|---|---|---|---|---|
| `aifs/mag_0.027500_layer_decoder` | paper | see `ablation_seed43` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `aurora/mag_0.025_layer_encoder` | paper | see `ablation_seed43` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `graphcast_operational/mag_0.01_layer_all` | paper | see `ablation_seed43` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |
| `sfno/mag_0.25_modes10` | paper | see `ablation_seed43` | 4 | 4 20230515..20241115 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 |  |  |

## baselines/

Total size: 82.0 TB

| experiment | status | description | zarrs | inits | schema | size | products |
|---|---|---|---|---|---|---|---|
| `aifs_all_s010` | paper | Rival AIFS all layers sigma=0.01 (Phase 1 cell). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members. Tab. rival_validation. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `aifs_det` | talk only | Unperturbed deterministic AIFS (sigma=0, 1 member), 112 inits, 0-360 h. Grey line in figures/talk/headline_crpss_vs_lead_8way_aifs_det (single-member CRPS = MAE). | 112 | 112 20230102_0000..20241008_1200 | 1 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 332 GB | eval/{deterministic} |
| `aifs_ic_only` | paper | IC-only arm: IFS-ENS perturbed analyses as ICs, weight noise off. 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h. Separates IC from weight spread (Tab. ic_decomp). | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `aifs_ic_only_truncated_2026-08-28` | failed run | Single init of aifs_ic_only written truncated during the 2026-08-28 capstor incident (plus its leftover _e2s_work member zarrs). Re-run into aifs_ic_only; kept as evidence. Not used. | 1 | 1 20230704_0000..20230704_0000 | 10 mem x 0..0 h/0 h, 7 vars | 47 GB |  |
| `aifs_p6c` | superseded | Phase 6c refresh-every-20, AIFS decoder@0.0275, old buggy seeding. Superseded by aifs_p6c_reseed. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.2 TB | eval/{deterministic,energy_spectra,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>spatial_mean_ssr |
| `aifs_p6c_reseed` | paper | Phase 6c refresh-every-20, corrected seeding, AIFS decoder@0.0275, 112 inits. Fig. refresh_frozen_vs_refresh_ssr. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.2 TB | eval/{probabilistic}<br>spatial_mean_ssr |
| `aifs_perturbed` | paper | SPW AIFS (single 1.1): decoder layer group, sigma=0.0275 (Phase 2 pick). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h, CDS ICs. Headline baseline. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB | eval/{deterministic,energy_spectra,energy_spectra_member0_pre_2026-09-17,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>eval_milton/{multivariate}<br>per_pixel_ssr<br>spatial_mean_ssr |
| `aifs_perturbed_ic` | paper | Phase 5 weight+IC arm: SPW decoder@0.0275 plus IFS-ENS perturbed analyses as per-member ICs (ifs_analysis_perturbed_ic.zarr). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-240 h. IC decomposition table and Milton. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.3 TB | eval/{deterministic,energy_spectra,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>per_pixel_ssr<br>spatial_mean_ssr |
| `aifs_perturbed_ic_SMOKE` | smoke test | 1-init, 2-member, 24 h smoke test of the weight+IC wiring (2026-06-09). Safe to ignore. | 1 | 1 20241004_0000..20241004_0000 | 2 mem x 0..24 h/6 h, 9 vars, lev 500,850 | 457 MB |  |
| `aifsens` | paper | AIFS-ENS 1.0 (ECMWF, trained probabilistic). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h. ICs from CDS. Headline 8-way baseline. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.4 TB | energy_variogram_score<br>eval/{deterministic,energy_spectra,energy_spectra_member0_pre_2026-09-17,ets,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>eval_milton/{multivariate}<br>spatial_mean_ssr |
| `atlas` | paper | Atlas (NVIDIA, trained probabilistic, stochastic interpolant). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h, ICs from ARCO-ERA5. Headline baseline. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.4 TB | energy_variogram_score<br>eval/{deterministic,energy_spectra,energy_spectra_member0_pre_2026-09-17,ets,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>eval_milton/{multivariate}<br>spatial_mean_ssr |
| `atmllm` | external, unused | SwissClim eval of AtmLLM forecasts produced and stored by a colleague outside this store (path in eval_main_config.yaml). Not in the paper. | 0 | 0  |  | 23 GB | eval/{deterministic,energy_spectra,fss,maps,multivariate,probabilistic,ssim,wd_kde} |
| `aurora_enc_s044` | paper | Rival (runner-up ablation cell) Aurora encoder sigma=0.044176 on the held-out production grid. 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h. Tab. rival_validation. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `aurora_encoder` | paper | SPW Aurora: encoder layer group, sigma=0.025 (Phase 2b pick). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h, ARCO ICs. Headline baseline. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.4 TB | energy_variogram_score<br>eval/{deterministic,energy_spectra,energy_spectra_member0_pre_2026-09-17,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>eval_milton/{multivariate}<br>per_pixel_ssr<br>spatial_mean_ssr |
| `aurora_encoder_ic` | paper | Phase 5 weight+IC arm: SPW encoder@0.025 plus IFS-ENS perturbed analyses as per-member ICs (ifs_analysis_perturbed_ic.zarr). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-240 h. IC decomposition table and Milton. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.3 TB | eval/{deterministic,energy_spectra,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>per_pixel_ssr<br>spatial_mean_ssr |
| `aurora_encoder_ic_SMOKE` | smoke test | 1-init, 2-member, 24 h smoke test of the weight+IC wiring (2026-06-09). Safe to ignore. | 1 | 1 20241004_0000..20241004_0000 | 2 mem x 0..24 h/6 h, 9 vars, lev 500,850 | 457 MB |  |
| `aurora_ic_only` | paper | IC-only arm: IFS-ENS perturbed analyses as ICs, weight noise off. 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h. Separates IC from weight spread (Tab. ic_decomp). | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `aurora_p6c` | superseded | Phase 6c weight-noise refresh every 20 steps, Aurora encoder@0.025, 112 inits, 0-240 h. Generated with the old additive per-step seeding bug (fixed 2026-07-10). Superseded by aurora_p6c_reseed. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.3 TB | eval/{deterministic,energy_spectra,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>spatial_mean_ssr |
| `aurora_p6c_reseed` | paper | Phase 6c refresh-every-20 with the corrected _mix_seed seeding, Aurora encoder@0.025, 112 inits, 0-240 h. Fig. refresh_frozen_vs_refresh_ssr. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.2 TB | eval/{probabilistic}<br>spatial_mean_ssr |
| `esfm` | external, dropped | ESFM (Ozdemir et al. 2026) intermediate checkpoint: ES/VS and spatial-mean SSR only; forecasts live outside this store. Briefly an 8th baseline, removed (CRPSS@240 h -0.127, not fully trained). | 0 | 0  |  | 0 MB | energy_variogram_score<br>spatial_mean_ssr |
| `fcn3` | paper | FourCastNet 3 (NVIDIA, trained probabilistic). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h, ICs from ARCO-ERA5. Headline baseline. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.4 TB | energy_variogram_score<br>eval/{deterministic,energy_spectra,energy_spectra_member0_pre_2026-09-17,ets,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>eval_milton/{multivariate}<br>spatial_mean_ssr |
| `graphcast_all` | paper | SPW GraphCast (operational): all layers, sigma=0.01 (Phase 1 pick). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h, ARCO ICs. Headline baseline. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.4 TB | energy_variogram_score<br>eval/{deterministic,energy_spectra,energy_spectra_member0_pre_2026-09-17,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>eval_milton/{multivariate}<br>per_pixel_ssr<br>spatial_mean_ssr |
| `graphcast_all_ic` | paper | Phase 5 weight+IC arm: SPW all@0.01 plus IFS-ENS perturbed analyses as per-member ICs (ifs_analysis_perturbed_ic.zarr). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-240 h. IC decomposition table and Milton. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.3 TB | eval/{deterministic,energy_spectra,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>per_pixel_ssr<br>spatial_mean_ssr |
| `graphcast_all_ic_SMOKE` | smoke test | 1-init, 2-member, 24 h smoke test of the weight+IC wiring (2026-06-09). Safe to ignore. | 1 | 1 20241004_0000..20241004_0000 | 2 mem x 0..24 h/6 h, 9 vars, lev 500,850 | 457 MB |  |
| `graphcast_g2m` | paper | Rival GraphCast g2m sigma=0.014 (the argmin-CRPSS ablation cell). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members. Tab. rival_validation. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `graphcast_ic_only` | paper | IC-only arm: IFS-ENS perturbed analyses as ICs, weight noise off. 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h. Separates IC from weight spread (Tab. ic_decomp). | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `graphcast_m2g` | paper | Rival GraphCast m2g sigma=0.029665. 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members. Tab. rival_validation. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `graphcast_p6c_reseed` | talk only | GraphCast weight refresh every 20 steps, all@0.01, corrected seeding, 112 inits (2026-10-07). Talk/revision material, not in the submitted paper: 240 h spatial-mean SSR 3.20 -> 2.97. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.2 TB | eval/{probabilistic}<br>spatial_mean_ssr |
| `ifs_ens` | paper | IFS-ENS reference: eval outputs only. Forecasts are read from the WB2-derived IFS-ENS zarr (/capstor/store/cscs/swissai/ab016/IFS/ifs_ens.zarr), 10 stratified members 0,5,...,45. ~20% whole-slab NaN gaps (msl ~30%). | 0 | 0  |  | 23 GB | eval/{deterministic,energy_spectra,energy_spectra_member0_pre_2026-09-17,ets,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>eval_milton/{multivariate}<br>spatial_mean_ssr |
| `sfno_enc_s035` | paper | Rival SFNO encoder sigma=0.035. 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members. Tab. rival_validation. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `sfno_enc_s054` | paper | Rival SFNO encoder sigma=0.053852. 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members. Tab. rival_validation. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `sfno_ic_only` | paper | IC-only arm: IFS-ENS perturbed analyses as ICs, weight noise off. 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h. Separates IC from weight spread (Tab. ic_decomp). | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.3 TB |  |
| `sfno_modes10` | paper | SPW SFNO: coarse spherical-harmonic modes l<10 only, sigma=0.25 (Phase 3 pick). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-360 h, ARCO ICs. Headline baseline. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..360 h/6 h, 9 vars, lev 500,850 | 3.4 TB | energy_variogram_score<br>eval/{deterministic,energy_spectra,energy_spectra_member0_pre_2026-09-17,ets,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>eval_milton/{multivariate}<br>per_pixel_ssr<br>spatial_mean_ssr |
| `sfno_modes10_ic` | paper | Phase 5 weight+IC arm: SPW modes10@0.25 plus IFS-ENS perturbed analyses as per-member ICs (ifs_analysis_perturbed_ic.zarr). 112 production inits (days 2-8 of Jan/Apr/Jul/Oct 2023+2024, 00/12 UTC), 10 members, 0-240 h. IC decomposition table and Milton. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.3 TB | eval/{deterministic,energy_spectra,fss,maps,multivariate,probabilistic,ssim,wd_kde}<br>per_pixel_ssr<br>spatial_mean_ssr |
| `sfno_modes10_ic_SMOKE` | smoke test | 1-init, 2-member, 24 h smoke test of the weight+IC wiring (2026-06-09). Safe to ignore. | 1 | 1 20241004_0000..20241004_0000 | 2 mem x 0..24 h/6 h, 9 vars, lev 500,850 | 456 MB |  |
| `sfno_p6c` | superseded | Phase 6c refresh-every-20, SFNO modes10 sigma=0.35, old buggy seeding. Superseded by sfno_p6c_reseed. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.2 TB | spatial_mean_ssr |
| `sfno_p6c_reseed` | paper | Phase 6c refresh-every-20, corrected seeding, SFNO modes10 sigma=0.35, 112 inits. Fig. refresh_frozen_vs_refresh_ssr. | 112 | 112 20230102_0000..20241008_1200 | 10 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 2.2 TB | eval/{probabilistic}<br>spatial_mean_ssr |

## baselines_m50/

Total size: 699 GB

| experiment | status | description | zarrs | inits | schema | size | products |
|---|---|---|---|---|---|---|---|
| `aifs_perturbed` | diagnostic | 50-member single-init run (2023-10-05 00 UTC, 0-240 h) for the member-distribution (Gaussianity) diagnostics, 2026-10-16. Not in the paper; figures/ens_anom/*_m50*. | 1 | 1 20231005_0000..20231005_0000 | 50 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 99 GB |  |
| `aifsens` | diagnostic | 50-member single-init run (2023-10-05 00 UTC, 0-240 h) for the member-distribution (Gaussianity) diagnostics, 2026-10-16. Not in the paper; figures/ens_anom/*_m50*. | 1 | 1 20231005_0000..20231005_0000 | 50 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 100 GB |  |
| `atlas` | diagnostic | 50-member single-init run (2023-10-05 00 UTC, 0-240 h) for the member-distribution (Gaussianity) diagnostics, 2026-10-16. Not in the paper; figures/ens_anom/*_m50*. | 1 | 1 20231005_0000..20231005_0000 | 50 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 100 GB |  |
| `aurora_encoder` | diagnostic | 50-member single-init run (2023-10-05 00 UTC, 0-240 h) for the member-distribution (Gaussianity) diagnostics, 2026-10-16. Not in the paper; figures/ens_anom/*_m50*. | 1 | 1 20231005_0000..20231005_0000 | 50 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 100 GB |  |
| `fcn3` | diagnostic | 50-member single-init run (2023-10-05 00 UTC, 0-240 h) for the member-distribution (Gaussianity) diagnostics, 2026-10-16. Not in the paper; figures/ens_anom/*_m50*. | 1 | 1 20231005_0000..20231005_0000 | 50 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 100 GB |  |
| `graphcast_all` | diagnostic | 50-member single-init run (2023-10-05 00 UTC, 0-240 h) for the member-distribution (Gaussianity) diagnostics, 2026-10-16. Not in the paper; figures/ens_anom/*_m50*. | 1 | 1 20231005_0000..20231005_0000 | 50 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 100 GB |  |
| `sfno_modes10` | diagnostic | 50-member single-init run (2023-10-05 00 UTC, 0-240 h) for the member-distribution (Gaussianity) diagnostics, 2026-10-16. Not in the paper; figures/ens_anom/*_m50*. | 1 | 1 20231005_0000..20231005_0000 | 50 mem x 0..240 h/6 h, 9 vars, lev 500,850 | 100 GB |  |

## scratch/

Total size: 1 GB

| experiment | status | description | zarrs | inits | schema | size | products |
|---|---|---|---|---|---|---|---|
| `sfno_fresh_debug2.zarr` | debug | see `scratch` | 1 | 0  | 2 mem x 0..24 h/6 h, 13 vars, lev 500,850 | 606 MB |  |
| `sfno_fresh_debug3.zarr` | debug | see `scratch` | 1 | 0  | 2 mem x 0..24 h/6 h, 13 vars, lev 500,850 | 606 MB |  |

## Other directories

| path | files | size | description |
|---|---|---|---|
| `analysis/ens_anom_gaussianity` | 205 | 8 GB | Member-anomaly NPZ caches for the Gaussianity diagnostics (tools/compute_ens_anomalies.py, tools/compute_point_ensemble.py); 551 MB per model. figures/ens_anom/. |
| `analysis/ic_decomp_12ch.csv` | 1 | 0 MB | Per-channel IC vs weight spread decomposition (tools/ic_weight_decomposition.py); condensed into tools/data/ic_decomp_4bb.csv. |
| `analysis/ifs_ens_spatial_ssr_audit` | 19 | 13 MB | Audit of IFS-ENS spatial-mean SSR on the WB2 NaN slabs (2026-10-07). |
| `analysis/milton_case_study` | 4922 | 58 MB | Milton TC tracks: IBTrACS netCDF + per-(baseline, init) TempestExtremes tracks rebuilt 2026-10-07 for Fig. F1. The master/verification CSVs for F2/F5/F9 were scratch-purged; see docs/REPRODUCING.md. |
| `analysis/milton_wind_frames` | 6 | 149 MB | Extracted Milton-box 10 m wind/MSL netCDFs for the talk animation (tools/milton/plot_wind_frames.py stage 1). |
| `analysis/milton_z500_plume` | 25 | 384 MB | Extracted z500 landfall-box series for figures/slides/milton_z500_plume_* (tools/milton/plot_z500_plume.py). |
| `analysis/per_init_crps_ifsens.csv` | 1 | 0 MB | Per-init CRPS of IFS-ENS (tools/compute_per_init_crps.py); block bootstrap and common-sample checks. |
| `analysis/per_init_crps_production_v2.csv` | 1 | 8 MB | Per-init CRPS of all production baselines (tools/compute_per_init_crps.py); input of the rival-validation table and the bootstrap CIs. |
| `analysis/rank_histograms` | 8 | 0 MB | Rank (Talagrand) histogram NPZs at 240 h (tools/submit_rank_histograms.sh). Input of Fig. rank_histograms_240h. |
| `analysis/spread_error_all.csv` | 1 | 2 MB | spread_error_binned.csv + spread_error_ifsens.csv concatenated; the only input of Fig. spread_error (tools/plot_spread_error.py). |
| `analysis/spread_error_binned.csv` | 1 | 2 MB | Binned spread-error pairs of the 7 non-IFS production baselines (default output of tools/spread_error_binned.py via tools/submit_spread_error.sh). |
| `analysis/spread_error_ifsens.csv` | 1 | 0 MB | Same binning for IFS-ENS (run separately because of its NaN slabs). |
| `analysis/table_metrics_fixedscale` | 37 | 0 MB | Fixed-scale (1990-2019 climatological channel scale) ES/VS/SIGK per baseline and lead (tools/submit_table_metrics_fixedscale.sh). Input of Tab. C1 and the calibration table. |
| `analysis/table_metrics_ic_decomp` | 96 | 0 MB | ES/VS/SIGK of the IC-decomposition arms (tools/submit_esvs_sigk_ic_decomp.sh). |
| `diagnostics/sfno_phase6_ssr` | 15 | 0 MB | Spatial-mean SSR CSVs of the SFNO Phase 3/6 frozen/fresh/refresh variants (sigma_N <= 1.93 refresh clause). |
| `e2s_cache_backup/README_DO_NOT_DELETE.md` | 1 | 0 MB | Explains the CDS cache below. |
| `e2s_cache_backup/aifs-ens-1.0` | 7 | 959 MB | AIFS-ENS 1.0 checkpoint package from hf://ecmwf/aifs-ens-1.0. |
| `e2s_cache_backup/aifs-single-1.1` | 3 | 1 GB | AIFS single 1.1 checkpoint package from hf://ecmwf/aifs-single-1.1. |
| `e2s_cache_backup/aifs-single-1.1_interpolation_matrix` | 2 | 7 MB | AIFS grid interpolation matrix (earth2studio download). |
| `e2s_cache_backup/aifs-single-1.1_inverse_interpolation_matrix` | 2 | 13 MB | AIFS inverse interpolation matrix (earth2studio download). |
| `e2s_cache_backup/cds` | 5393 | 46 GB | Persistent earth2studio CDS GRIB cache (~5400 files, 43 GB) for all AIFS/AIFS-ENS ICs. Rebuilding needs a CDS key and 24-48 h of throttled downloads. |
| `e2s_cache_backup/ifs-opendata` | 4 | 2 MB | Small ECMWF open-data cache used by AIFS. |
| `earth2studio_cache/sfno` | 7 | 7 GB | SFNO checkpoint package (ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0) used for the weight audit. |

## Slurm logs

| path | files |
|---|---|
| `ablation_logs` | 2507 |
| `baseline_logs` | 2067 |
| `baseline_eval_logs` | 623 |
| `logs` | 76 |
| `model_test_logs` | 19 |

## Variables seen across all forecasts

`100m_u_component_of_wind`, `100m_v_component_of_wind`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `2m_dewpoint_temperature`, `2m_temperature`, `convective_precipitation_6hr`, `geopotential`, `high_cloud_cover`, `low_cloud_cover`, `mean_sea_level_pressure`, `medium_cloud_cover`, `runoff`, `skin_temperature`, `snowfall`, `soil_temperature_level_1`, `soil_temperature_level_2`, `specific_humidity`, `surface_pressure`, `surface_solar_radiation_downwards_6hr`, `surface_thermal_radiation_downwards_6hr`, `swvl1`, `swvl2`, `temperature`, `total_cloud_cover`, `total_column_water`, `total_column_water_vapour`, `total_precipitation_6hr`, `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity`
