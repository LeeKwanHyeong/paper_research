# TitanTPP v0.7 validation-freeze artifact manifest

- Status: `validation_freeze_complete_held_out_locked`
- Frozen model: original mark-free Hard-LMM / Count-aware TitanTPP-T0
- Main comparison: Count-aware RMTPP, THP, and TitanTPP
- Datasets: Intermittent-5000, Taxi, and Instacart
- Seeds: 42, 52, and 62
- Qualified validation rows: 27
- Held-out test evaluated: false
- Quantity-interface superiority claim: excluded because the existing ablation fails the v0.7 matched contract
- Time metric: artifact `time_nll` is reported as clamped time loss; executed caps are intercept 300 and `w * delta_t` 10
- Training source revisions: Intermittent `044add1f`, Taxi `6a01aea9`, Instacart `28293c43`

## Source identity

| Path | SHA-256 | Bytes |
| --- | --- | ---: |
| `paper/contracts/titantpp_v0_7_final_model_claim_contract_v1.md` | `f4bf1829d340dc7f43cf5356d6830d9fe58bb5e7ecc3ec7b6dec5dcb3364551d` | 13,625 |
| `paper/scripts/build_v0_7_paper_artifacts.py` | `f2dfaa2a27d6d3ec32f621e059380a4c8534c62fcdf78c7d72d39dc07e181824` | 38,601 |
| `paper/scripts/generate_v0_7_mark_free_figures.py` | `aa6d746c785fd5f37c0501e6c18554771453e96beb1bb1ef6d526542d5591196` | 18,504 |
| `paper/scripts/audit_v0_7_time_head_revisions.py` | `98f38a8a1812944b109f089f774526c5102de5683bb3a154244cc8f84f1f3cc5` | 15,535 |
| `paper/scripts/verify_v0_7_paper_artifacts.py` | `9bdd68ba256714471c8cb75ebb09d87f9254c49a415c60302e41bd367aefd9da` | 48,857 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/build_freeze.py` | `5bdce92866dfe2f266080c089a089e1c96aedd54f9d47af4b072b70c6b0f6650` | 55,457 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/independent_verify.py` | `63320a5e540e00bb5baede975750c3c69b39ff386f5b0d48d8614b9c01883083` | 50,352 |
| `paper/results/count_aware_tpp_backbone_control_20260812/source_5080/run_summaries.csv` | `5610d48d5c3ab0fc8b852dfb1e43dd1711824cdab43ae0d253add5c0f56617c2` | 11,040 |
| `paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_summary.csv` | `4ebb349b42c16c38dc4d267538c38fece4c11b4ffffda2a2fad852233862bd49` | 5,345 |
| `paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_seed_metrics.csv` | `948e0ec26531a72667396d2411264386caa0a750ef28943f4eaf488c36978c3a` | 9,818 |
| `paper/results/count_aware_tpp_backbone_control_20260812/source_5080/history_summary.csv` | `1af643ccdf110795a696f00404eabf4d2ab171b4aff79c4b54f123638fad7b0c` | 3,437 |
| `paper/results/count_aware_tpp_backbone_control_20260812/qualification_briefing.md` | `282370e21e17e1c34b4c3db2ae4904c844de4cc14bbb16513e5dfa257b85ba57` | 2,127 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/validation_seed_metrics.csv` | `4bb3b00157611ea76753b99906a72d54ddae2a850cefbcccb4948db5db3f1fda` | 7,997 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_strata_aggregate.csv` | `d9ea289d47867b3bf58a06f84754cc9bfb45fe3e1bd32aabeab738167b3068e0` | 8,598 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_strata_seed_metrics.csv` | `ccda8656fd2803748ed5b4d8659e579960e07b8b7ba3d31092eaddc137ed44db` | 16,351 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_interface_audit.json` | `1eb2cea9974089992c435244a7cbc15459a8cccc1f3e0e999fd9cdf593de90c6` | 5,310 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/dataset_identity_audit.json` | `7578ebef41fd99e2fe639a071d5f0deaac2ba6d00e00bd8a113b486d09d2b609` | 3,709 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/source_manifest.json` | `dc86efce930330127de096cc00d051ef2a17f83d59a7bcf7905222e53fab3965` | 8,155 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/independent_verification.json` | `4628c00102de203b708551b966a02def9b52f5b7754f39b9215db07c42a4ac48` | 22,059 |
| `paper/results/titantpp_v0_7_validation_freeze_20260905/time_head_revision_audit.json` | `7346048625519a2dea69dd0d26e7d403c52af5bd5b795e61374b02b0a0e0611a` | 10,432 |
| `paper/results/hard_lmm_key_value_screening_5090_20260904/README.md` | `b3b83ef98a7d7653dfecf28dcc957f33592a973f8dce5d39340cced09e25b46b` | 5,086 |
| `paper/results/hard_lmm_elapsed_age_screening_5090_20260905/README.md` | `d2c1ba1e0278b162634419f0475f8d29b374e2ee35596a1ea83af8db6ad9d456` | 6,676 |
| `paper/results/hard_lmm_instacart_raw_history_20260905/README.md` | `45ffb794aaafa52df2092b6501c57ad68d8b6ff8e9af4095ff6e0a34b275f911` | 11,965 |
| `sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet` | `85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f` | 2,146,184 |
| `sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet` | `b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46` | 544,394 |
| `sample_data/insta_market_basket/instacart_marked_target_with_split.parquet` | `06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2` | 31,701,203 |
| `sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json` | `393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04` | 4,876 |
| `sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json` | `4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85` | 3,331 |
| `sample_data/insta_market_basket/instacart_marked_target_split_manifest.json` | `6c6cdd41f847878fbb405b73dfa038fbb7a88ad53df6843b0cc9e64531a8b71d` | 4,603 |

## Generated artifacts

| Path | SHA-256 | Bytes |
| --- | --- | ---: |
| `paper/tables/T1_v0_7_dataset_statistics.csv` | `c1685ee851d70e1893725152db8feebcc8da93a22105193f64dab3deb2027ea8` | 1,046 |
| `paper/tables/T2_v0_7_model_training_contract.csv` | `4ed533cd64c7df43f861aacc128c13da75479dbafbb9d57c59cab849f05e0e80` | 3,047 |
| `paper/tables/T3_v0_7_backbone_validation.csv` | `c27fa119eb387af7533c28c7dadaac90a1e5c46a8f7bb1b3da8f5a851084a3e9` | 2,406 |
| `paper/tables/T4_v0_7_paired_titan_deltas.csv` | `d34ca2479c0d21cac9700cf0a9ed90bdc273ddf8b6805ec2af12544c7c7a27c7` | 4,301 |
| `paper/tables/T5_v0_7_quantity_strata_validation.csv` | `45cedf8019108773c8a2de7cda7ae8dfe747fab41d7bad2eff64b192829f1d72` | 7,653 |
| `paper/tables/T1_v0_7_dataset_statistics.md` | `6cac946cc7d6e682f5a0d6649efe25b1eb7e9e4692dbf6c53f895724b2eb3971` | 852 |
| `paper/tables/T2_v0_7_model_training_contract.md` | `3051cc20e329b12ac4aea435535681f5a43ec5999cfebebb9971367356b43c27` | 1,803 |
| `paper/tables/T3_v0_7_backbone_validation.md` | `d33187f516fb50c15e248d94f39962c5656b7e6ab3d515a7aa3c1356921e054f` | 1,403 |
| `paper/tables/T4_v0_7_paired_titan_deltas.md` | `60537b1f405bd5087a7c31d893a3177324c55a20d015649218231320282fc8be` | 771 |
| `paper/tables/T5_v0_7_quantity_strata_validation.md` | `04e9015a0d206e3801771e3d329094312706b6a5037a8b192cb62169729447b6` | 4,781 |
| `paper/figures/F1_v0_7_mark_free_architecture.png` | `0ed6578c9ff7b35ba7b96eacb24d4ff4d2362e28e95f4df9c9c819356a0a2f9d` | 442,744 |
| `paper/figures/F1_v0_7_mark_free_architecture.pdf` | `2fd9dcc22d15627260d152c5ad182fb9f84853d6db08281d882678cf9b8af435` | 46,318 |
| `paper/figures/F1_v0_7_mark_free_architecture.svg` | `d295b903451024ae0025cdf224afadae80c5f3bc81e5b50d755559d9b626abc5` | 64,007 |
| `paper/figures/source_data/F2_v0_7_dataset_validation_errors.csv` | `13820f84dca2435ce5ac2fd4f06fafdc4f0e96af12954769e4b5e1d004e2f54a` | 1,094 |
| `paper/figures/F2_v0_7_dataset_validation_errors.png` | `4ce1611b3c91731ec676e51b650648dc927a45653956df3ccd7d42c9b0a772bd` | 123,955 |
| `paper/figures/F2_v0_7_dataset_validation_errors.pdf` | `fb42e0aa134e1dbc853f8e33aaa8bb9aa1ce2549992d22ac461c7d2894f3a760` | 24,775 |
| `paper/figures/F2_v0_7_dataset_validation_errors.svg` | `c677f32d38fa20ca8ead206cff914f4464a45f9116dd3031334e65f3c37e9ff4` | 70,218 |

## Publication documents

| Path | SHA-256 | Bytes |
| --- | --- | ---: |
| `paper/titantpp_short_paper_draft_v0_7_manuscript.md` | `ab0e577691f87332824eb4ce07631d3961da607f9a5ff30b19b00c659830811c` | 21,203 |
| `README.md` | `04af65a76e2029e331c72d5952eda473b967dd3e845306999f2a63898765537e` | 21,221 |

This manifest freezes validation evidence only. The one-time held-out evaluation remains a separate approved step after the checkpoint list and execution contract are fixed.
