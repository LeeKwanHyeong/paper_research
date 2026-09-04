# Separate-Key Retrieval 5090 Seed42 e300 Launch

Status: **running**, not a completed performance result.
Repository/branch: local `paper_research/master`. Only orchestration/audit changes.

## Current Baseline

- Frozen model: `c5fc6bc6b271f52324ea3090a6edf15c265b9251`.
- CUDA/full e1: `db7d70684b81f83380acfe08307854546e9ea75f`, 29 CUDA tests and
  Taxi/Instacart full e1 passed. See the adjacent smoke evidence directory.
- e300 execution revision: `2fc49bcb3aef88fd6b14cbad89825ee8ee4bc018`.
- Local model/contract/orchestration tests: 58 passed in 8.74s, `local_pytest.xml`.
- Training files (models, loaders, losses, core runner) are byte-identical to
  the CUDA/e1 snapshot. e300 changes only budget and execution role.

## Execution

- Server: SSH alias `5090`, tmux `hard_lmm_key_value_e300_0904`.
- Snapshot: `/home/leekwanhyeong/workspace/paper_research_key_value_e300_2fc49bc_5090`.
- Artifact: `<snapshot>/search_artifacts/hard_lmm_key_value_seed42_e300_5090_20260904`.
- Launcher log: `<snapshot>/screening_launcher.log`.
- Source manifest SHA256: `9996f470856fd03401a28ce568f84642a0d032dc3e2ba6308c2d885b40c4527f`.
- Package SHA256: `be346938afa1593b6655f4c97e9273660b34050be06135cdd15c907415c1ae8c`.
- Python: `/opt/miniconda3/envs/ai_env/bin/python`, version3.12.13,
  torch2.11.0+cu130, CUDA13.0, polars1.39.3.
- Only candidate `titantpp_key_value_static_memory`, Taxi then Instacart,
  seed42, fresh initialization; no e1 checkpoint reuse.
- Fixed maximum300/minimum40/patience40, batch128, lr0.001, clip1, direct
  log-MSE, lambda_tail0, legacy RMTPP and validation joint selection.

Started 2026-09-04 **21:29:24 KST**. Single initial check at **21:30:11 KST**:
Taxi completed epoch5, best4, 0/2 finalized runs, tmux alive, CUDA PID3680137,
GPU2606MiB/37%, GDM inactive. Source and complete smoke gate passed; preflight
reported no recent kernel errors. No additional polling was done this turn.

## Timing and Monitoring

Hourly heartbeat: `hard-lmm-key-value-5090-e300`, ACTIVE. Reports meaningful
progress, ETA change, completion/failure or required action; quiet otherwise.
No automatic resume/retry, service modification or unrelated process termination.

Initial ETA is approximate, not a guaranteed deadline. Full e1 wall costs were
Taxi10.65s and Instacart160.14s, including setup/final validation. Early stopping
near the first legal epoch could finish around **Sep4 23:30 KST**; both runs
reaching300 at those costs imply **Sep5 11:30-12:00 KST**. Screening speed and
new best epochs will change this estimate.

## Remaining Order

1. Continue the two approved runs and hourly checks without changing the model.
2. Sync final artifacts without deletion, audit full histories/strata/finite
   checkpoints and recompute the original Hard-LMM comparison.
3. Keep body MAE >=5% improvement, RMSE/tail regression <=2%, and Time NLL
   increase <=0.01. Do not relax gates. Record seed42 and missing Instacart
   weighted-control limitations; no general superiority claim.
4. Update Notion, commit only final results on local `paper_research/master`,
   and pause the heartbeat. Additional datasets/seeds/held-out require approval.

Notion: https://app.notion.com/p/3d1bbe40561381c2a67afd4e5da20c67
