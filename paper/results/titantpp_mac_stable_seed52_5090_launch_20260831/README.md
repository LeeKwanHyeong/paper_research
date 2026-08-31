# TitanTPP-MAC: 5090 Seed52 Companion Launch

## Current State

Observed at 2026-08-31T09:47:42.425973+09:00. This is a launch record, not an e300 result.

- User authorized adding 5090. It retains seed52 on four datasets.
- 5080 seed62 snapshot and processes were not modified.
- Seed42 is not added. The new inner-gradient policy requires fresh matched
  seed42 results before eventual three-seed reporting.
- RAF full e1 and artifact gate passed: 25,779 train targets / 202 batches,
  6,690 validation targets, all metrics finite, exact checkpoint prediction
  and observed-history memory replay.
- Taxi full e1 started at 09:46:52 KST; first batch entered.
- Context qualification: 1/3 complete. Validated e300 runs: 0/4.
- After Taxi and Intermittent gates pass, four fresh seed52 e300 runs execute
  serially: Instacart, RAF, Taxi, Intermittent.

## Contract and Verification

Repository/branch: paper_research/master.
Training source: c4dbf856c32e6502acc660ffac23c3e2f68e5375.
Orchestration commit: dd9790ef190784bf0084246efb5f8614016a5411.
Contract: paper/contracts/count_aware_titantpp_mac_stable_seed52_5090_v1.json.

No model, loss, optimizer or training implementation changed. The 5090-specific
entrypoint differs from the frozen 5080 controller only by server/seed routing.
Parity tests compare all training arguments and source/data/policy fields.

- Local focused tests: 53 passed.
- 5090 focused tests: 53 passed.
- All 24 frozen training hashes and four data/split digests match.
- Completed Instacart seed52 full-e1 proof was verified and not repeated.
- New prelaunch file inspection caught two missing RAF files in the shared
  default directory before any training started. Exact-checksum local parquet
  and split manifest were copied without deletion into the new snapshot.
  No existing data was altered and no partial training was restarted.
- Current GPU: 8,790 MiB, 33% utilization; GDM inactive; one training CUDA process.
- The tmux libtinfo version warning did not prevent session/process creation.
- Kernel journal access is restricted on 5090. No claim of Xid absence is made.
- Port8011 was already not listening before this stage. CPU services were left
  alone. No service start/stop/redeployment or GDM changes were performed.

## Runtime and Monitoring

Snapshot: /home/leekwanhyeong/workspace/paper_research_mac_stable_c4dbf85_dd9790e_5090
Artifact: /home/leekwanhyeong/workspace/paper_research_mac_stable_c4dbf85_dd9790e_5090/search_artifacts/count_aware_titantpp_mac_stable_seed52_e300_20260831_5090
Tmux: titantpp_mac_stable_seed52_5090_0831
Python: /opt/miniconda3/envs/ai_env/bin/python
Tmux binary: /opt/miniconda3/envs/ai_env/bin/tmux
Launcher log: /home/leekwanhyeong/workspace/paper_research_mac_stable_c4dbf85_dd9790e_5090/search_artifacts/titantpp_mac_stable_seed52_5090_0831.launch.log

The app allows one active heartbeat per thread. A separate 5090 heartbeat was
rejected, so the existing titantpp-mac-5080-seed62 heartbeat was updated to
monitor BOTH servers every hour. No workaround cron or second thread was made.
Server-specific progress remains separate. No automatic restart or service
changes are allowed. A failure on one side does not stop healthy training on
the other. Notion was updated and re-read:
https://www.notion.so/3ccbbe405613817ea095f605888d39ac

## Remaining Work

1. Complete the remaining Taxi/Intermittent full-context gates on5090.
2. Run/validate four fresh seed52 e300 jobs alongside the unchanged5080 seed62 jobs.
3. Sync without --delete, audit artifacts, and record each seed's results.
4. Keep seed42 and held-out evaluation outside this stage.

ETA requires observed per-epoch timing after compile. Full-e1 success is not
a guarantee against all later failures. Historical unbounded-memory results
must not be pooled with the clipped-inner-gradient results.
