# TitanTPP-MAC: 5080-only seed62 launch

## Scope and Current State

Initial observation: 2026-08-31T09:18:42.787578+09:00. Updated at 09:20:01 KST.
This is a launch record, not a final performance result.

- The user's 5080-only request retains the previous seed62 assignment.
- 5090 was not contacted or modified in this stage. Other seeds are deferred.
- RAF full e1 and artifact validation passed: 25,779 training targets / 202 batches and 6,690 validation targets; finite metrics and exact checkpoint prediction/memory replay.
- Taxi full e1 started at 09:19 KST. Context gates are 1/3 complete; e300 runs remain 0/4.
- Remaining context gates: Taxi and Intermittent.
- Fresh seed62 e300 runs start only after all three context gates complete: Instacart, RAF, Taxi, Intermittent.
- Completed Instacart seeds42/52/62 e1 audits are the existing baseline, not repeated work.
- E1 checkpoints, failed runs, and historical unbounded-memory runs are not reused.

## Verification and Provenance

- Repository/branch: paper_research/master.
- Frozen training source: c4dbf856c32e6502acc660ffac23c3e2f68e5375.
- Orchestration/contract commit: f814e03f8a9f842937bd0de3a8dc4d78a96d61e7.
- This stage changed no model, loss or training implementation.
- Source/data/contract verification passed on 5080, including all 24 frozen training file hashes and four data/split checksums.
- Focused controller/validator tests: 29 passed locally and 29 passed on 5080.
- Tests cover unauthorized host/seed/policy, missing context, busy/wrong GPU, low VRAM, GDM/desktop activity, overwrite refusal, finite/full-epoch header checks, and failed-child status without launching the next run.
- Initial 5080 kernel query since 05:30 KST returned successfully with no Xid/OOM/watchdog matches. This is a bounded observation, not a guarantee.
- Each training and artifact-validation stage is a separate process. No automatic retries or checkpoint resume.
- Inner/outer gradient clipping: 1/1. This inner-learning policy differs from the historical unbounded MAC recurrence.
- Raw launch status and process evidence: launch_record.json.
- First successful full-context artifact gate and Taxi transition: first_context_gate.json.
- Source and orchestration checksum evidence: source_manifest.json.

## Execution and Monitoring

- Snapshot: /home/leekwanhyeong/workspace/paper_research_mac_stable_c4dbf85_f814e03_5080
- Tmux: titantpp_mac_stable_seed62_5080_0831
- Artifact: /home/leekwanhyeong/workspace/paper_research_mac_stable_c4dbf85_f814e03_5080/search_artifacts/count_aware_titantpp_mac_stable_seed62_e300_20260831_5080
- Launch log: /home/leekwanhyeong/workspace/paper_research_mac_stable_c4dbf85_f814e03_5080/search_artifacts/titantpp_mac_stable_seed62_5080_0831.launch.log
- Hourly thread automation: titantpp-mac-5080-seed62 (5080 only).
- Notion page: https://www.notion.so/3ccbbe405613817ea095f605888d39ac
- Notion launch update was fetched again and verified.

The controller reports context gates separately (0-3) from validated e300 runs (0-4).
The monitor must not label an e1 artifact as a completed e300 run.
A failure is preserved and reported, not silently restarted. The monitor is removed after
complete or confirmed failed status. No services on 5090 are within this scope.

## Remaining Work

1. Complete all three full-context e1 artifact gates on 5080.
2. Run and validate the four fresh seed62 e300 jobs under the frozen contract.
3. Synchronize without --delete, audit source/log/contract/history/scale metrics/checkpoints,
   and update the existing Notion page with seed62 results only.
4. Defer other seeds and 5090 until the user sets that scope.

No reliable finish time is established yet. Use actual post-compile epoch timings rather
than extrapolating a first compile or a different GPU. Full e1 qualification does not
guarantee later epochs cannot fail. Held-out test remains locked.
