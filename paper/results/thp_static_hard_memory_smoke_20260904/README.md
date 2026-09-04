# THP + Static Hard Memory: 5080 CUDA / Full e1

## Approved Scope

- Repository / branch: `paper_research/master`; implementation baseline `781a753`.
- Candidate: `thp_static_hard_memory`, not an adopted model or a Titans-MAC reproduction.
- User approved source sync, checksum verification, CUDA contracts, then full Taxi and RAF e1 only.
- No model, objective, time head, optimizer, selector or frozen acceptance changes.
- No baseline retraining, e300 screening, held-out evaluation, dependency installation or service change.
- Source is exported from a commit into an isolated 5080 snapshot, without `--delete`.
- Each CUDA test / dataset child runs in a separate Python process. GDM, competing CUDA jobs,
  free VRAM >= 12000 MiB, desktop processes and recent kernel errors are checked before each child.
- Existing output directories fail closed; ordinary failures atomically mark the owned artifact failed.
- The e1 launcher has no screening mode. Separate approval and fresh output are required for e300.

## Local Verification

- Python 3.12.10 / Torch 2.7.1, CPU; installed packages unchanged.
- 219 tests passed, zero failures or skips (`local_tests.xml`, 2.51 seconds).
- Includes the full prior 209-test suite and 10 new orchestration/reference-export checks.
- The auditor was exercised against a synthetic actual-runner artifact, including strict checkpoint
  replay, finite checkpoint/optimizer tensors, stratum reconciliation, scope and launch drift rejection.
- Reference-factory snapshot export is pinned to SHA-256
  `0937ccf3219f5c75f5d8ec01cee6a5716e2472cdf546734029f021f9bfc8c588`;
  both unchanged-legacy-factory CUDA tests remain mandatory without a Git database.
- Packaging includes only committed allowlisted source and a pinned historical factory export,
  with a per-file source manifest; no dataset or secret is packaged.

## Server Verification

Completed and independently re-audited locally. Actual source revision:
`093629b96455568f702fc651146664eb176ae13c`.

- Snapshot: `/home/leekwanhyeong/workspace/paper_research_thp_static_093629b_smoke_0904` on alias `5080`.
- Artifact: `search_artifacts/count_aware_thp_static_memory_cuda_e1_20260904`.
- tmux: `thp_static_memory_cuda_e1_0904` (exited after completion).
- Execution: 2026-09-04 19:49:53 to 19:50:14 KST, approximately 21 seconds for CUDA tests,
  full e1 runs, baseline verification and audits. Not a long-epoch duration estimate.
- CUDA contracts: **35 passed**, zero failures/skips, 2.22 seconds.
- Source: 464 file checksums matched both the deployed snapshot and the committed Git objects.
- Artifact: 32 file checksums matched the final remote inventory after local sync without `--delete`.
- Runtime: Python 3.12.13, Torch 2.11.0+cu130, CUDA 13.0, Polars 1.39.3, pytest 9.0.3,
  RTX 5080 driver 595.84. No packages were changed.
- Postflight: no CUDA process, GPU utilization 0%, 10 MiB used / 15,825 MiB free,
  GDM inactive and no recent Xid/OOM. Services were not changed.

| Dataset | Full train targets | Full validation targets | e1 run elapsed | MAE | RMSE | Time NLL | Joint objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Taxi | 38,393 | 8,268 | 10.640 s | 59.732959 | 229.053097 | 1.367272 | 1.593199 |
| RAF | 25,779 | 6,690 | 2.815 s | 9.430184 | 38.125877 | 3.298434 | 3.882918 |

These are **e1 feasibility values, not screening results or superiority claims**. Both runs used
fresh initialization and the complete fixed train/validation split. Historical THP/Hard-LMM
baseline files were revalidated but neither rerun nor used as initialization.

The audit followed source manifest, launch log, launch contract/summary, held-out absence,
history, quantity/history CSV reconciliation and checkpoint validation. All metrics and
checkpoint/optimizer tensors were finite. Best/last state digests and strict-load CPU synthetic
predictions matched exactly. The CUDA model tests separately exercised save/restore and
optimizer-next-update replay. No held-out evaluation/artifact occurred. The e1 runner does not
generate plots; no plotting evidence is claimed.

## Deployment Correction And Limits

The first bootstrap attempt stopped during import because the source-only snapshot lacked the
empty `sample_data` directory used by root discovery. No output directory, CUDA tests or model
training had started. The failed launcher log and bootstrap record are retained separately.
Creating that empty sentinel and checking both entrypoints with `--help` resolved the issue;
the 464 source files remained unchanged. No baseline or experiment rerun was needed.

Packaging fix `b33ca9c` now includes the empty directory automatically, with an archive-extraction
root-discovery regression test. This packaging-only change was made after the completed run;
it is not falsely reported as the executed training revision. The full local suite then passed
**220 tests**, zero failures/skips (`local_packaging_regression.xml`, 2.43 seconds).

Torch emitted a memory-efficient attention backward non-determinism warning in both e1 runs.
The existing training behavior was preserved; strict bitwise training replay is not claimed.
Peak VRAM was not instrumented. The measured pre/postflight snapshots are not peak values.

## Reproduction And Documentation

Run the local artifact audit without real-data forward or additional optimization:

```bash
MPLCONFIGDIR=/private/tmp/weighted-mpl /usr/local/bin/python3 -s paper/results/thp_static_hard_memory_smoke_20260904/audit_synced.py
```

`verification.json` records the audit, source identity, runtime, metrics and limitations.
`source_5080/` preserves the JSON/CSV/log evidence for version control. The four binary
checkpoints remain in the local and remote `search_artifacts` directory and are digest-pinned,
not added to this results commit.

Notion was created, updated and fetched again under `5. Model Design Enhancement`:
[2026-09-04 Count-aware THP Static Hard Memory CUDA and Full e1](https://www.notion.so/3d1bbe40561381f996b4c9677cfb4446).
The fetched content and correct parent are preserved in `notion_record.md`.

## Next Boundary

**Next / separate approval:** fresh Taxi and RAF seed42 e300/min40/patience40 screening against
the unchanged original THP and Hard-LMM criteria. Do not initialize with e1 checkpoints.
No additional dataset/seed, model adoption, held-out opening or e300 scheduler was started.
The short e1 work completed in this session, so it needs no ongoing monitor.
