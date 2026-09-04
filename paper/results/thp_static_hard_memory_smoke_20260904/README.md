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

Pending at orchestration commit time. Initial read-only preflight at 2026-09-04 19:38 KST:
RTX 5080, driver 595.84, Python 3.12.13, 10/16303 MiB VRAM, no GPU jobs, GDM inactive,
no recent kernel log entries. Services were not changed.

## Next Boundary

After CUDA and full e1 artifact verification, report feasibility only. e1 metrics do not establish
improved performance and e1 checkpoints must not initialize a future fresh e300 screening.
