# THP + Static Hard Memory: Local Implementation Verification

## Scope

Local repository/branch: `paper_research/master`. Starting revision: `7f0bf8d`.
The user approved the local implementation/test stage after the prospective
contract freeze. This does not authorize server sync, CUDA or actual-data e1,
performance screening, services, scheduling, Notion publication or push.

The original contract and baseline registry remain immutable freeze records;
their `implementation_not_started` fields describe the freeze date, not this
subsequent implementation. This evidence directory records the newer stage.
No performance threshold or earlier candidate decision is changed.

## Implementation

`CountAwareTHPStaticMemory` inherits the unmodified project `CountAwareTHP`.
Only one original `HardLocalMemoryMatcher` is appended after the final layer.
Both heads receive the same masked h+r, using cosine hard top-4 selection and
the arithmetic mean of original unnormalized prototype vectors. No new loss,
head, persistent tokens, online writes, gates, temperature or local-time bypass.

Same-seed THP parameters and caller RNG are preserved by constructing the bank
after the complete base inside a CPU RNG-fork context. Total parameters are
104,387, including exactly 4,096 extra bank parameters. All train fresh.

Factory/runner registration uses the dedicated `thp_static_hard_memory` backbone
and `t0_thp_static_hard_memory` role, without changing the default benchmark list.
The route/head/scope/contract digest is checked before checkpoint restoration.
The candidate cannot use a prior cache, resume or force-overwrite a run. Launch
arguments permit only Taxi/RAF seed42 with the frozen learning contract and
either the full e1 feasibility budget or e300/min40/patience40. These capabilities
are implemented, not exercised on real data in this task.

New candidate data is filtered to train/validation before collection. Full target
counts and train-derived cutoffs are checked against the pinned baseline registry.
Runtime time statistics remain separate from legacy launch arguments. Errors
after creating an owned launch atomically mark it failed. Pre-existing or racing
foreign output directories are never claimed or rewritten. This cannot catch a
SIGKILL, host reboot or GPU driver crash; external monitoring remains necessary.

## Verification Boundaries

The executable tests cover model identity, nonzero gradients, exact retrieval,
causality, short/all-padding inputs, series/batch isolation, finite synthetic
optimizer steps, checkpoint and optimizer replay, invalid metadata, launch scope,
cache protection, filtered data and failure lifecycle.

A tiny synthetic CPU integration case exercises the real runner, saving its
history, summary, best/last checkpoints and metrics. It is a unit integration
test, NOT the Taxi/RAF e1 smoke or a performance experiment. Negative synthetic
test-only rows are excluded before materialization. Deliberately injected
training/reporting errors must produce failed status without automatic retries.

Actual local Taxi and RAF train/validation files are used only for byte hashes,
loader target counts, train cutoffs and validation quantity/history bin counts.
No actual-data model inference or training is performed. Counts and bins match
the original THP and Hard-LMM seed42 summaries, including omitted empty RAF
history bins. Held-out rows are not materialized or evaluated.

Initial runner tests exposed two test-fixture issues: too few synthetic train
values for distinct p50/p90/p95/p99 boundaries, and an assertion expecting empty
history bins that the historical reporter omits. The fixture/assertion was fixed;
production quantiles, model behavior and acceptance criteria were not relaxed.

Final result: **209 passed, 0 failed, 0 skipped** on Python 3.12.10 / Torch 2.7.1
CPU. Exact counts, commands, source hashes and the 16-requirement mapping are in
`verification.json` and `local_tests.xml`. Runtime packages are not modified. Tests of
the prior THP/Hard-LMM paths compare original factory metadata, parameters and
outputs as well as targeted existing contracts.

## Remaining Work, in Order

1. Approval required: synchronize the committed source to 5080 without --delete,
   verify checksums and GPU availability, then CUDA model contracts and full
   Taxi/RAF e1 feasibility. No resource readiness or CUDA compatibility is claimed.
2. Separate performance approval after feasibility: two fresh Taxi/RAF seed42
   e300 runs in independent processes with the frozen joint selector. Recheck
   baseline identities and add the requested monitor when long training begins.
3. Audit full artifacts and apply both THP and Hard-LMM gates on both datasets.
   A failure stops expansion; a pass is not statistical superiority or adoption.

No finish-time estimate is given before measuring the candidate on the target
GPU. Parameter equality and CPU synthetic tests do not establish a memory-cost
bound or an improvement over the existing benchmarks.
