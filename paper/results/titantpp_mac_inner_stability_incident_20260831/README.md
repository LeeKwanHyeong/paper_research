# TitanTPP-MAC Instacart Failure: Corrected Diagnosis

## Scope And Findings

Both recovery2 jobs failed again with source
`b1d9e638c68bc7bab9ace6b5c8fa9d0af989c8f7`: Instacart seed 42 on the 5090
at zero-based batch 3705, and seed 62 on the 5080 at batch 13439. Neither
completed its first epoch or produced a usable training checkpoint.

The earlier time-head explanation was not established on either actual
failure. The intercept-limit regression fixed a separate synthetic overflow
case, but the same real batches still failed afterward. Claims that shorter
datasets passed because the time intercept had not grown enough are withdrawn.

Bounded diagnostic runs reproduced both failures at exactly the same batches,
including the same reported time slope. The pre-forward input, model and RNG
were captured. Inputs and outer model parameters were finite. On both GPUs,
the first non-finite stage was the neural associative memory update, before
the time or quantity heads. Eager token-wise replay located the failure in
`associative_gradients`; the compiled scan also failed.

In the seed-42 eager trace, memory weights grew from sub-unit values to about
`3.07e23` in fourteen writes. The next associative gradient became non-finite.
Learned inner update rates reached about 0.116 and momentum about 0.999.
Outer `grad_clip=1` runs only after loss backward and cannot bound these
forward-time inner updates.

## Verification Gap

The former extreme-input tests used sequences of length four. They did not
exercise the 16-event MAC segment boundary with a learned unstable memory.
A short smoke also did not cover a full Instacart epoch of 15,557
batches. Most importantly, the actual failing batch was not reproduced before
the preceding recovery was launched. This was a diagnosis and release-gate
failure, not evidence that the user needed to reboot or change the dataset.

## Correction And Evidence Boundary

Source `c4dbf856c32e6502acc660ffac23c3e2f68e5375` adds the explicit option
`--titans-memory-gradient-clip 1`. It clips the joint norm of the four
associative-gradient tensors separately for each series and observed token,
before momentum and forgetting. Norm computation scales before squaring to
avoid overflow. The operation remains differentiable. The disabled `None`
setting preserves the unbounded reference recurrence.

This is an inner-learning policy change, not a semantics-preserving speed
optimization. Earlier unbounded MAC seed results cannot be pooled with it.
It does not change the count objective, time head, input, split, persistent
parameters, segment size, series reset or predict-before-write order. It does
not discard batches, replace NaNs with zero, or resume failed checkpoints.
RMTPP/THP/NHP/SAHP code and results are not changed by this option.

Verified before the full-epoch checks:

- Both actual failing fixtures pass compiled and eager forward AND backward
  with the explicit clipping policy.
- Both GPUs pass all nine inner-gradient stability tests, including a
  deterministic unbounded-recurrence counterexample, 64-write stability,
  masking, per-series clipping and compiled/token scan agreement.
- 5080 runtime: the combined causal/state/optimization/checkpoint and gate
  suite has 38 passed, one skipped and one deselected. CUDA was deselected
  in that CPU suite because the nine CUDA/stability tests ran separately.
  The raw report is `cpu_contract_tests.xml`. One
  historical Git-object digest test is skipped in the exported snapshot;
  historical digests remain anchored to revision `08e5988`, not overwritten.
- Seven fail-closed preflight-header tests pass locally. The broader local
  suite cannot collect because the existing local Polars installation is
  inconsistent; those behavior tests were run in the server environment.

Raw fixtures and traces are preserved under local
`search_artifacts/titantpp_mac_nonfinite_20260831/seed42` and `seed62`.
`evidence.json` records their digests and condensed outcomes. No held-out test
target was evaluated.

## Full-Epoch Verification Status

Both servers use the isolated snapshot
`/home/leekwanhyeong/workspace/paper_research_mac_stability_diagnostic_20260831`.
Its 24 declared source digests were checked against the committed source.
The old failed runs and original snapshots remain untouched.

| Server | tmux | Run | Status verified at 2026-08-31 05:37 KST |
|---|---|---|---|
| 5090 | `mac_stability_full_e1_42_0831` | Instacart seed 42 | Complete; full coverage and checkpoint gate passed |
| 5080 | `mac_stability_full_e1_62_0831` | Instacart seed 62 | Complete; full coverage and checkpoint gate passed |
| 5090 | `mac_stability_full_e1_52_0831` | Instacart seed 52 | Complete; full coverage and checkpoint gate passed |

Artifact roots are
`search_artifacts/titantpp_mac_stability_preflight_20260831_<server>`.
The diagnostic wrapper atomically records completion or failure and preserves
the offending forward on non-finite loss. The validator checks source/data/
split digests, exact training-target and batch counts, full validation strata
counts, finite metrics, checkpoint tensor digest, strict restore, prediction
and observed-history memory replay, and held-out artifact absence.

For each completed run, the gate verified 1,991,192 training targets across
15,557 batches and 503,733 validation targets. All three histories and summaries
are finite. Strict checkpoint restore reproduced predictions and
observed-history memory tensors exactly. See `validation_report_seed42.json`,
`validation_report_seed52.json` and `validation_report_seed62.json`; these are stability checks, not e300
results or evidence of model quality. The artifacts were synchronized locally
without `--delete`, and all 24 source digests were checked again locally.
The completion audit also independently rechecked each local launch contract,
history/summary finite values, quantity/history stratum counts, held-out
artifact absence and canonical checkpoint tensor digest. The three existing
complete remote validation reports were retained, not regenerated.

The 5090 kernel journal is not readable by the current SSH user and noninteractive
sudo requires a password. Absence of NVIDIA Xid on that server is therefore
not verified. At the completion snapshot, both GDM services were inactive,
there were no CUDA compute processes, and GPU utilization was zero (5080:
34 MiB, 5090: 2 MiB). All three training tmux sessions had exited normally.
No relevant Xid/OOM/watchdog event appeared in the readable 5080 kernel log
since 03:00 KST. No display or shared runtime service was changed as part of
this repair. The machine-readable observation is `completion_audit.json`.

Hourly heartbeat `titantpp-mac` tracked all three preflights. Seed52's bounded
postflight waiter produced its complete validation report before the hourly
check. This close-out retires the completed preflight monitor; it does not
authorize an old unbounded launcher or a new e300 grid.

The existing Notion page `2026-08-30 TitanTPP-MAC 3-Seed Validation Launch`
was updated and fetched again to verify the corrected diagnosis, new policy,
all-three-seed completion and stability/performance distinction:
https://app.notion.com/p/3ccbbe405613817ea095f605888d39ac

## Next Order

Current baseline: all three Instacart full-epoch stability preflights and
checkpoint gates are complete; no held-out evaluation or new e300 was run.

1. Verify the remaining dataset/context shapes under the same
   explicit policy before a new long-running grid.
2. Confirm the expanded run scope, then freeze a new matched MAC grid and
   monitoring target before e300. The user has been asked to approve all
   four datasets times three seeds (12 MAC runs), rather than mixing three
   historical seed42 runs with nine new ones. No new e300 grid is launched.
   Recompute every MAC seed used in a three-seed table under the same inner policy;
   preserve old runs only as separately labelled historical evidence.
