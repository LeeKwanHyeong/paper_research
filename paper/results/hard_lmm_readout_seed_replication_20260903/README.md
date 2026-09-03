# Intermittent frozen readout checkpoint-seed replication

## Status: blocked before main fits

The registered attempt stopped at the seed-62 original-checkpoint replay gate.
**No new main readout fit ran, so there is no seed-52/62 performance comparison or
replication verdict.** This is not evidence that the readouts failed the body/tail
acceptance criteria. The implementation and failure evidence are verified; the
proposed performance experiment is not complete.

The attempt ran on local CPU from 2026-09-03 10:29:31 to 10:40:00 KST. It used the
existing `/usr/local/bin/python3 -s` runtime, torch 2.7.1, one thread, without an
installation change. Neither server ran additional compute or had a service
changed. Only the two existing checkpoints were downloaded from 5080, without
`--delete`. The hourly safety monitor was paused after the explicit failure.

## Frozen design and completed checks

Implementation/contract commit on `paper_research/master`:
`882d8a956f2d7ad1864549308992d3d2c1a45aad`.
The contract is `paper/contracts/hard_lmm_readout_seed_replication_v1.json` with its
Markdown companion. Seed 42 remains the prior discovery result, not a new fit.
Seeds 52/62 identify original Hard-LMM checkpoints, not adapter random seeds.

- Planned: two checkpoint seeds times constant/linear times log-MSE/raw-MAE,
  eight main fits, each with the same two registered checkpoint selectors.
- All eight discarded train-only one-epoch preflights passed, with nonzero first
  gradients and changed parameters/predictions. These are not main-fit results.
- Original checkpoint file/state digests match the historical audit for both
  seeds. All 226 inherited source files and three execution files are unchanged.
- Both 65,536-target train caches were extracted, frozen, finite, and exactly
  aligned to the prior seed-42 train target/series/context/history/quantity arrays.
- Seed 52's complete 86,285-target validation cache is also aligned and finite;
  all four original scalar replay metrics passed the absolute 1e-5 gate.
- Focused regression suite: **88 passed**, including 11 new replication tests.
- No original parameter was trained, no held-out artifact was generated, and no
  held-out target was evaluated. Body/tail gates and official T0 are unchanged.

## Failure and what it does not establish

| Replay check | Absolute difference | Registered tolerance | Status |
| --- | ---: | ---: | --- |
| Seed 52 quantity MAE | 0.000002824801 | 0.000010 | Pass |
| Seed 52 quantity RMSE | 0.000005622933 | 0.000010 | Pass |
| Seed 52 Time NLL | 0.000000001326 | 0.000010 | Pass |
| Seed 52 joint objective | 0.000000015223 | 0.000010 | Pass |
| Seed 62 quantity RMSE | 0.000025503964 | 0.000010 | Fail |

The seed-62 discrepancy is about 0.0011299% of its original RMSE 2.2571532387633177.
Its small magnitude is not permission to silently relax the registered tolerance.
The original summary is pinned, and the runner correctly recorded `failed` and
exited with code 1 before any main fit.

Read-only inspection of the historical evaluator at source revision
`044add1f3de768d804d9f0269fd0013bd9658a35` showed that raw predictions and quantities
were cast to NumPy float64 before error accumulation. The current probe likewise
uses float64 error aggregation. Consequently, **a simple float32-versus-float64
accumulation explanation is not established**. Device/backend arithmetic or hard
top-k sensitivity are possibilities, not verified causes. No hardware-matched,
event-wise prediction comparison was performed in this CPU-only attempt.

The failed seed-62 validation cache was not saved: replay validation occurs before
that write. Its baseline audit therefore contains only the successfully retained
train cache. The error records the absolute RMSE discrepancy, not its direction.
Do not reconstruct an observed RMSE by guessing the sign. Later replay metrics
were not reached after the RMSE exception. The status's `cell` field retains the
last train-preflight cell; `stage=extract_validation` is the actual failure stage,
not a main linear/raw-MAE fit failure.

## Evidence

`failure_verification.json` verifies the partial attempt and retained caches. It is
deliberately not named `artifact_verification.json`: no completed candidate result
has been verified. The immutable raw attempt remains at
`search_artifacts/hard_lmm_readout_seed_replication_20260903`.
Large caches/checkpoints are not committed. Copied contracts, source manifest,
status, train-only preflights and baseline audits preserve the partial provenance.
There is no aggregate summary, main epoch history, selected adapter checkpoint,
acceptance decision, or 3-seed result for this attempt.

The original seed-42 findings are unchanged; see
`paper/results/hard_lmm_readout_factorial_20260903/README.md`.
They must not be relabeled as replicated evidence.

Notion: [Intermittent frozen readout seed replication](https://app.notion.com/p/3d0bbe40561381bda2f5d38ea85a1e24).

## Next action, not executed

Keep this failed attempt immutable and the readout hyperparameters/body-tail
thresholds fixed. First isolate the original-prediction discrepancy using a
separately scoped, hardware/runtime-matched frozen-inference comparison; do not
start backbone training or new adapter fits to investigate it. Use train inputs
for the initial diagnostic, then compare the full validation only when the
prediction path is understood. Record both sides' event predictions and metrics
before applying a fail-closed check, so a failed check retains usable evidence.

Using 5080 compute is outside this CPU-only contract and needs an explicit scope
approval and resource check. If a numerical-equivalence amendment is required,
justify and commit it before viewing any new candidate performance; do not choose
its threshold to make a candidate pass. Only after baseline equivalence passes
should the eight unchanged main fits resume in a new artifact. No fresh backbone
training, benchmark replacement, or held-out evaluation is authorized here.
