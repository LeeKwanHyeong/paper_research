# Weighted Static Retrieval: Stopped By User

## Current Status

The user approved cancellation after reviewing the completed Taxi and RAF
results. Training stopped on **2026-09-03 at 16:12:57 KST**. The controller was
stopped before the training child so the queued Instacart run could not start.
Only this experiment's verified processes were terminated. GPU compute
processes were empty afterward, with 34 MiB used and utilization 0%; GDM stayed
inactive. The tmux pane is dead and retained. No service or 5090 was changed.

- Taxi and RAF: completed runs, locally audited; original results preserved.
- Intermittent: interrupted, not a completed benchmark. Last checkpoint and
  history are epoch 91; best epoch is 91. Current weights, optimizer state,
  best weights and history are preserved in `last_epoch_state.pt`.
- Instacart: e300 run never started. Its earlier successful e1 is unaffected.
- Hourly automation `hard-lmm-weighted-retrieval-5080`: **PAUSED**.
- Artifact and original pre-stop status/log/history/checkpoint backups were
  synchronized locally without deletion. No held-out test was used.

The status is `cancelled_by_user`, not success or an infrastructure failure.
No Intermittent final summary was fabricated. The earlier ETA no longer applies.
Verification is in `stop_verification.json`; completed-only comparisons are in
`partial_comparison.json`. Raw process/backup evidence remains in ignored
`search_artifacts/hard_lmm_weighted_static_seed42_20260903/stop_record.json`.

## Completed-Only Decision

| Dataset | Original MAE / RMSE | Candidate MAE / RMSE | Body MAE change | Gate |
| --- | ---: | ---: | ---: | --- |
| Taxi | 51.767732 / 181.537594 | 50.794521 / 177.751678 | -0.5877% | Fail body |
| RAF | 9.086491 / 36.123388 | 9.087862 / 36.111766 | +0.0751% | Fail body |

Both completed seed-42 runs pass the RMSE, gt-p99 MAE and Time NLL guardrails,
but fail the fixed 5% body improvement requirement. Because the prospective
broad-follow-up rule requires all four datasets to pass, this candidate is not
expanded. This is **0/2 evaluated datasets passing**, with one interrupted and
one unstarted, NOT four completed negative results. The stop decision was made
after observing these two results; it is recorded rather than retroactively
added to the original contract. No selector, loss, threshold or model was changed.

## Completed

- Repository/branch: local `paper_research/master`.
- Implementation and prospective contract: `ed80d04e033202585cd4e6609510afdb0e477245`.
- Only backbone change: the original hard top-4 static prototype average becomes
  a cosine-softmax weighted sum, temperature 1.0. No added parameters or head.
- Legacy initialization/state keys and the original mean formula/gradients pass
  regression tests. Weighted retrieval changes the residual for different scores
  within the same selected set; causal masks and checkpoint replay are tested.
- The entire candidate model will train fresh, not a frozen readout. Existing
  original Hard-LMM and external benchmark results will be reused.
- Local verification: `/usr/local/bin/python3 -s`, PyTorch 2.7.1; 93 tests passed,
  one CUDA-only legacy test skipped. No dependency installation was performed.

The first pytest group (weighted memory/runner, legacy memory backbones,
B0 diagnostics, persistent/dual memory) passed 69 tests with one skip. The second
group (B012 screening, count-aware TPP, dataset, model-role contracts) passed 24.
`git diff --check` passed before the implementation commit.

## Approved Transfer and Smoke

The user explicitly approved the original source-only archive transfer on
2026-09-03. The 275-file, 3,112,960-byte archive was transferred and verified:
SHA-256 `444ca993e66490c31aebd41554cbfc848001c298c525b103b34931b698084daa`.
All packaged source checksums matched revision `ed80d04`.

The package omitted four common Python helper files and the empty `sample_data`
root sentinel. Initial CUDA test collection therefore failed before training.
A revised archive transfer was not approved; it was not sent by another route.
Instead, checksum-identical helpers already present on 5080 were copied locally
on that server into a separate snapshot. An empty sentinel directory was added.
The corrected package passed 24 isolated local tests. No dataset, credential or
checkpoint was included in either source package.

Active snapshot on 5080:
`/home/leekwanhyeong/workspace/paper_research_weighted_static_ed80d04_serverdeps`.
Its 279-file source manifest SHA-256 is
`03dd8ac8bcd66f73651e587c253617534270bc65d556eb800d347020a46736c0`.
Existing `ai_env/bin/python -s` uses PyTorch 2.11.0+cu130. No runtime installation,
main project source overwrite, service/GDM change or 5090 command was performed.

- CUDA contract tests: **13 passed in 2.14 seconds**.
- Full train/validation e1: **Taxi, RAF, Intermittent and Instacart all passed**
  local final audits, including summary/history finite checks, fixed target and
  quantity-stratum counts, source revision and checkpoint digest.
- Held-out rows were not materialized; no held-out test artifact was generated.
- The first three e1 runs are under
  `search_artifacts/hard_lmm_weighted_static_smoke_20260903_serverdeps`.
- Instacart alone is under
  `search_artifacts/hard_lmm_weighted_static_smoke_20260903_instacart`.

The server smoke orchestrator stopped after completed Intermittent training
because its original baseline summary does not record `train_target_std`.
Local audit fix `7be9ff14cc61cd75dede9ca5f5b1353fdf0caff2` explicitly allows this
one pinned legacy direct log-MSE schema. The optional log-normal scale head is
absent, so that statistic is unused; train mean and all other comparisons remain
mandatory. The reference summary was not modified or given a fabricated value.
The fix and targeted regression tests passed **26 tests**. It was not deployed
over the frozen training source. Failed historical statuses were preserved;
completed e1 runs were not repeated. Consolidated evidence: `smoke_readiness.json`.

## Screening Launch (Historical)

- Started **2026-09-03 13:22:32 KST** on 5080 in tmux
  `hard_lmm_weighted_seed42_0903`.
- Artifact under the main server project:
  `search_artifacts/hard_lmm_weighted_static_seed42_20260903`.
- Four candidate-only fresh runs: Taxi, RAF, Intermittent, Instacart, seed 42,
  maximum e300, minimum e40, patience 40. No baseline/readout retraining.
- Initial observation at 13:23:27 KST: Taxi epoch 4 completed, best epoch 2;
  CUDA PID 356299, 2,344 MiB total VRAM used, utilization 78%, GDM inactive,
  no recent accessible kernel Xid/OOM entries. This is a snapshot, not a claim
  about sustained utilization or final performance.
- Hourly heartbeat was created as `hard-lmm-weighted-retrieval-5080`; it is now paused.

The tmux controller called the already-approved frozen `command`, `preflight`,
`baseline` and training entrypoint in isolated processes. It does not call the
old monolithic `main`, which would repeat the known Intermittent audit failure.
Modern summaries are audited remotely; Intermittent is explicitly marked pending
local audit. All four receive the corrected local final audit after sync.
`trained_pending_local_audit` means training finished, NOT accepted results.
Any other training, resource or audit failure is recorded as `failed`; there is
no automatic resume or performance-gate relaxation.

Approximate e1 train-plus-validation epoch times, excluding the second final
checkpoint evaluation, were 11.70 / 2.53 / 105.45 / 149.52 seconds in dataset
order. Assuming original baseline stop epochs 42 / 60 / 240 / 66 gives about
**10 hours**, around **September 3 23:20 KST**. If every run reaches e300 at those
speeds, about **22.4 hours**, around **September 4 11:50 KST**. These were conditional
estimates, not guaranteed limits; new early stopping and sustained speed can
change them. They were superseded by cancellation, not a forecast of remaining
work. Structured historical launch/ETA evidence: `launch_record.json`.

## Next Work

The requested post-stop investigation is complete. See `mechanism_analysis.md`,
`mechanism_audit.json` and `mechanism_verification.json`: weak direct quantity
effect of weighting, tail-concentrated Taxi gains, joint-versus-raw metric
selection mismatch, and bounded train-only task-gradient measurements. No new
training or model change ran. The proposed next experiment is an unlaunched,
original-Hard-LMM quantity route with a local-only time route; it is a diagnostic
candidate, not an adopted improvement. Local-encoder attribution follows only
if needed. No selector or acceptance threshold was changed retrospectively.
The new diagnostic is local only: its Notion update was blocked by external
transfer approval and was not retried. The earlier Notion stop record is unchanged.

1. Current baseline: preserve the two audited results and the interrupted
   checkpoint. Do not resume this candidate or expand seeds automatically.
2. Next decision: agree on the isolated routing contract before implementing or
   training another backbone. Selecting a favorable MAE epoch after the fact
   cannot replace the official validation-joint checkpoint.
3. No new training, backbone modification, service change or push is authorized
   by this cancellation. Stop evidence and partial results are scoped to local
   `paper_research/master` and the existing Notion experiment page.

The prospective JSON/Markdown contract is in `paper/contracts/hard_lmm_weighted_static_v1.*`.
Notion: [2026-09-03 Hard-LMM Similarity-Weighted Static Retrieval](https://app.notion.com/p/3d0bbe40561381d98efecd94ec3976a8).
The Notion page records the user-requested stop and completed-only results;
Intermittent and Instacart are explicitly not final evaluated results.
