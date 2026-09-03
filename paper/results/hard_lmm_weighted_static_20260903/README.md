# Weighted Static Retrieval: Screening In Progress

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

## Active Screening

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
- Hourly heartbeat: `hard-lmm-weighted-retrieval-5080`, active.

The tmux controller calls the already-approved frozen `command`, `preflight`,
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
speeds, about **22.4 hours**, around **September 4 11:50 KST**. These are conditional
estimates, not guaranteed limits; new early stopping and sustained speed can
change them. Structured launch/ETA evidence: `launch_record.json`.

## Next Work

1. In progress: leave the four candidate-only runs to finish; hourly monitoring
   checks progress and resource failures without restarting work.
2. Next: sync artifacts without `--delete`, validate manifests/logs/contracts,
   summary/history/strata/checkpoint digests and held-out absence using the local
   fixed `audit_run`. Original Intermittent summary is available at
   `paper/results/count_aware_tpp_backbone_control_20260812/source_5080/runs/titantpp/count_only_log_regression/seed_42/summary.json`;
   its digest must match the registry. Other baselines use registry paths.
3. Compare against original frozen baseline identities using the unchanged body,
   RMSE, tail and time gates. Record runtime differences and single-seed limits,
   update Notion and commit related results only to local `paper_research/master`.

The prospective JSON/Markdown contract is in `paper/contracts/hard_lmm_weighted_static_v1.*`.
Notion: [2026-09-03 Hard-LMM Similarity-Weighted Static Retrieval](https://app.notion.com/p/3d0bbe40561381d98efecd94ec3976a8).
The Notion result section remains empty because there are no performance results.
