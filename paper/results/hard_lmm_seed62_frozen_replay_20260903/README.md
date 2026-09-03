# Hard-LMM seed-62 frozen inference replay

## Completed diagnostic, not a new performance experiment

The same original Intermittent Hard-LMM seed-62 checkpoint was evaluated without
training on local CPU and RTX 5080 CUDA. Both inference routes completed and all
event evidence was independently reconciled. **5080 replay passes the unchanged
absolute 1e-5 gate; local CPU replay fails it.** No original parameter, loss,
time head, checkpoint-selection rule, installed package or service was changed.
No readout fit, backbone training or held-out evaluation ran.

Execution/contract commit on `paper_research/master`:
`d3bd2cab498a0b2b96a2f418fc5c3f6f1506d22f`.
The isolated 5080 source snapshot is
`/home/leekwanhyeong/workspace/paper_research_seed62_replay_d3bd2ca`.
All 233 registered source files were verified before/after inference and during
comparison. The shared server source tree was not overwritten.

Train-only 256-target checks completed on both devices before full validation.
The CPU logits exactly match the first 256 rows of the prior failed attempt's
retained seed-62 train cache. Both validation runs used the same 86,285 targets,
batch 128, no shuffle, and one CPU thread. Target/series/context/history/quantity
alignment is exact. Held-out rows were excluded before dataframe materialization.

| Execution | Runtime | Start (2026-09-03 KST) | End (KST) |
| --- | --- | --- | --- |
| Local train preflight | macOS CPU, torch 2.7.1 | 11:24:11 | 11:24:12 |
| 5080 train preflight | Linux CUDA 13.0, torch 2.11.0+cu130 | 11:24:19 | 11:24:19 |
| Local full validation | macOS CPU, torch 2.7.1 | 11:28:02 | 11:34:06 |
| 5080 full validation | Linux CUDA 13.0, torch 2.11.0+cu130 | 11:28:24 | 11:28:35 |

These times include verification and two instrumented inference passes per
batch; they are not training-cost benchmarks. The 5080 GPU was idle, GDM inactive
and free VRAM above 12,000 MiB before execution. No CUDA compute process remained
at the post-inference check. 5090 was not used. The hourly safety monitor was
paused after the diagnostic and result verification finished.

## Results

`official` calls the existing evaluator and retains its native-device raw
quantity predictions. `probe` calls the existing frozen feature extractor,
retains CPU logits and applies CPU softplus/expm1 as in the failed attempt.
Neither function's input/return value is modified by retrieval instrumentation.

| Validation route | Quantity MAE | Quantity RMSE | RMSE absolute difference from historical | Replay |
| --- | ---: | ---: | ---: | --- |
| Historical original summary | 0.806570615713 | 2.257153238763 | reference | reference |
| CPU official | 0.806563558774 | 2.257127734800 | 0.000025503964 | Fail |
| CPU probe | 0.806563558774 | 2.257127734800 | 0.000025503964 | Fail |
| 5080 official | 0.806570615713 | 2.257153238763 | 4.440892e-16 | Pass |
| 5080 probe | 0.806570420017 | 2.257152489519 | 7.492444e-7 | Pass |

The CPU discrepancy exactly reproduces the prior failure's absolute RMSE
difference, 2.550396374223496e-5. This new saved evidence establishes its direction:
CPU RMSE is slightly **lower**, not higher. The prior failed artifact remains
unchanged; its error alone did not contain that direction.

All four registered scalar checks pass on 5080 in both routes. Official MAE,
Time NLL and joint objective equal the historical summary; official RMSE differs
only at float64 reduction precision. CPU MAE, Time NLL and joint differences pass,
but CPU RMSE does not. Detailed unrounded values are in `metric_comparison.csv`.

Event-wise findings from `event_comparison.json`:

- CPU/CUDA official predictions differ bitwise on 83,506 of 86,285 targets;
  mean absolute difference is 1.3496121e-5 and maximum is 0.001037598.
- The selected top-4 prototype **set differs on zero targets**, both for train
  and validation. Validation has one ordering difference within the same set.
  This does not support top-k membership flips as the cause in these checked
  target retrievals; intermediate positions were not separately audited.
- CPU official and probe quantity predictions are bitwise identical on every
  target. Thus the separate feature-extraction route is not needed to reproduce
  the local CPU discrepancy.
- On 5080, native versus CPU postprocessed predictions differ by at most
  0.000213623 at an individual target, but both scalar replay gates pass.
  Scalar equivalence is not a claim of bitwise event equality.

## Interpretation and limits

**Confirmed:** the replay failure tracks the execution environment, not a changed
checkpoint, mismatched target population or readout-training failure. Both
environments use float64 quantity-error aggregation, so changing only the final
accumulation precision is not a supported remedy. The failed attempt correctly
stopped before new candidate fits; its small discrepancy did not justify silently
relaxing a registered gate.

**Inference:** environment-dependent forward arithmetic explains the observed
small prediction drift. This experiment changes hardware, operating system and
PyTorch/backend together; it does not uniquely isolate a CPU hardware effect, a
PyTorch version bug or a particular operator. The current 5080 environment is not
claimed to be an exact reconstruction of the historical installation. Historical
event predictions were not retained, so only historical scalar replay is shown.

This diagnostic says nothing new about body-MAE improvement, tail acceptance or
the eight pending readout fits. There is no candidate checkpoint, training
history, candidate scale-wise comparison or plot because no candidate was trained.
The original model and official T0 remain unchanged.

## Evidence and reproduction

`verification.json` verifies frozen weights, finite tensors/metrics, identical
event alignment, unchanged source/input/output digests, and raw-prediction metric
reconciliation. `evidence/` preserves the source manifest, contract, per-run
summaries/status, baseline audits, input/output digests and GPU preflights.
Large `events.pt` files remain under
`search_artifacts/hard_lmm_seed62_frozen_replay_20260903`; they are not committed.
Results are saved even when scalar replay fails. No run is overwritten.

Focused regression suite: **94 passed**, including six frozen-replay tests.
Tests cover non-mutating tracing, target masking, frozen-state rejection,
train/validation-only indices, failure evidence, native-prediction aggregation,
alignment rejection, and prototype order versus membership differences.

Example of the completed local validation invocation:

```bash
/usr/local/bin/python3 -s paper/scripts/run_hard_lmm_seed62_frozen_replay.py \
  --phase validation --device cpu \
  --data sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet \
  --checkpoint search_artifacts/hard_lmm_readout_replication_inputs_20260903/seed_62.pt \
  --reference-dir paper/results/count_aware_tpp_backbone_control_20260812/source_5080 \
  --source-manifest search_artifacts/hard_lmm_seed62_frozen_replay_20260903/source_manifest.json \
  --train-preflight search_artifacts/hard_lmm_seed62_frozen_replay_20260903/cpu_train \
  --output search_artifacts/hard_lmm_seed62_frozen_replay_20260903/cpu_validation
```

Independent comparator (supply a fresh output directory when repeating):

```bash
/usr/local/bin/python3 -s paper/scripts/compare_hard_lmm_seed62_frozen_replay.py \
  --artifact search_artifacts/hard_lmm_seed62_frozen_replay_20260903 \
  --output paper/results/hard_lmm_seed62_frozen_replay_20260903
```

Notion: [Hard-LMM seed-62 frozen inference replay](https://app.notion.com/p/3d0bbe40561381a8a21acd52a7db76c5).

## Next work, not executed

Register and commit an explicit extraction-environment amendment before resuming
the pending readout experiment. Prefer verified 5080 frozen extraction over
relaxing the 1e-5 scalar gate. Define cache provenance and cross-seed consistency,
then validate all required train/full-validation caches in that environment.
The seed-62 validation cache exists, but the complete 65,536-target CUDA train
cache and seed-52 CUDA caches are not produced by this 256-train diagnostic.

Keep readout architectures, losses, training/selection rules and body/tail gates
unchanged. Only after cache checks pass should the same eight fits run in a new
artifact with monitoring. Do not silently treat CPU feature extraction and CUDA
feature extraction as identical contracts. Further training was not authorized
by this inference-only step, and was not started.
