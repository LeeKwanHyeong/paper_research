# Hard-LMM Frozen Calibration and Shrinkage Diagnostic

## Status and evidence

- Experiment complete: 5080, 2026-09-03 08:20:05 to 08:21:07 KST (62 seconds).
- All four datasets and both independent candidates completed. No fresh backbone
  training, seed expansion, held-out evaluation, or service modification occurred.
- Contract commit: `e9eaf89`; execution commit:
  `474ebdde2598e4a2509a83d41fc8d404ca99fce0`.
- Runtime: PyTorch `2.11.0+cu130` on 5080. Frozen feature extraction used CUDA;
  the additional 2,241-parameter modules were fitted on CPU.
- Source-only packaging initially omitted an indirect import and the empty
  `sample_data` root sentinel. This was resolved before training by packaging
  all 221 tracked Python/contract files with checksums. No model code was changed.
- Initial local tests: 26 passed with PyTorch 2.7.1. Actual-data CUDA smoke:
  8/8 passed, with finite gradients and exact adapter checkpoint replay.
- Final expanded local test collection: 29 passed, 2 failed after a user-site
  PyTorch 2.14.0 began shadowing the original 2.7.1 installation. The two failures
  concern the zero-initialized clamp derivative. A direct check gives derivative
  0 on the current default runtime and 1 on `python3 -s` / original 2.7.1.
  The environment has not been altered or downgraded. Final verification now
  explicitly uses `/usr/local/bin/python3 -s` (original PyTorch 2.7.1): all
  31 retrieval/probe/artifact regression tests passed. See `final_local_tests.xml`.
  The independent eight-run artifact audit also passed again with this runtime.
  The default user-site runtime remains untouched and is not approved for this
  zero-initialized clamp probe. The executed experiment is not retroactively
  changed to obtain different results.

## Frozen diagnostic contract

The original benchmark seed-42 checkpoints are the reference, not the later
fresh B0/B1/B2 screening checkpoints. Encoder, persistent tokens, static
prototype bank, quantity head, and time head stay frozen in eval mode.

Calibration adds `0.05*tanh(MLP(features))` to the original quantity logit.
Shrinkage keeps top-4 arithmetic-mean retrieval but multiplies its quantity
contribution by a gate in `[0.8, 1]`. The original memory-enhanced time route is
unchanged. Features contain observed history and retrieval statistics only.
Neither true next quantity nor next delta-time enters the adapter.

Both candidates retain direct log-MSE plus constant frozen Time NLL, Adam
lr=0.001, batch=128, max40/min10/patience10 and validation-joint selection,
including the exact identity at epoch 0. This is a bounded diagnostic budget,
not a matched fresh e300 experiment. Hyperparameters were not retuned by dataset.

| Dataset | Available train targets | Used train targets | Full validation targets |
| --- | ---: | ---: | ---: |
| Intermittent v2 | 393,824 | 65,536 | 86,285 |
| Taxi | 38,393 | 38,393 | 8,268 |
| RAF | 25,779 | 25,779 | 6,690 |
| Instacart | 1,991,192 | 65,536 | 503,733 |

Train samples use fixed seed-42 uniform sampling without replacement. Validation
is never subsampled. All test rows are excluded before feature materialization.

## Results

Negative percentage changes indicate lower error. Body means quantity at or
below the pinned train p95; extreme tail means quantity above train p99.

| Dataset | Candidate | Best/final epoch | Body MAE change | Overall MAE change | RMSE change | Extreme-tail MAE change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Intermittent | Calibration | 0/10 | 0.00% | 0.00% | 0.00% | 0.00% |
| Intermittent | Shrinkage | 0/10 | 0.00% | 0.00% | 0.00% | 0.00% |
| Taxi | Calibration | 20/30 | -2.82% | -9.09% | -9.93% | -13.76% |
| Taxi | Shrinkage | 0/10 | 0.00% | 0.00% | 0.00% | 0.00% |
| RAF | Calibration | 39/40 | +1.44% | -0.16% | -0.72% | -0.90% |
| RAF | Shrinkage | 0/10 | 0.00% | 0.00% | 0.00% | 0.00% |
| Instacart | Calibration | 0/10 | 0.00% | 0.00% | 0.00% | 0.00% |
| Instacart | Shrinkage | 0/10 | 0.00% | 0.00% | 0.00% | 0.00% |

Taxi calibration changes overall MAE/RMSE from `51.767734 / 181.537602` to
`47.063531 / 163.508461`; body MAE changes `23.167407 -> 22.514055`.
RAF calibration changes body MAE `4.225942 -> 4.286596`, despite modest aggregate
and tail improvements. Time NLL is exactly unchanged for every candidate.

**Exploratory acceptance: 0/8 pass.** No candidate meets the preregistered 5%
body-MAE improvement requirement with the remaining guardrails. None is promoted
to fresh training or multi-seed evaluation.

## Interpretation and limitations

- Taxi provides evidence of a useful readout correction for this checkpoint,
  particularly in large-error events. This is not a demonstrated backbone gain
  or a sufficient solution to the body-MAE problem.
- Calibration on Intermittent and Instacart makes the validation joint objective
  worse, so checkpoint selection correctly returns the unmodified epoch 0.
- Shrinkage selects epoch 0 everywhere. Its first gradients were nonzero on the
  execution runtime; nevertheless, Intermittent, Taxi and RAF validation histories
  remain exactly at identity. A one-sided clamped gate can enter an inactive
  region. The saved evidence does not establish whether confidence-based
  shrinkage is intrinsically unhelpful or this parameterization is limiting it.
- The selected calibration correction has absolute p95 approximately 0.05 on
  Taxi and RAF, close to the fixed bound. This does not authorize increasing the
  bound after seeing validation results.
- Reusing checkpoints already selected on validation, a bounded train sample,
  and one seed makes these results exploratory. Frozen-probe gains do not
  guarantee fresh-training generalization. Held-out test remains locked.

## Artifact verification

The full artifact is stored locally and remotely under
`search_artifacts/hard_lmm_frozen_probe_20260903`. Feature caches, adapter
checkpoints and event-level parquet files are retained there, not committed.
This directory retains compact summaries, histories, scale tables and audits.

- All source, data, split, baseline checkpoint and adapter/cache digests match.
- Baseline validation replay maximum metric discrepancy is `8.073e-6`, within
  the fixed `1e-5` replay tolerance. Time NLL discrepancy is zero.
- Base state digest is unchanged before/after all eight fits; checkpoint replay
  was exact on the execution runtime.
- Independent event-level MAE/RMSE and error-delta reconciliation covers all
  quantity/history strata. Maximum metric discrepancy is `1.600e-8`, due to
  FP32 whole-vector versus stratum-wise softplus/expm1 rounding. The reporting
  check permits `atol=rtol=1e-6`; acceptance thresholds are unchanged.
- All numerical metrics are finite; empty strata are explicit. No test artifact
  or NaN/Traceback/OOM appears in the completed experiment.

Audit command:

```bash
/usr/local/bin/python3 -s paper/scripts/validate_hard_lmm_frozen_probe.py \
  --artifact search_artifacts/hard_lmm_frozen_probe_20260903 \
  --output search_artifacts/hard_lmm_frozen_probe_20260903/local_verification.json
```

## Remaining work

1. Audit the inactive shrinkage gate using train-only score/gradient traces and
   explain Taxi calibration gains with a bounded train-fitted constant control.
   Final interpreter alignment and the 31-test verification are complete without
   changing installed packages. Experiment code and the v1 contract stay frozen.
2. Only after that diagnosis, separately define and authorize any fresh-training
   candidate. Do not automatically combine probes or change loss/time selection.

Notion: https://app.notion.com/p/3cfbbe4056138105aab2f39e3dd99749
