# TitanTPP M0 Intermittent Seed-42 e50 Validation Screening Result

Notion target:

- `5. Model Design Enhancement > Enhancement & Validation History`
- Existing page: `TitanTPP M0 Intermittent Seed-42 e50 Validation Screening`
- Do not create a duplicate page under `2. Confirm and Refine Topic`.

## Run Status

- Server / tmux: `5090` / `titantpp_m0_inter_e50_0713`
- Initial start: `2026-07-13 11:26:04 KST`
- Retry start: `2026-07-13 11:27:45 KST`
- End: `2026-07-13 11:34:38 KST`
- Retry runtime: `6m 53s`
- Exit code: `0`
- Completion marker: `SCREENING_SUCCESS`
- Best validation NLL: epoch `24`
- NaN / Inf / Traceback / runtime error: none in applicable metrics

The first attempt stopped before M0 training because the V2 reference exporter tried
to serialize a decoder-inapplicable NaN. The exporter was changed to write only
decoder-inapplicable metrics as JSON null, and the model equation and training
configuration were not changed. The first attempt remains in the `_attempt_1`
artifact directories.

## Artifact Contract

- M0: `search_artifacts/model_enhancement_m0_inter_short_e50_0713`
- Frozen V2 reference: `search_artifacts/model_enhancement_v2_inter_validation_reference_m0_0713`
- V2 reference checkpoint: `best_val_nll`, epoch `19`
- V2 reference targets: `41,901`
- V2 reference manifest: `evaluation_split=validation`, `held_out_test_read=false`
- M0 train / validation / test sample counts: `136,256 / 41,901 / 41,344`
- Train-only magnitude count / mean / std: `159,643 / 1.266239 / 1.453546`

## Validation Gate

All comparisons use the M0 `best_val_nll` checkpoint at epoch `24` and the frozen
V2 validation-only reference.

| Metric | V2 | M0 | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Total NLL | `5.666520` | `5.574098` | `-1.631%` | PASS, regression <= `0.5%` |
| Marker NLL | `0.991274` | `0.999921` | `+0.872%` | PASS, regression <= `1%` |
| Time NLL | `4.675246` | `4.574177` | `-2.162%` | PASS, regression <= `0.5%` |
| Quantity MAE | `3.060182` | `2.760559` | `-9.791%` | PASS, improvement >= `3%` |
| Log2 quantity MAE | `0.588742` | `0.645850` | `+9.700%` | FAIL, improvement >= `3%` |
| Mark accuracy | `57.249%` | `53.614%` | `-3.635%p` | FAIL, gap >= `-0.25%p` |
| DT MAE | `42.064581` | `40.197025` | `-4.440%` | PASS, regression <= `2%` |
| Share >= 5% quantity bucket | `1-9` baseline | `+8.623%` MAE | limit `+5%` | FAIL |
| Runtime / artifact | completed | completed | no applicable non-finite value | PASS |

Overall decision: `FAIL`. Log2 quantity MAE, mark accuracy, and the dominant
quantity-bucket safety condition fail. M0 does not advance to matched multi-seed.

## Secondary Marker Diagnostics

| Metric | V2 | M0 | Change |
| --- | ---: | ---: | ---: |
| Normalized RPS | `0.035283` | `0.035833` | `+1.560%` |
| Mark MAE | `0.487411` | `0.549820` | `+12.804%` |
| Balanced accuracy | `42.664%` | `46.062%` | `+3.398%p` |
| Macro F1 | `43.302%` | `46.067%` | `+2.765%p` |
| Predicted mark-0 share | `45.030%` | `63.368%` | `+18.338%p` |
| Mark-0 recall | `75.543%` | `90.043%` | `+14.500%p` |
| Mark-1 recall | `49.616%` | `18.234%` | `-31.383%p` |

M0 increases mark-0 prediction substantially. This raises mark-0 recall and the
macro-oriented metrics, but the loss of mark-1 recall lowers overall accuracy and
increases mark-distance error.

## Validation Scale-Wise Result

| Quantity bucket | Share | Quantity MAE change | Log absolute error change | Mark accuracy gap |
| --- | ---: | ---: | ---: | ---: |
| `1-9` | `88.666%` | `+8.623%` | `+12.155%` | `-4.099%p` |
| `10-99` | `10.723%` | `-2.289%` | `-0.435%` | `-0.601%p` |
| `100-999` | `0.527%` | `-13.737%` | `-26.267%` | `+13.575%p` |
| `1000-9999` | `0.084%` | `-41.977%` | `-59.627%` | `-8.571%p` |

The overall raw quantity MAE improves because medium and tail quantities carry
large absolute errors. The `1-9` bucket contains `88.666%` of validation targets
and regresses on both quantity and log error, which explains the failed log2 MAE
gate. The `1000-9999` row has only `35` targets and is diagnostic rather than a
standalone conclusion.

## Learning-Curve Reading

- Best total/time NLL: epoch `24`; total NLL improvement is led by time NLL.
- Best marker NLL: epoch `29`.
- Best log2 quantity MAE: epoch `19`, value `0.633809`; still `7.654%` worse than V2.
- Best quantity MAE and mark accuracy: epoch `34`, values `2.642614` and `58.080%`.
- Epoch `34` still fails log2 quantity MAE and slightly exceeds the total-NLL
  regression allowance, so an alternative checkpoint does not rescue the gate.
- Validation curves remain noisy after the early convergence phase; final epoch
  NLL `5.800540` is worse than the epoch-24 checkpoint.

## Interpretation And Decision

- Direct global-normalized regression helps absolute errors on larger quantities,
  but does not improve the dominant low-quantity region.
- The NLL gain is a time-head gain, not evidence that marker modeling improved.
- The simultaneous raw/log quantity requirement correctly catches the tail-weighted
  raw-MAE improvement.
- The mark-0 prediction shift indicates shared-representation or multi-task
  interference remains. This is a hypothesis, not a causal conclusion from one seed.
- Per the predeclared branch, stop M0 matched multi-seed and do not start the
  existing log-domain M1-M4 candidates.
- Retain M0 as a negative ablation showing that direct regression alone is not a
  common Intermittent replacement for the marked quantity decoder.

## Post-Result Domain Reclassification

- M0 uses `log2(qty)` with fixed train-global statistics. It does not compute
  instance statistics and is therefore not a RevIN experiment.
- The existing M1-M4 design is also log-domain because it normalizes
  `z=log2(qty)`.
- The failed M0 gate rejects the completed log-domain direct/global setting and
  stops its dependent log-domain branch; it does not reject raw-quantity RevIN.
- Raw-domain Q0 global, Q1 causal masked RevIN, and Q2 causal shrinkage RevIN
  performance remains untested. Their separate contract is complete, while the
  raw train-only audit and implementation have not started.
- V5b class-prior correction remains a fallback, not the only next option.

## Validation-Only Audit Note

The V2 reference evaluation itself remained validation-only and records
`held_out_test_read=false`. The M0 runner nevertheless writes test columns into
`leaderboard/runs.csv`. During schema inspection, those columns were displayed
before the gate calculation. They were not used in any calculation, table, or
decision above, and the gate had been fixed before execution. This screening is
therefore kept as a validation decision with an explicit audit note. Future blind
screening must not open `leaderboard/runs.csv`, `leaderboard/test_*`, or
`paper_outputs/report.md` before the validation decision is recorded.
