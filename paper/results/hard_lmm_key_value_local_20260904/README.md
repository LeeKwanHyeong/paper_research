# Hard-LMM separate-key retrieval: local implementation verification

## Decision

**Local implementation gate passed. Performance is unknown.** On 2026-09-04,
the approved contract, separate-key model path and synthetic contract tests were
completed in `paper_research/master`. No remote command, source transfer, GPU
training, real-data fit, scheduler, held-out evaluation, Notion update or push was
performed. Existing benchmark models and experimental artifacts remain frozen.

The candidate is `titantpp_key_value_static_memory`; it is not an adopted main
model. Relative to the weighted static control, only addressing keys are untied
from prototype values. Both output heads still consume h+r, and direct log-MSE,
the legacy time head, encoder and persistent-token architecture are unchanged.

## Verified behavior

- Values retain the original `lmm.mem` parameter and keys use independent
  `lmm.memory_keys` storage. Initialization clones values without consuming RNG.
- All original parameters match original/weighted initialization. Candidate
  outputs match weighted outputs exactly on CPU in both eval and seeded train
  mode. This is not output identity with the original uniform retrieval model.
- The independent float64 reference formula and gradients pass rtol/atol 1e-12.
  At tied initialization, the weighted value-bank gradient equals the sum of the
  candidate key and value gradients, within float32 tolerance.
- Key-only perturbation changes addressing and final quantity predictions while
  all other model tensors are unchanged. Value-only perturbation leaves the
  addressing indices and weights unchanged.
- A synthetic key-only fit with fixed values reduces its synthetic retrieval
  loss. Ordinary full-model synthetic optimizer steps also activate quantity-loss
  gradients to keys after the existing zero-initialized quantity head learns.
  These are functional tests, not dataset accuracy results.
- Unselected keys receive no per-query loss gradient. Degenerate singleton or
  identical-value cases can have zero addressing gradient; no gradient surrogate
  or extra loss is hidden in the implementation.
- Quantity and time losses reach keys, values and the original encoder.
  Extreme finite inputs produce finite outputs and gradients in tested cases.
- Target/future/padding perturbations preserve history predictions. Left/right
  padding, batch reordering and per-series evaluation agree. Evaluation leaves
  static parameters unchanged and there is no online memory state.
- Checkpoint metadata and strict tensor loading reject incorrect candidate IDs
  and implicit legacy conversions. Saved model/optimizer states reproduce eval
  predictions and the next seeded CPU optimizer step exactly.
- Original Hard-LMM, weighted retrieval, local-time and benchmark regression
  tests pass. `memory.py`, the Titan encoder, THP/NHP/SAHP implementations, MAC,
  TPP gated memory and THP static-memory implementation have no source diff.

The implementation adds 4,096 parameters at hidden dimension 64. Total candidate
counts are 93,891 for max_len 256, 82,883 for max_len 84 and 81,603 for max_len 64.
This is not a parameter-count-matched or compute-matched experiment.

## Test evidence

- Runtime unchanged: `/usr/local/bin/python3 -s`, Python 3.12.10, PyTorch 2.7.1,
  Polars 1.31.0, pytest 9.0.3, CPU. CUDA is unavailable locally.
- Final suite: **277 passed, 1 skipped**, 9.88 seconds, exit 0.
- The two new test files contain 35 passing cases. The remaining tests cover
  existing retrieval, routing, shared heads, model roles and benchmark behavior.
- The skipped case is an existing CUDA-only Titan-memory test, not a CPU pass.
  The new test module accepts `HARD_KEY_VALUE_TEST_DEVICE` for later authorized
  CUDA validation; no CUDA claim is made here.
- Machine-readable evidence: `pytest.xml`; command/file hashes: `verification.json`.
- First local run: 30 passed and one formula test failed on a 2.22e-16 float64
  broadcast-layout rounding difference. The independent formula tolerance was
  explicitly recorded as 1e-12; exact initial-output tests remain at zero error.
- First wider run: 272 passed, 1 skipped and one exact registry-list test failed.
  Its expected list was updated for the new opt-in candidate and the already
  registered THP static-memory candidate. No benchmark implementation changed.
- `git diff --check` and unchanged-reference-source diff checks passed.

## Integration boundaries

The factory, dedicated role/CLI registry, filtered train/validation loader branch
and fresh-run guard are integrated. Default benchmark lists are unchanged. An
existing candidate run directory is refused even with force-rerun, protecting
evidence from implicit resume, reuse or overwrite. No server launcher, deployment
package, dataset-specific smoke orchestration or performance run was added.

The contract audit distinguishes this candidate from prior `GatedSoftMemory`
(dense retrieval, projections, gate) and `TPPSpecificGatedMemory` (confidence/null
selection and online writes). Independent keys are established prior art, not an
invention or a proven contribution of this implementation.

## Remaining work

1. **Approval required / 5080:** finalize the execution wrapper and checksum/data
   audit, transfer committed source without `--delete`, then CUDA contracts and
   full Taxi/Instacart e1 smoke. Do not reuse smoke weights for screening.
2. **Separate approval required / 5080:** fresh seed-42 screening under the frozen
   contract. Keep the existing body/tail/time thresholds and hourly monitoring
   only when an actual long-running job is approved and started.
3. **Conditional:** Intermittent/RAF and then seed expansion only after the stated
   gates pass. Missing or interrupted weighted controls are not completed runs;
   in particular Instacart needs a separately approved matched weighted control
   before attributing an improvement solely to key/value untying. No automatic
   extra training or held-out evaluation follows from this local test result.
