# Hard-LMM prototype value-norm boundary audit

## Decision

**Proceed with one local candidate: VNC-Hard-LMM.** It is not a frozen readout
or scalar residual gate. It keeps B's cosine top-4 search and adds one signed,
zero-initialized contrast between the original raw-value mean and a mean whose
selected prototype row norms are equalized.

The exact formula has not been implemented or trained previously. It is,
however, adjacent to the old similarity-weighted static experiment because both
act at the top-4 aggregation boundary. The distinction is concrete: the old
candidate used cosine-softmax relevance weights on raw values; VNC uses
inverse-row-norm coefficients, leaves indices and similarities unchanged, and
contains B exactly at `alpha_raw=0`.

This audit supports implementing the candidate. It does not support a performance
claim. No train, validation, or held-out row was loaded, no prediction was made,
and no parameter was fitted.

## Overlap audit

| Boundary | Decision | Evidence |
| --- | --- | --- |
| Scalar gate after aggregate residual | Reject | Algebraically the existing shrinkage/readout direction; smooth and scalar variants passed 0/4 datasets. |
| General linear/MLP frozen readout | Reject | Joint selector passed 0/24 fits; it did not expose individual selected values or change shared time state. |
| Similarity-weighted static retrieval | Adjacent, not identical | Same aggregation position, different coefficient source; completed Taxi and RAF runs both failed the old body gate. |
| Online write/update confidence | Defer | B1/B2/Surprise already cover dynamic writes; cost and short-history evidence are unfavorable. |
| Aggregate residual/state normalization | Defer | Weak separation from aggregate shrinkage and broader state perturbation. |
| Prototype row-norm consistency | Select | Directly tests the mismatch between norm-free cosine search and norm-sensitive raw-value averaging. |

The most accurate description is **prototype row-norm equalization with a signed
identity blend**. It is not a new query-adaptive search or online memory system.

## Checkpoint-only geometry

The audit loaded the exact three seed-42 B checkpoints selected by validation raw
RMSE and verified their file and canonical state SHA-256 values. It inspected the
`1×64×64` static banks and enumerated all `64 choose 4 = 635,376` possible
four-row subsets in each bank.

| Dataset | Row-norm CV | Median `||r_N-r_B||/||r_B||` | p1 cosine `r_B,r_N` | Median non-scalar component | Non-collinear subsets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intermittent | 0.016978 | 0.005946 | 0.999996 | 0.001321 | 635,376 / 635,376 |
| Taxi | 0.195372 | 0.072332 | 0.994629 | 0.022968 | 635,376 / 635,376 |
| Instacart | 0.160172 | 0.091904 | 0.989194 | 0.061566 | 635,376 / 635,376 |

“Non-scalar component” is the normalized residual left after fitting the best
scalar multiple of `r_B` to `r_N`; the non-collinearity tolerance is `1e-10`.
Thus the operation is not algebraically a scalar shrinkage on these banks. It is
nearly scalar for many Intermittent subsets and materially larger for Taxi and
Instacart. That pattern is a structural observation, not evidence that the
actually retrieved subsets improve prediction error.

All row norms are finite and nonzero. The audit result and exact quantiles are in
`analysis.json`; the reproducible entrypoint is
`paper/scripts/audit_hard_lmm_value_norm_boundary.py`.

## Fixed implementation

- Backbone: `titantpp_hard_memory_value_norm`.
- Role: `hard_lmm_value_norm_candidate`.
- Contract: `hard_lmm_value_norm_consistent_v1`.
- One new scalar: `lmm.alpha_raw`, initialized to exact zero.
- Combination: `r_B + tanh(alpha_raw) * (r_N - r_B)`.
- Reference scale: mean row L2 norm over the same 64-row bank, detached after
  computation so unselected rows do not receive dense gradient.
- Tiny-row rule: norm `<=1e-8` gives exact zero normalized output and exact zero
  normalization-branch gradient.
- Norm computation: max-absolute-value-scaled L2, accumulated in FP32 for
  FP16/BF16/FP32 and in FP64 for FP64.
- The same state feeds time and quantity heads. This is a backbone candidate,
  not an output calibration or selector-only change.

## Local contract verification

The local suite verifies the following before any GPU execution:

- same-seed B/candidate common state, RNG, output, loss, and common gradient
  identity at the zero gate;
- finite nonzero `alpha_raw` gradient at the exact zero gate;
- unchanged top-4 indices and cosine similarities;
- a non-collinear opened-gate correction and changed shared outputs;
- selected-only memory gradient with detached bank reference;
- zero/tiny-row finite output and zero normalization gradient;
- FP16, BF16, FP32, and extreme-value finite behavior within representable input
  ranges;
- prefix causality, padding and target masking;
- strict B state injection with key, shape, dtype, finite, and failure-atomicity
  checks;
- dedicated checkpoint route, model/optimizer roundtrip, and fixed training role;
- raw-RMSE selector, zero adaptive-loss strength, and legacy intercept cap 300
  enforced before data access.

The implementation stays exploratory until CUDA/e1 and Instacart seed42 pass the
prospective gates. Existing bank diagnostics did not prove that averaging or row
norms cause prediction error, so a negative Instacart result should stop further
hand-designed static weighting variants rather than trigger post-result tuning.

Final local results were `72 passed, 1 skipped` for the VNC, runner, factory,
memory-backbone, and role-contract suite, plus `28 passed` for checkpoint
selection and the preceding QKV candidate audit regressions. Python compilation,
JSON parsing, `git diff --check`, and regenerated geometry validation also passed.

## Evaluation boundary

No GPU command or held-out evaluation was run. The next authorized decision
boundary is a separate 5090 approval: CUDA contracts and full-data e1 on all
three datasets, then Instacart seed42 only if every execution contract passes.
Taxi, Intermittent, additional seeds, and held-out evaluation remain conditional
and are not launched automatically.
