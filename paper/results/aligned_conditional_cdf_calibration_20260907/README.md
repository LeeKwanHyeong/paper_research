# Conditional CDF calibration validation decision

The common conditional Kumaraswamy CDF calibration candidate was rejected at
the frozen Taxi seed-42 gate. The result supports a small duration-distribution
gain from the aligned-B hidden state, but it does not satisfy the predeclared
attribution and point-prediction requirements needed for a common paper model.

## Execution contract

- Source revision: `cf7faa58c5a58d0772d486a5748abcdbe02e1b60`
- Contract SHA-256: `86632f4f7ea22960c8f028cc8e2347986ae837bee29c9fee804fdf25b46c1a52`
- Runtime: NVIDIA GeForce RTX 5080
- Remote artifact root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/aligned_cdf_cf7faa5_5080_20260907`
- Evaluation scope: validation only; held-out test was not loaded or evaluated.

All three full-data e1 runs passed CUDA, finite-gradient, checkpoint,
full-count, quantity-identity, and held-out-exclusion checks. The admitted
train/validation target counts were 393,824/86,285 for Intermittent,
38,393/8,268 for Taxi, and 1,991,192/503,733 for Instacart. Peak reserved GPU
memory was at most 75,497,472 bytes.

## Taxi seed-42 screening

| Path | Selected epoch | Primary proper Time NLL |
| --- | ---: | ---: |
| aligned-B | 0 | 0.6956474254 |
| candidate | 99 | 0.6889874164 |
| global-shape control | 0 | 0.6956474254 |
| permuted-hidden control | 29 | 0.6934882532 |
| aligned-A reference | fixed | 0.6506978539 |

The candidate improved aligned-B by `0.0066600090` and the global control by
the same amount, passing both `0.005` gates. The continuous reference NLL also
changed from `6.5362210527` to `5.5902077818`.

Three predeclared gates failed:

- The candidate remained `0.0382895625` above aligned-A, exceeding the allowed
  `0.01` gap.
- Its advantage over the same-capacity permuted-hidden control was
  `0.0045008368`, short of the `0.005` attribution margin by `0.0004991632`.
- Time-median MAE changed from `1.1569396578` to `1.1879788571`, a `2.6829%`
  worsening against the `2%` guardrail. Time-median RMSE worsened by only
  `0.1425%` and passed.

The quantity path remained bitwise identical on the 5080 before and after
fitting. Both runtime prediction digests are
`444e94066bd94d3dfe59410633b2faf3b01f69bbcbfd366b99511d4582d17791`;
quantity MAE/RMSE remained `28.6740223784`/`88.1950014903`. The frozen source
model state was unchanged.

## Decision

The candidate is not accepted as the common time-path improvement. The
controller stopped after Taxi as required. Intermittent and Instacart full
fits, additional seeds, and held-out evaluation were not run. The Taxi result
may be retained as exploratory evidence that conditional calibration can move
proper Time NLL without changing quantity, while the permuted control and
median-MAE result prevent attributing enough robust benefit to the hidden
state under this contract.
