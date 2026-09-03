# Hard-LMM Similarity-Weighted Static Retrieval

## Status and Purpose

Prospective seed-42 backbone candidate, not an adopted improvement. Replace only
the equal average of the original four retrieved prototypes with a cosine-score
softmax weighted sum at fixed temperature 1.0. The observation motivating this
candidate is that equal averaging cannot distinguish different similarities
when the selected set stays unchanged. It does not prove that weighting helps.

The 64 static prototypes and 16 persistent tokens remain outer-loop parameters.
There is no new gate, null-memory path, projection, head, online memory update,
or added parameter. This is not the earlier multi-change B2 and is not Titans-MAC.
The historical `titantpp` implementation and metadata stay unchanged. The new
backbone is `titantpp_weighted_static_memory`, contract identifier `W0`.

## Frozen Comparison

Use the dataset and seed-42 baseline identities in the hash-pinned registry
referenced by the JSON contract. That registry is used only for identity, not to
resume frozen readout, calibration or shrinkage experiments. No baseline is
retrained. New candidate runs initialize all parameters afresh and train the
entire model, not just the output head.

The shared time and quantity paths, direct log1p quantity MSE, legacy clamped
RMTPP time head, Adam, batch 128, lr 0.001, gradient clip 1, validation joint
checkpoint selection, maximum 300/minimum 40/patience 40 remain fixed. Context
lengths and data hashes match each original dataset. The launch time_scale=3
and time_w_max=10/3 are not confused with the separately derived train-time
statistics. Held-out rows are excluded before materialization; train and
validation target counts and quantity strata must match the baseline.

Predict from observed history before the target event. Target quantity is masked
and cannot affect retrieval, weights or prediction. No series-specific online
state exists. Evaluation cannot modify the static prototype or persistent bank.

## Execution and Gates

Target: 5080 only, isolated committed source snapshot and fresh artifact paths.
First run synthetic CUDA contracts, then full train/validation e1 for Taxi, RAF,
Intermittent and Instacart. Only after all four smoke audits pass, run the same
four candidate-only seed-42 e300 experiments. Each dataset uses its own Python
process. Refuse occupied GPU, active desktop, insufficient VRAM, recent kernel
error, source mismatch, or existing output. Never auto-resume a failed run.

All parameters train, but the retrieval operation adds no new parameters. Test
formula/gradient parity for the historical mean path, identical initialization,
same top-k membership, actual score-dependent residual differences, causal masks,
padding/series isolation, finite training and checkpoint replay.

For each dataset, require <=p95 MAE improvement >=5%, overall RMSE and >p99 MAE
regression <=2%, Time NLL increase <=0.01, and finite metrics. Compute body MAE by
count-weighting the three disjoint body strata, not averaging stratum MAEs.
All four gates are needed for a broad follow-up claim; report partial effects
without relabeling them general improvements. Seed 42 is exploratory and cannot
establish robustness or trigger automatic seed expansion. Do not change thresholds
or temperature after validation results. Original artifacts were produced by
older revisions/runtimes: record this limitation; matched settings and a synthetic
legacy regression test are not proof of identical historical optimizer trajectories.

Example (run from an immutable source snapshot with source_manifest.json):

```bash
python -s paper/scripts/run_hard_lmm_weighted_static.py --phase smoke \
  --project-root /home/leekwanhyeong/workspace/paper_research \
  --output-root /home/leekwanhyeong/workspace/paper_research/search_artifacts/hard_lmm_weighted_static_smoke_20260903 \
  --source-revision <committed-40-character-revision>
python -s paper/scripts/run_hard_lmm_weighted_static.py --phase screening \
  --project-root /home/leekwanhyeong/workspace/paper_research \
  --output-root /home/leekwanhyeong/workspace/paper_research/search_artifacts/hard_lmm_weighted_static_seed42_20260903 \
  --smoke-root /home/leekwanhyeong/workspace/paper_research/search_artifacts/hard_lmm_weighted_static_smoke_20260903 \
  --source-revision <same-committed-revision>
```

Maintain hourly monitoring for long runs. Report completion/failure without
automatic retries or model changes. Performance results remain pending until
the complete artifact audit and original-baseline comparison are available.
