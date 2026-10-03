# Frozen B / dual-timescale gradient probe read-only review

Status: source review completed; no dataset rows, checkpoints, model forward, gradients, or remote state were read/executed in this review. Only /tmp/jqt_probe_review.md is written. Findings refer to the initial probe snapshot inspected; other agents own edits.

## Verdict
The byte/state/population boundaries are broadly sound. Make the training-mode/RNG choice explicit before freezing and running. Results remain an endpoint gradient diagnostic, not training-trajectory causality or benchmark performance.

## P1: train-mode primary analysis needs explicitly paired dropout RNG
- File: paper/scripts/probe_time_quantity_gradients.py:84,110,114
- Current primary mode is eval with dropout disabled. This is deterministic, but its batch gradients and clipping coefficients are not the stochastic training gradients that produced the checkpoints.
- Recommend model.train() for primary analysis and a contract-declared dropout_seed + batch index reset immediately before each model/batch. Reset Python, NumPy and Torch with the same per-batch value; write the seed/mode in each record.
- engine.gradient_diagnostics uses one forward for both losses and restores RNG afterward, which is correct for within-model task comparison. Without explicit per-batch resets, introducing train mode would repeat RNG patterns and depend on preceding model initialization.
- Same seed across B/candidate does not by itself imply identical elementwise dropout masks across differing architectures. Describe reproducible paired seeds, not exact shared masks, unless masks are directly checked.
- Eval mode can be a separately labeled sensitivity analysis; do not combine its gradients with train-mode statistics or add uncontracted executions after observing results.

## P1 if interpreting clipping as time-caused quantity suppression: add quantity-only coefficient
- Probe call: paper/scripts/probe_time_quantity_gradients.py:115.
- Read-only engine inspected at paper/scripts/time_quantity_diagnostic.py:159-177 computes full joint global norm including both heads correctly.
- c_joint * ||g_Q,encoder|| alone cannot establish time-specific suppression; Q gradients may already trigger clipping.
- Compute norm_Q_all = sqrt(sum of squared quantity-task group norms), c_Q = min(1, clip/(norm_Q_all+1e-6)); record c_joint/c_Q and both coefficients. Likewise T-only coefficient is useful but optional.
- This is an analytic counterfactual at the same weights, not an optimizer update or an estimate of an AdamW training trajectory. Direction cancellation can make c_joint/c_Q > 1; do not clamp the ratio or presuppose suppression.

## P2: certify actual imported source modules
- File: paper/scripts/probe_time_quantity_gradients.py:56-80.
- HEAD and declared source bytes are checked; sys.path prioritization alone cannot rule out previously cached modules when run() is imported into another process.
- Require resolved module.__file__ paths for core, loader, source model builder, factory, CountAwareTPP, current model classes and canonical-state helper to be within the pinned root and match declared digests. Record origins.
- The fresh CLI is expected to resolve correctly, so this is a provenance hardening finding, not an observed wrong-source bug.
- Ensure contract source file inventory also covers imported run_taxi_quantity_interface_ablation.py, run_matched_frozen_lognormal_duration.py and simple_lab_test/search/common/runner.py, not just model and loader directories.

## Verified design properties
- population(), lines33-42 matches the canonical identity format in paper/scripts/run_count_aware_tpp_backbone_control.py:254-267: prefix, split, ordered part list, little-endian int64 part/position/seq arrays and little-endian float64 quantity arrays.
- The frozen loader stores val_lists as float32, and both canonical and probe quantity hashes convert those values to float64; this matches the historical contract rather than changing the numeric population convention.
- Lazy train filtering precedes collect at line92. make_loader uses target_splits={train}; parts and target indices are generated deterministically. Population exact equality at line97 refuses changed train membership/order.
- One prebuilt sampled batch list is reused for both models, fixing target pairing and batch ordering. The full population hash plus stored ordered indices and index hash establish sample identity. No outcome-dependent resampling occurs.
- File SHA before deserialization, weights_only=True, route/variant/epoch checks, strict source model construction and canonical loaded-state digest provide strong selected-checkpoint identity.
- build_source_model restores the explicit source cap/time settings from encoder_config/interface_meta. Current default cap30 should not enter the source reconstruction.
- State and checkpoint hashes are rechecked after diagnostics; autograd.grad does not populate .grad. The engine restores requires_grad and RNG, checks nonfinite values and rejects state mutation.

## Additional bounded improvements
- Assert payload selection monitor/source revision/max sequence and model non-state hyperparameters against the contract, even though pinned binary SHA already binds the known payload.
- Record failure status before exit so partially written result folders cannot look merely running indefinitely; refusing overwrite remains appropriate.
- Use a predetermined near-zero-norm applicability rule (or explicitly report very small norms) before interpreting cosine; legacy saturation can make normalized direction unstable. Do not set a threshold based on observed outcomes.
- Tests should use synthetic loader populations/targets and verify B/task-output parity with frozen core, causal target masking, deterministic per-batch seeds, unchanged state/RNG, wrong split/SHA/source failures, and global-clip counterfactual arithmetic. No actual data needed for these tests.
