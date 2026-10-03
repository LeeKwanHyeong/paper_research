# Independent CLI review: fixed legacy J/Q/T diagnostic

Status: read-only source review completed. Scope: paper/scripts/run_time_quantity_diagnostic.py, related engine/loader code and synthetic tests; no research data deserialization, checkpoint loading, forward/backward or experiments. No repository edits. Line references reflect the inspected snapshot.

## Verdict
No observed train/test leakage or wrong-selector implementation. Same initialization, loader RNGs, fixed complete epoch budgets and matching per-task selectors are designed correctly. The CLI is suitable for a controlled legacy-objective diagnostic after the deterministic execution and source-origin qualifications below are addressed or explicitly limited. It is not a benchmark superiority study, nor a pure test of gradient direction conflict alone.

## Finding 1: deterministic warn-only weakens strict replay claims
- Severity: P2 for present CPU-only local preparation; address before claiming strict CUDA replay.
- Locations: run_time_quantity_diagnostic.py:173-180,250,305-307; imported run_taxi_quantity_interface_ablation.py:175-184.
- configure_runtime and set_seed both select torch.use_deterministic_algorithms(True, warn_only=True). Unsupported nondeterministic kernels are therefore permitted, despite fixed runtime identities and reproducible loader order.
- Use fail-closed deterministic algorithms after every imported set_seed call, or explicitly qualify the entrypoint as CPU-only until deterministic CUDA behavior is tested. Changing only configure_runtime is insufficient because set_seed later restores warn_only=True.
- For CUDA qualification, pin/record relevant device identity and backend settings. The wrapper current_runtime omits GPU identity and several backend flags; engine captures more flags but that does not make arbitrary same-version hardware numerically equivalent.

## Finding 2: verify actual imported source origins
- Severity: P2 provenance hardening; no wrong source was observed.
- Locations: run_time_quantity_diagnostic.py:22-39,94-149,246-251,374.
- AST dependency closure includes conditional and relative imports, useful for capturing first-party source changes. Git revision plus byte-closure equality correctly allows explicitly recorded dirty/new source, without claiming it is identical to the old historical training revision.
- But already imported modules can come from another checkout via sys.modules or preexisting sys.path. Hashing PROJECT_ROOT files does not prove those bytes supplied the loaded classes/functions.
- Require resolved origin paths and digests for factory/model/loader/core/runner/engine under PROJECT_ROOT, including the engine loaded at line374. Fail before data materialization on mixed origins.
- Type-checking branches make current Titan lazy module exports visible to AST traversal; no demonstrated missing active static-B source module was found.

## Default execution boundary
- Lines334-373: --execute defaults false; --resume and --stop-after-epochs require --execute. Default path cannot call run_arm and creates only wrapper_manifest.json.
- However default validation reads full admitted train+validation rows, derives populations/statistics, constructs the initial model and writes the manifest. Describe it as validation without training, not a data-free dry-run. User authorization for train-only gradient diagnosis does not automatically authorize running this full train/validation validation path or the J/Q/T training campaign; keep preparation distinct.
- Synthetic test test_default_validates_without_training_and_excludes_heldout confirms no engine call. No tests were executed by this reviewer.

## Data and held-out boundaries
- load_train_validation_frame filters chronological_split to train/validation before collect and sorts series/sequence (core.py:19-25).
- CLI requires exact data and split SHA from frozen registry, both admitted splits, finite admitted values, unique series/sequence, and no validation before a train target (lines246-281).
- Population equality checks exact train and validation counts/identity/quantity hashes; hashes come from the canonical dataset representation.
- Initialization mean/std uses train rows only. Validation context may use preceding permitted validation observations, matching the frozen conditional next-event evaluation; validation targets do not enter training.
- Test contains synthetic NaN targets which are excluded before validation checks. No historical held-out source was opened in this review.

## J/Q/T initialization, order and budget
- build_arm_inputs resets the same global seed and an independent train loader generator for each arm; a distinct validation generator prevents validation iteration from advancing the training generator (lines305-320).
- Each arm initial full state hash and encoder config are compared against the single recorded initial state (lines386-389).
- Engine resets execution RNG consistently and uses the same encoder path for all objectives. Exclusive inactive heads are removed from AdamW and verified unchanged; no accidental decay of an unused head.
- Every arm receives exactly the contract epochs, batch size and population. Engine has no early stopping and checks all target/batch counts per epoch; optional stop_after_epochs pauses at an unchanged fixed-budget boundary.
- Engine records full batch-content order SHA per epoch. Final integration should assert matching hashes and global steps across corresponding J/Q/T epochs, not merely infer it from initial generator equality. Synthetic CLI tests currently check initialization and first-batch order; engine outputs support the fuller receipt audit.
- Quantity-only versus joint changes both removal of the time objective and the global clipping composition. Any observed benefit is evidence for the combined training interaction, not specifically a direction-conflict mechanism.

## Selector fairness
- Engine applies earliest strict finite raw-RMSE selector to J and Q, and earliest strict finite legacy-time-loss selector to J and T. All arms run the same fixed epoch range.
- Compare J(raw-RMSE selected) with Q(raw-RMSE selected), and J(time-loss selected) with T(time-loss selected). Retain J's two separate selected checkpoints.
- Do not combine J's best quantity metric and its best time metric into a single fictitious joint checkpoint. Use the matching history row to report every available metric at each actual selected checkpoint.
- An inactive head's metric is None, correctly unavailable; random untouched-head predictions are not evaluated as model performance.
- Two independently selected single-task models also are not a single jointly deployed predictor or a claim of equal total compute. This design establishes task-specific attainable performance at a controlled budget.
- Legacy cap300 score remains a compatibility metric. No proper likelihood or cross-benchmark superiority claim follows from passing this experiment.

## Output and resume handling
- Existing wrapper identity cannot be repurposed; all arm destinations are checked before the first arm starts. Existing nonempty arms require resume and an atomic epoch checkpoint.
- Resume preserves exact settings, source identity, inactive heads, optimizer state, RNG and selected checkpoints; epoch extension through the same output is rejected by identity equality.
- Top-level status execution_returned means the engine returned, not necessarily full completion. Only per-arm complete with epochs_completed==epochs_budget and common-step audit can support completion.
