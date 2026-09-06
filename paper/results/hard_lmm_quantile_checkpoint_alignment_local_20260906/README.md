# Hard-LMM checkpoint alignment: source preflight and final A-vs-B audit

## Revised evaluation scope

- Final comparison: `A_t0_joint` versus `B_t0_raw_rmse` on Seed42 validation data.
- A is the existing Hard-LMM T0 artifact selected by the joint objective. B keeps the same T0 architecture and loss, but trains and early-stops against validation raw RMSE and restores the earliest raw-RMSE minimum.
- The user withdrew C from the final comparison and adoption decision. Completed Intermittent and Taxi C artifacts remain preserved, Instacart C did not start, and none of them are used below.
- The original frozen A/B/C gate is not reinterpreted and no pass is claimed for it. The only final decision is whether B improves raw RMSE over A on every dataset.
- Scope is validation only. Held-out test and seeds52/62 were not run.

## Final decision

The common raw-RMSE goal is **MET** for Seed42 validation: B improves raw RMSE over A on all three datasets. This is evidence for checkpoint-selection alignment, not a new Backbone architecture result. Body MAE, `>p99` MAE, and time NLL are reported as descriptive trade-offs under the revised scope.

| Dataset | A raw RMSE | B raw RMSE | A→B improvement | A body MAE | B body MAE | A `>p99` MAE | B `>p99` MAE | A time NLL | B time NLL | Time NLL Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| intermittent_frozen_5000 | 1.72240325 | 1.49955507 | 12.9382% | 0.60336928 | 0.46578960 | 6.06798283 | 5.14810801 | -3.59285604 | -3.48379369 | +0.10906235 |
| yellow_trip_hourly | 181.53759358 | 88.19499665 | 51.4178% | 23.16740661 | 18.99747812 | 1166.28242126 | 332.76974449 | 1.36643025 | 1.47339063 | +0.10696038 |
| insta_market_basket | 5.97416001 | 5.87221693 | 1.7064% | 3.42454564 | 3.43860150 | 23.00099934 | 21.91779477 | 3.20644001 | 3.21552290 | +0.00908288 |

B also reduces body and tail MAE on Intermittent and Taxi. On Instacart, body MAE increases by 0.4104% while `>p99` MAE decreases by 4.7094%. Time NLL increases on all three datasets, by 0.1091 on Intermittent, 0.1070 on Taxi, and 0.0091 on Instacart. B therefore gives a consistent quantity-RMSE gain, with a material time-likelihood trade-off on Intermittent and Taxi.

## Run and integrity evidence

- Immutable training source: `f75243473adc25d622319dbca9bda7e076d8240f`.
- B-only auditor base commit: `00fb364`; final auditor revision with the direct-CLI project-root bootstrap correction: `de0ffc9`. The audited script SHA-256 is `95f09b036cf90ff5da918f589d3114c4da93859955e8d65a3e8062b0fd710900`.
- Local focused contract tests: **57 passed**; legacy regression tests: **88 passed**; auditor bootstrap subset: **23 passed**.
- 5090 CUDA contract tests: **34 passed**. Full-data e1 completed for B and C on all three datasets before the scope revision.
- The final B-only audit passed: all A and B datasets were audited, each B checkpoint is the earliest validation raw-RMSE minimum, and all B runs used CUDA with the full train and validation populations.
- B run endpoints were Intermittent best epoch 77 / completed epoch 117, Taxi best epoch 45 / completed epoch 85, and Instacart best epoch 72 / completed epoch 112.
- After the Instacart B summary, checkpoint, resume state, and aggregate files existed, the watcher sent `SIGTERM` to the original controller to prevent Instacart C from starting. The controller's preserved outcome is `failed_intentional_sigterm`; the revised A-vs-B scope is `complete_for_revised_scope`.
- Post-stop audit confirms that the controller, trainer, and watcher exited, no associated GPU compute remained, Instacart C did not start, no additional seeds ran, and held-out test data was not evaluated.
- A and B use identical pinned validation target populations per dataset. Source, input, split, checkpoint, summary, history, and stop-record hashes are recorded in the machine-readable audit.

Final audit artifacts are under `paper/results/hard_lmm_raw_rmse_checkpoint_alignment_seed42_20260906/`:

- `b_only_run_audit.json`
- `b_only_scope_completion.json`
- `a_vs_b_metrics.json`
- `a_vs_b_metrics.csv`
- `a_vs_b_metrics.md`
- `post_stop_process_audit.json`

## Historical selector headroom

The pre-run diagnostic below motivated the selector comparison. It did not reuse historical raw-min checkpoints as B; every B result above came from a fresh matched run.

| Dataset | A selected epoch | A raw RMSE | Historical raw-min epoch | Historical raw minimum | Relative headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| intermittent_frozen_5000 | 200 | 1.72240325 | 203 | 1.56542244 | 9.11% |
| yellow_trip_hourly | 2 | 181.53759358 | 35 | 93.49819231 | 48.50% |
| insta_market_basket | 26 | 5.97416001 | 21 | 5.88096406 | 1.56% |
