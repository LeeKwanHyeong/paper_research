# Completed A100 candidates: fixed Validation / Test reevaluation

This directory binds the two completed A100 campaigns separately:

| Frozen campaign | Models | Datasets and seeds |
| --- | --- | --- |
| `mlp_candidates` | `titantpp_history_cross_product`, `titantpp_history_recent4_mean` | Taxi, RAF, Intermittent; seed42 |
| `routing_placement` | `titantpp_history_recent4_attention`, `titantpp_history_post_block` | Taxi, RAF, Intermittent; seed42 |

There are twelve selected checkpoints. Instacart was **not scheduled** for these
four models. No new training or checkpoint selection occurs. The first campaign's
Intermittent rows bind final `resume/run/` evidence, including the original epochs;
the old two-epoch interrupted records remain preserved.

`prepare.py` verifies original terminal, checkpoint, history, endpoint replay and
frozen source SHA identities. `evaluate.py` is a byte-identical copy of the sealed
legacy common-head evaluator: both A100 campaigns already register their models
in their own frozen Factory, so no hook or source modification is required.

The approved inference host is the existing 5080, in the parent scope's dedicated
root plus `/a100_workspace`. This is cross-device validation replay followed by
existing Test reevaluation, **not reproduction of the native A100 Runtime**.
Original Runtime/source/selection provenance remains attached to each row.

## Execution boundary

Preparation and CPU checkpoint identity checks are local. Remote inference must
wait for the parent's explicit release after the width16 campaign finishes.
The parent commits the prepared code/contracts before deployment. `run_all.py`
has no SSH, rental, training, resume, retry, or checkpoint-writing operation.

After deployment preserving relative paths, a static gate may be run with:

```sh
python run_all.py --root <approved-a100-workspace>
```

Only the parent's released run adds `--execute`. This requires the specified
5080 UUID, approved Python, a free GPU, one owned worker, and immutable inputs.
An exception stops only its own evaluation process group and preserves failure
evidence. Existing outputs cannot be silently retried or reused. Limits are
inherited from `../scope_and_approval.json` and enforced during execution.

All twelve **full-population Validation** predictions must reproduce original
selected overall and tail MAE/RMSE/Time NLL within the frozen tolerance before any
Test prediction is produced. Test uses the same original full data files and
chronological one-step observed-history policy as the prior common evaluation.
RAF uses its bound full parent dataset, not its train/validation export.

`verify.py` independently reads bounded prediction parts, recomputes file,
target-identity and truth SHA, and aggregates event-weighted errors. Tail means
raw target quantity strictly above the training-derived threshold. Test rows are
paired with the preserved width4 seed42 population and loader; no row intersection,
post-hoc filtering or changed history is allowed. These identities also permit
the parent's comparison with all already-verified seed42 benchmarks.

Results report single-seed point estimates; sample SD is unavailable. They are
exploratory reevaluations of a previously accessed Test split, not independent
untouched confirmation or efficiency measurements. Any additional uncertainty
analysis belongs to the parent's separately verified paired analysis. Selected
checkpoint CPU identity checks do not imply an audit of last-checkpoint binaries.

## Outputs

- `evaluation_registry.json`, `selected_binding.json`: fixed selected identities.
- `comparison_binding.json`: preserved width4 seed42 Test prediction SHA and population.
- `execution_contract.json`, `deployment_manifest.json`: approval, resources, code/source identities.
- `local_checkpoint_preflight.json`: original selected state SHA and strict CPU loading.
- `runs/<split>/<condition>/`: raw evaluator receipt, prediction parts, resource/gate receipts.
- `validation_gate.json`: all twelve gates, completed before Test.
- `results.json`, `completion_receipt.json`: all twenty-four split rows when complete.
- `campaign_failure.json`: preserved partial progress and exact blocking failure, if any.
