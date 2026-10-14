"""Meaningful raw-diagnostic conservation and strict identity tests; no inference."""
import math
import unittest

import polars as pl

from paper.scripts.analyze_titantpp_frozen_rows import (
    add_groups, align_pair, analyze_pair, canonical_frame, model_metrics,
    paired_group, safe_share, shared_entity_decomposition, truth_digests,
    verify_exit_payload, verify_pipeline_hosts,
    receipt_contract_bridge,
    verify_executable_proof,
    evaluator_sha_for_receipt,
)


def rows(split="test", truth=(1., 3., 40., 250.), prediction=(2., 2., 42., 230.),
         entities=("a", "a", "b", "c")):
    return pl.DataFrame([
        {"target_id": f"id-{split}-{i}", "entity_id": entity, "seq": i + 1,
         "split": split, "recorded_date": f"2026010{i + 1}",
         "raw_quantity": quantity, "recorded_gap": float(i + 1),
         "history_length": (1, 64, 128, 200)[i], "last_quantity": quantity + 1,
         "mean_quantity": quantity - 1 if quantity > 1 else 1.,
         "predicted_raw_quantity": predicted, "log_qty_mse": float(i) / 10}
        for i, (entity, quantity, predicted) in enumerate(zip(entities, truth, prediction))
    ])


def canonical(frame, split="test"):
    return canonical_frame(frame, split=split, bounds=(2, 30, 60, 200))


class FrozenRowAnalysisTests(unittest.TestCase):
    def test_permutation_join_retains_exact_pair_and_sign(self):
        left = canonical(rows())
        right = canonical(rows(prediction=(1., 3., 40., 250.)).reverse())
        joined = align_pair(left, right)
        self.assertEqual(joined.height, 4)
        self.assertAlmostEqual(joined["delta_sq"].sum(), 406.)
        self.assertAlmostEqual(joined["delta_abs"].sum(), 24.)

    def test_duplicate_target_and_truth_change_block_before_analysis(self):
        frame = rows()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            canonical(pl.concat([frame, frame.head(1)]))
        left = canonical(frame)
        right = canonical(frame.with_columns((pl.col("mean_quantity") + .01).alias("mean_quantity")))
        with self.assertRaisesRegex(RuntimeError, "mean_quantity"):
            align_pair(left, right)
        right = left.with_columns(
            pl.col("target_id").str.replace("id", "other"))
        with self.assertRaisesRegex(RuntimeError, "do not coincide"):
            align_pair(left, right)

    def test_saved_error_and_frozen_boundary_mismatch_block(self):
        with self.assertRaisesRegex(RuntimeError, "Saved raw error"):
            canonical(rows().with_columns(pl.lit(0.).alias("squared_error")))
        with self.assertRaisesRegex(RuntimeError, "quantity bin"):
            canonical(rows().with_columns(pl.lit(0).alias("quantity_bin")))
        boundary = canonical(rows(truth=(2., 30., 60., 200.)))
        self.assertEqual(boundary["quantity_bin"].to_list(), [0, 1, 2, 3])

    def test_all_groupings_conserve_mse_mae_bias_sums(self):
        left, right = canonical(rows()), canonical(rows(prediction=(1., 3., 40., 250.)))
        result, entities = analyze_pair(left, right)
        self.assertAlmostEqual(result["mse_gap"], 101.5)
        self.assertAlmostEqual(result["mae_gap"], 6.)
        for groups in result["groups"].values():
            self.assertEqual(sum(g["count"] for g in groups), 4)
            self.assertAlmostEqual(sum(g["population_mse_gap_contribution"] for g in groups), 101.5)
            self.assertAlmostEqual(sum(g["population_mae_gap_contribution"] for g in groups), 6.)
        self.assertEqual(entities.height, 3)

    def test_membership_weights_within_decomposition_conserves_global_change(self):
        lv = canonical(rows("validation", prediction=(2., 4., 43., 255.), entities=("a", "a", "b", "old")), "validation")
        rv = canonical(rows("validation", prediction=(1., 3., 40., 250.), entities=("a", "a", "b", "old")), "validation")
        lt = canonical(rows(prediction=(3., 6., 45., 260.), entities=("a", "b", "b", "new")))
        rt = canonical(rows(prediction=(1., 3., 40., 250.), entities=("a", "b", "b", "new")))
        v = paired_group(align_pair(lv, rv), "entity_id")
        t = paired_group(align_pair(lt, rt), "entity_id")
        result = shared_entity_decomposition(v, t)
        total = sum(result[k] for k in ("test_membership_term", "validation_membership_term",
                                      "shared_entity_weight_mix_term", "shared_entity_within_term"))
        self.assertAlmostEqual(total, result["global_mse_gap_change"])
        self.assertEqual(result["validation_only_entities"], 1)
        self.assertEqual(result["test_only_entities"], 1)
        self.assertEqual(result["shared_entity_count"], 2)

    def test_raw_and_original_log_loss_are_kept_separate(self):
        frame = canonical(rows())
        metrics = model_metrics(frame)
        self.assertAlmostEqual(metrics["fixed_model_log_qty_mse"], .15)
        self.assertAlmostEqual(metrics["qty_rmse"], math.sqrt(101.5))
        legacy = frame.drop("log_qty_mse")
        self.assertIsNone(model_metrics(legacy)["fixed_model_log_qty_mse"])
        self.assertIsNone(safe_share(1, 0))

    def test_null_or_nonfinite_raw_truth_never_enters_metrics(self):
        for quantity in (None, float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                canonical(rows().with_columns(pl.lit(quantity).alias("raw_quantity")))

    def test_digest_binds_full_common_truth_not_only_target_ids(self):
        one = canonical(rows())
        other = one.with_columns((pl.col("last_quantity") + 1).alias("last_quantity"))
        target_one, truth_one = truth_digests(one)
        target_two, truth_two = truth_digests(other)
        self.assertEqual(target_one, target_two)
        self.assertNotEqual(truth_one, truth_two)

    def test_actual_exit_binds_native_contract_worker_and_observation(self):
        good = {"exit_code": 0, "actual_child_exit_observed": True,
                "contract_sha256": "a" * 64, "job_id": "model", "split": "train"}
        verify_exit_payload(good, "a" * 64, job_id="model", split="train")
        for modified in ({**good, "contract_sha256": "b" * 64},
                         {**good, "actual_child_exit_observed": False},
                         {**good, "job_id": "different"}):
            with self.assertRaises(RuntimeError):
                verify_exit_payload(modified, "a" * 64, job_id="model", split="train")

    def test_duplicate_native_pipeline_host_never_satisfies_two_host_gate(self):
        verify_pipeline_hosts([{"host": "5080"}, {"host": "5090"}])
        with self.assertRaises(RuntimeError):
            verify_pipeline_hosts([{"host": "5080"}, {"host": "5080"}])

    def test_child_exit0_cannot_hide_contract_change_or_supervisor_failure(self):
        good = {"exit_code": 0, "actual_child_exit_observed": True,
                "contract_sha256": "a" * 64, "contract_unchanged": True,
                "supervisor_exit_code": 0}
        verify_exit_payload(good, "a" * 64, pipeline=True)
        for modified in ({**good, "contract_unchanged": False},
                         {**good, "supervisor_exit_code": 1}):
            with self.assertRaisesRegex(RuntimeError, "supervisor"):
                verify_exit_payload(modified, "a" * 64, pipeline=True)

    def test_validation_cache_only_accepts_exact_whitelisted_sha(self):
        job = {"id": "model", "family": "component", "host": "5080"}
        entry = {"split": "validation", "receipt": {"sha256": "r" * 64},
                 "worker_exit": {"sha256": "e" * 64}}
        contract = {"reused_validation": {"model": {"receipt_sha256": "r" * 64,
                    "worker_exit_sha256": "e" * 64, "contract_sha256": "o" * 64}}}
        self.assertEqual(receipt_contract_bridge(job, entry, "n" * 64, contract), ("o" * 64, True))
        for modified in ({**entry, "receipt": {"sha256": "x" * 64}},
                         {**entry, "worker_exit": {"sha256": "x" * 64}}):
            with self.assertRaisesRegex(RuntimeError, "exact"):
                receipt_contract_bridge(job, modified, "n" * 64, contract)
        for split in ("train", "test"):
            self.assertEqual(receipt_contract_bridge(job, {**entry, "split": split}, "n" * 64, contract), ("n" * 64, False))
        for wrong_job in ({**job, "family": "external"}, {**job, "host": "5090"}):
            with self.assertRaisesRegex(RuntimeError, "outside"):
                receipt_contract_bridge(wrong_job, entry, "n" * 64, contract)

    def test_unstored_resolved_executable_uses_sealed_guard_and_actual_argv(self):
        source = """
def check_runtime(actual, expected):
    require(Path(sys.executable).resolve() == Path(expected['python']).resolve(), 'fixed executable')
def worker():
    check_runtime(runtime, expected)
    check_runtime(runtime_after, expected)
"""
        result = verify_executable_proof({"python": "/native/python"},
                                         {"argv": ["/native/python", "evaluate.py"]}, source)
        self.assertFalse(result["direct_resolved_path_stored"])
        self.assertTrue(result["before_and_after_guard_in_successful_worker"])
        with self.assertRaisesRegex(RuntimeError, "argv"):
            verify_executable_proof({"python": "/native/python"},
                                     {"argv": ["/different/python", "evaluate.py"]}, source)
        with self.assertRaisesRegex(RuntimeError, "resolution guard"):
            verify_executable_proof({"python": "/native/python"},
                                     {"argv": ["/native/python", "evaluate.py"]},
                                     source.replace("Path(sys.executable).resolve() == Path(expected['python']).resolve()", "True"))
        with self.assertRaisesRegex(RuntimeError, "before and after"):
            verify_executable_proof({"python": "/native/python"},
                                     {"argv": ["/native/python", "evaluate.py"]},
                                     source.replace("    check_runtime(runtime_after, expected)", "    pass"))

    def test_historical_evaluator_source_only_has_exact_cache_authority(self):
        self.assertEqual(evaluator_sha_for_receipt("a" * 64, "a" * 64, authorized_cache=False), "a" * 64)
        self.assertEqual(evaluator_sha_for_receipt("b" * 64, "a" * 64, authorized_cache=True), "b" * 64)
        with self.assertRaisesRegex(RuntimeError, "Fresh"):
            evaluator_sha_for_receipt("b" * 64, "a" * 64, authorized_cache=False)
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            evaluator_sha_for_receipt("../" * 22, "a" * 64, authorized_cache=True)


if __name__ == "__main__":
    unittest.main()
