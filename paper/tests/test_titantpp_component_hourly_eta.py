"""Stdlib-only tests; no project write, SSH, CUDA, dataset or checkpoint access."""
from __future__ import annotations

from copy import deepcopy
import unittest

from paper.scripts import titantpp_component_hourly_eta as eta


def payloads(count, *, metrics=None, elapsed=None):
    metrics = metrics if metrics is not None else [10.] * count
    elapsed = elapsed if elapsed is not None else [2.] * count
    return ({"history": [{"epoch": i + 1, "val_qty_rmse": value} for i, value in enumerate(metrics)]},
            {"scope": eta.TIMING_SCOPE,
             "epochs": [{"epoch": i + 1, "elapsed_seconds": value} for i, value in enumerate(elapsed)]})


class FitEtaTests(unittest.TestCase):
    def estimate(self, count, **kwargs):
        history, timing = payloads(count, **kwargs)
        return eta.estimate_fit_eta(history, timing, count)

    def test_strict_earliest_finite_minimum_retains_ties(self):
        result = self.estimate(6, metrics=[float("nan"), 8., 3., 3., 4., float("inf")])
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(result["best_epoch"], 3)
        self.assertEqual(result["early_stop_target_epoch"], 43)
        self.assertEqual(result["conditional_early_stop_seconds"], 74.)
        self.assertEqual(result["conditional_maximum_seconds"], 588.)

    def test_epoch_sort_uses_earliest_epoch_for_ties_and_recent_pace(self):
        history, timing = payloads(3, metrics=[5., 1., 1.], elapsed=[1., 2., 3.])
        history["history"].reverse()
        timing["epochs"].reverse()
        result = eta.estimate_fit_eta(history, timing, 3)
        self.assertEqual((result["best_epoch"], result["median_epoch_seconds"]), (2, 2.))

    def test_only_last_ten_completed_epochs_determine_median(self):
        elapsed = [999999.] * 10 + list(range(11, 21))
        result = self.estimate(20, elapsed=elapsed)
        self.assertEqual(result["recent_epoch_count"], 10)
        self.assertEqual(result["median_epoch_seconds"], 15.5)

    def test_before_ten_epochs_uses_only_this_fits_available_completed_rows(self):
        result = self.estimate(3, elapsed=[2., 9., 4.])
        self.assertEqual((result["recent_epoch_count"], result["median_epoch_seconds"]), (3, 4.))

    def test_fixed_minimum_and_maximum_target_formula(self):
        self.assertEqual(eta._early_stop_target_epoch(0), 40)
        self.assertEqual(eta._early_stop_target_epoch(1), 41)
        self.assertEqual(eta._early_stop_target_epoch(260), 300)
        self.assertEqual(eta._early_stop_target_epoch(290), 300)

    def test_late_best_caps_early_target_at300(self):
        result = self.estimate(280, metrics=[9.] * 279 + [1.], elapsed=[3.] * 280)
        self.assertEqual(result["best_epoch"], 280)
        self.assertEqual(result["early_stop_target_epoch"], 300)
        self.assertEqual(result["conditional_early_stop_seconds"], 60.)
        self.assertEqual(result["conditional_maximum_seconds"], 60.)

    def test_no_negative_remaining_and_zero_is_not_actual_exit_evidence(self):
        result = self.estimate(50)
        self.assertEqual(result["conditional_early_stop_seconds"], 0.)
        self.assertEqual(result["status"], "estimated")
        self.assertIn("best", result["condition"])
        self.assertIsNone(result["whole_queue_eta"])
        maximum = self.estimate(300)
        self.assertEqual(maximum["conditional_maximum_seconds"], 0.)

    def test_no_saved_epoch_does_not_borrow_another_fits_speed(self):
        result = eta.estimate_fit_eta({"history": []}, {"scope": eta.TIMING_SCOPE, "epochs": []}, 0)
        self.assertEqual(result["status"], "unconfirmed")
        self.assertIsNone(result["median_epoch_seconds"])
        self.assertIsNone(result["conditional_early_stop_seconds"])

    def test_saved_epoch_type_range_and_bool_negatives(self):
        history, timing = payloads(1)
        for saved in (None, True, -1, 1., "1", 301):
            with self.subTest(saved=saved):
                self.assertEqual(eta.estimate_fit_eta(history, timing, saved)["status"], "unconfirmed")

    def test_missing_history_timing_or_scope_returns_unconfirmed(self):
        history, timing = payloads(1)
        for h, t in ((None, timing), ({}, timing), (history, None), (history, {}),
                     (history, {**timing, "scope": "batch_only"})):
            self.assertEqual(eta.estimate_fit_eta(h, t, 1)["status"], "unconfirmed")

    def test_history_or_timing_saved_epoch_count_mismatch(self):
        history, timing = payloads(2)
        self.assertEqual(eta.estimate_fit_eta(history, timing, 1)["status"], "unconfirmed")
        timing["epochs"].pop()
        self.assertEqual(eta.estimate_fit_eta(history, timing, 2)["status"], "unconfirmed")

    def test_zero_based_duplicate_gap_and_noninteger_epoch_rejected(self):
        for labels in ([0, 1], [1, 1], [1, 3], [1, True], [1, 2.]):
            history, timing = payloads(2)
            for rows in (history["history"], timing["epochs"]):
                for row, epoch in zip(rows, labels):
                    row["epoch"] = epoch
            self.assertEqual(eta.estimate_fit_eta(history, timing, 2)["status"], "unconfirmed")

    def test_invalid_all_validation_metrics_do_not_select_best(self):
        for value in (None, float("nan"), float("inf"), -1., True, "3.0"):
            result = self.estimate(1, metrics=[value])
            self.assertEqual(result["status"], "unconfirmed")
            self.assertIsNone(result["best_epoch"])

    def test_missing_nonfinite_zero_negative_or_bool_timing_rejected(self):
        for value in (None, float("nan"), float("inf"), -1., 0., True, "3.0"):
            result = self.estimate(2, elapsed=[2., value])
            self.assertEqual(result["status"], "unconfirmed")
            self.assertEqual(result["recent_epoch_count"], 0)
            self.assertIsNone(result["conditional_maximum_seconds"])

    def test_partial_epoch_not_used_even_if_elapsed_was_recorded(self):
        history, timing = payloads(2)
        timing["epochs"][-1]["finished_at_unix"] = None
        self.assertEqual(eta.estimate_fit_eta(history, timing, 2)["status"], "unconfirmed")
        history, timing = payloads(2)
        timing["epochs"].append({"epoch": 3, "elapsed_seconds": 100., "finished_at_unix": None})
        self.assertEqual(eta.estimate_fit_eta(history, timing, 2)["status"], "unconfirmed")

    def test_eta_overflow_is_unconfirmed(self):
        self.assertEqual(self.estimate(1, elapsed=[1e308])["status"], "unconfirmed")
        self.assertEqual(self.estimate(1, elapsed=[10 ** 1000])["status"], "unconfirmed")

    def test_helpers_do_not_mutate_supplied_snapshot(self):
        history, timing = payloads(3)
        before = deepcopy((history, timing))
        eta.estimate_fit_eta(history, timing, 3)
        self.assertEqual((history, timing), before)


if __name__ == "__main__":
    unittest.main()
