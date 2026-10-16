"""Focused synthetic checks: no real model, dataset, remote call, or inference."""
from pathlib import Path
import copy
import importlib.util
import math
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


analysis = module(ROOT / "paper/scripts/analyze_titantpp_history_diagnostics.py", "diagnostic_analysis_toy")
telemetry = module(ROOT / "paper/scripts/titantpp_history_diagnostic_telemetry.py", "diagnostic_telemetry_toy")


def fixture(errors):
    collector = telemetry.AggregateDiagnostics([1., 2., 31., 187.], [2., 4.], [1., 2.], [1., 3.], 4)
    for i, (quantity, error) in enumerate(zip([1., 1., 188., 188.], errors)):
        collector.add(entity="a" if i < 2 else "b", calendar="2026-10", last_input_calendar="2026-09",
            history_length=3 if i < 2 else 5, unclipped_history_length=3 if i < 2 else 7,
            truncated=i >= 2, series_in_validation=i < 2, true_quantity=quantity,
            predicted_quantity=quantity + error, time_nll=[.2, .4, 1., 2.][i],
            recorded_gap=2., last_input_gap=1., last_quantity=1., mean_quantity=1.,
            features={"duration_location": float(i), "duration_scale": 1.,
                      "gru_correction_base_ratio": float(i) / 10})
    return collector.finish()


def native(aggregate):
    all_metrics = analysis.metrics_of_cell(aggregate["strata"]["all"][0])
    bins = {x["key"]: x for x in aggregate["strata"]["quantity_bin"]}
    quantity_cells = [{"bin": i, **analysis.metrics_of_cell(bins[str(i)])} if str(i) in bins else
                      {"bin": i, "count": 0} for i in range(5)]
    return {"split": "test", **all_metrics, "quantity_boundaries": aggregate["quantity_boundaries"],
            "quantity_cells": quantity_cells, "tail": analysis.metrics_of_cell(bins["4"])}


class AggregateTests(unittest.TestCase):
    def test_negative_cancellation_and_whole_denominator(self):
        f, c = fixture([4., 0., 0., 0.]), fixture([1., 1., 3., 3.])
        result = analysis.contrast_cells(f["strata"]["quantity_bin"], c["strata"]["quantity_bin"], 4)
        self.assertAlmostEqual(result["positive_gross_whole_mse"], 3.5)
        self.assertAlmostEqual(result["negative_gross_whole_mse_magnitude"], 4.5)
        self.assertAlmostEqual(result["net_whole_mse_contribution"], -1.)
        self.assertEqual(result["covered_count"], 4)
        self.assertFalse(result["rmse_is_additive"])
        bykey = {x["key"]: x for x in result["cells"]}
        self.assertAlmostEqual(bykey["0"]["whole_mse_contribution"], 14. / 4.)
        self.assertAlmostEqual(bykey["4"]["whole_mse_contribution"], -18. / 4.)
        # Error variance + squared conditional bias accounts exactly for MSE.
        for cell in result["cells"]:
            self.assertAlmostEqual(cell["whole_mse_contribution"], cell["bias_squared_mse_contribution"]
                                   + cell["centered_error_variance_mse_contribution"])

    def test_count_mismatch_rejected_even_same_keys(self):
        f, c = fixture([4., 0., 0., 0.]), fixture([1., 1., 3., 3.])
        c["strata"]["quantity_bin"][0]["count"] += 1
        with self.assertRaises(analysis.AnalysisError):
            analysis.contrast_cells(f["strata"]["quantity_bin"], c["strata"]["quantity_bin"], 4)

    def test_unmatched_stratum_rejected(self):
        f, c = fixture([4., 0., 0., 0.]), fixture([1., 1., 3., 3.])
        c["strata"]["quantity_bin"][0]["key"] = "unexpected"
        with self.assertRaises(analysis.AnalysisError):
            analysis.contrast_cells(f["strata"]["quantity_bin"], c["strata"]["quantity_bin"], 4)

    def test_complete_partition_native_reproduction(self):
        f = fixture([4., 0., 0., 0.])
        analysis.check_telemetry(f, native(f), {"absolute": 1e-7, "relative": 1e-9})
        self.assertEqual(sum(x["count"] for x in f["strata"]["tail_by_series"]), 2)
        self.assertEqual({x["key"] for x in f["strata"]["history_vs_train_max"]},
                         {"within_train_max", "above_train_max"})

    def test_false_own_metric_match_rejected(self):
        f = fixture([4., 0., 0., 0.])
        expected = native(f)
        f["strata"]["series"][0]["sums"]["squared_error"] += 1
        with self.assertRaises(analysis.AnalysisError):
            analysis.check_telemetry(f, expected, {"absolute": 1e-7, "relative": 1e-9})

    def test_actual_joint_moment_and_constant_feature(self):
        f = fixture([4., 0., 0., 0.])
        cell = f["strata"]["all"][0]
        features = analysis.summarize_features(cell)
        # mean(x*error)-mean(x)*mean(error) = 0 - 1.5*1 = -1.5.
        self.assertAlmostEqual(features["duration_location"]["covariance_with_signed_error"], -1.5)
        self.assertEqual(features["duration_scale"]["population_sd"], 0)
        self.assertEqual(features["duration_scale"]["covariance_with_signed_error"], 0)
        cell["features"]["duration_location"]["covariance_with_signed_error"] = 20.
        with self.assertRaises(analysis.AnalysisError):
            analysis.summarize_features(cell)

    def test_exit0_requires_actual_observation_and_integer(self):
        for value in ({"exit_code": False, "actual_child_exit_observed": True},
                      {"exit_code": 0, "actual_child_exit_observed": False},
                      {"exit_code": 1, "actual_child_exit_observed": True}):
            with self.assertRaises(analysis.AnalysisError):
                analysis.check_exit(value, "toy")
        analysis.check_exit({"exit_code": 0, "actual_child_exit_observed": True}, "toy")

    def test_known_selected_reference_without_new_moments(self):
        f = native(fixture([4., 0., 0., 0.]))
        original = {"split": "test", "count": 4, "tail": {"count": 2}}
        for key in analysis.METRICS:
            original[key] = f[key]
            original["tail"][key] = f["tail"][key]
        analysis.compare_metrics(f, original, {"absolute": 1e-7, "relative": 1e-9}, "toy")

    def test_unsafe_paths_and_infinite_values(self):
        with tempfile.TemporaryDirectory() as root:
            for relative in ("../elsewhere", "/tmp/elsewhere"):
                with self.assertRaises(analysis.AnalysisError):
                    analysis.safe(root, relative)
        for value in (math.inf, math.nan, True):
            with self.assertRaises(analysis.AnalysisError):
                analysis.finite(value, "toy")

    def test_explicit_contract_pin_rejects_changed_or_unpinned_contract(self):
        payload = {"scope": "synthetic immutable contract", "fits": 0}
        pin = analysis.canonical(payload)
        contract = {**payload, "contract_sha256": pin}
        analysis.require_contract_pin(contract, pin)
        for wrong in (None, "", "a" * 63, "A" * 64, "a" * 64):
            with self.assertRaises(analysis.AnalysisError):
                analysis.require_contract_pin(contract, wrong)
        changed = {**contract, "fits": 1}
        with self.assertRaises(analysis.AnalysisError):
            analysis.require_contract_pin(changed, pin)
        # A newly supplied exact pin is allowed, but never inferred from bytes.
        second = {"scope": "synthetic separately sealed repair", "fits": 0}
        second_pin = analysis.canonical(second)
        second["contract_sha256"] = second_pin
        analysis.require_contract_pin(second, second_pin)
        with self.assertRaises(analysis.AnalysisError):
            analysis.require_contract_pin(second, pin)

    def test_intermittent_focus_uses_true_frozen_range(self):
        boundaries = [2., 31., 46., 187.]
        self.assertEqual(analysis.focus_quantity_bin("intermittent_frozen_5000", boundaries), "1")
        self.assertEqual(analysis.quantity_bin_range(boundaries, "1"), "2 < q ≤ 31")
        self.assertEqual(analysis.quantity_bin_range(boundaries, "2"), "31 < q ≤ 46")
        # A different indexed layout still selects the same physical interval.
        self.assertEqual(analysis.focus_quantity_bin("intermittent_frozen_5000", [1., 2., 31., 187.]), "2")
        with self.assertRaises(analysis.AnalysisError):
            analysis.focus_quantity_bin("intermittent_frozen_5000", [2., 30., 46., 187.])


if __name__ == "__main__":
    unittest.main()
