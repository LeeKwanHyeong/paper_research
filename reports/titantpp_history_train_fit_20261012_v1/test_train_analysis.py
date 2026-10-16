"""Focused fail-closed and sufficient-statistic tests; no inference or SSH."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("selected_train_fit_analysis", ROOT / "paper/scripts/analyze_titantpp_history_train_fit.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
CONTRACT = json.loads((ROOT / "search_artifacts/titantpp_history_train_fit_20261012_v1/contract.json").read_bytes())


def metrics(count, sse, sae, signed, nll):
    return {"count": count, "qty_sse": sse, "qty_absolute_error_sum": sae,
        "qty_signed_error_sum": signed, "time_nll_sum": nll,
        "qty_rmse": math.sqrt(sse/count), "qty_mae": sae/count,
        "qty_bias": signed/count, "time_nll": nll/count}


def cell(key, n, sse, sae, signed, nll):
    m = metrics(n, sse, sae, signed, nll)
    return {"key": key, "count": n,
        "sums": {"squared_error": sse, "absolute_error": sae, "signed_error": signed, "time_nll": nll},
        **{k: m[k] for k in ("qty_rmse", "qty_mae", "qty_bias", "time_nll")}}


class SelectedIdentityTests(unittest.TestCase):
    def test_current_exact_contract_passes(self):
        A.validate_contract(CONTRACT, A.CONTRACT)

    def test_foreign_canonical_contract_rejected(self):
        foreign = copy.deepcopy(CONTRACT)
        foreign["train_fit"]["new_fits"] = 1
        foreign["contract_sha256"] = A.canonical({k:v for k,v in foreign.items() if k != "contract_sha256"})
        with self.assertRaisesRegex(ValueError, "Pinned"):
            A.validate_contract(foreign, A.CONTRACT)

    def test_only_same_selected_weights_join(self):
        job = CONTRACT["jobs"][0]
        row = {k:job[k] for k in ("id", "dataset", "arm", "seed", "selected_epoch")}
        row.update(checkpoint_sha256=job["checkpoint"]["sha256"], state_sha256=job["checkpoint_state_sha256"], source_closure=A.SOURCE)
        A.selected_identity_equal(job, row)
        row["state_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "tensors"):
            A.selected_identity_equal(job, row)

    def test_incomplete_collection_writes_no_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"contract.json").write_text(json.dumps(CONTRACT))
            with self.assertRaises(FileNotFoundError):
                A.analyze(root, root/"missing_prior", A.CONTRACT)
            self.assertFalse((root/"analysis_claim.json").exists())
            self.assertFalse((root/"analysis/completion_receipt.json").exists())


class FitAccountingTests(unittest.TestCase):
    def test_split_gap_never_declares_pure_overfitting(self):
        train = {**metrics(10, 100, 10, -10, 2), "tail":metrics(2,50,8,-8,1)}
        test = {**metrics(4,400,30,-30,80), "tail":metrics(1,300,15,-15,20)}
        gap = A.split_gap(train, test)
        self.assertAlmostEqual(gap["overall"]["qty_rmse"], 10-math.sqrt(10))
        self.assertTrue(gap["different_target_populations"])
        self.assertFalse(gap["pure_overfitting_established"])
        self.assertFalse(gap["new_independent_generalization_established"])

    def test_mse_cancellation_uses_whole_denominator(self):
        full = [cell("body",9,9,9,0,9), cell("tail",1,100,10,-10,4)]
        controls = [cell("body",9,54,18,0,18),cell("tail",1,64,8,-8,1)]
        d = A.D.contrast_cells(full, controls, 10)
        self.assertAlmostEqual(d["net_whole_mse_contribution"], -0.9)
        self.assertAlmostEqual(d["positive_gross_whole_mse"],3.6)
        self.assertAlmostEqual(d["negative_gross_whole_mse_magnitude"],4.5)
        self.assertFalse(d["rmse_is_additive"])

    def test_unpaired_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            A.D.contrast_cells([cell("tail",1,100,10,-10,1)], [cell("tail",2,100,10,-10,1)], 10)

    def test_population_uses_target_means_without_log_reconstruction(self):
        t = {"quantity_boundaries":[10], "strata": {"all":[{"sums":{"true_quantity":30.,"predicted_quantity":20.}}],
            "quantity_bin":[{"key":"0","count":1,"sums":{"true_quantity":5.,"predicted_quantity":6.}},
                            {"key":"1","count":1,"sums":{"true_quantity":25.,"predicted_quantity":14.}}]}}
        m = {**metrics(2,122,12,-10,2),"tail":metrics(1,121,11,-11,1)}
        p = A.population_description(t,m)
        self.assertEqual(p["tail_share"],.5)
        self.assertEqual(p["true_quantity_mean"],15.)
        self.assertEqual(p["tail_true_quantity_mean"],25.)
        self.assertEqual(p["tail_predicted_quantity_mean"],14.)
        self.assertAlmostEqual(p["tail_mse_whole_denominator_contribution"],60.5)
        self.assertEqual(p["tail_bias_squared_share_of_tail_mse"],1.)
        self.assertEqual(p["tail_centered_error_population_variance"],0.)
        self.assertFalse(p["population_is_cross_split_identical"])


class QuantityBaselineTests(unittest.TestCase):
    def make_baseline(self):
        b = metrics(2,8,4,-4,0)
        b.pop("time_nll");b.pop("time_nll_sum")
        b["tail"] = copy.deepcopy(b)
        b["quantity_cells"] = [copy.deepcopy(b)]
        b["quantity_boundaries"] = [10]
        return b

    def test_input_only_baseline_has_no_dummy_time(self):
        b = self.make_baseline()
        A.raw_quantity_baselines({"count":2,"tail":{"count":2},"quantity_boundaries":[10]},
                                 {"last":b,"mean":copy.deepcopy(b)}, {"absolute":1e-7,"relative":1e-9})

    def test_dummy_zero_time_rejected(self):
        b = self.make_baseline();b["time_nll"]=0
        with self.assertRaisesRegex(ValueError,"fabricated"):
            A.raw_quantity_baselines({"count":2,"tail":{"count":2},"quantity_boundaries":[10]},
                                     {"last":b,"mean":b},{"absolute":1e-7,"relative":1e-9})

    def test_corrupt_baseline_moment_rejected(self):
        b = self.make_baseline();b["qty_rmse"]=5
        with self.assertRaisesRegex(ValueError,"Statistic differs"):
            A.raw_quantity_baselines({"count":2,"tail":{"count":2},"quantity_boundaries":[10]},
                                     {"last":b,"mean":b},{"absolute":1e-7,"relative":1e-9})


class PopulationDistributionTests(unittest.TestCase):
    def test_pinned_prior_population_is_inference_free(self):
        p = A.read_population_diagnosis(CONTRACT)
        self.assertEqual(p["model_inference_calls"], 0)
        self.assertEqual(p["checkpoint_loads"], 0)

    def test_distribution_target_identity_is_not_optional(self):
        desc = {"count":2,"tail_count":1,"true_quantity_mean":15.,"tail_true_quantity_mean":25.}
        native = {"target_identity_sha256":"a"*64,"truth_history_sha256":"b"*64}
        q = {"count":2,"mean":15.,"population_sd":10.,"minimum":5.,"maximum":25.}
        tail = {"count":1,"mean":25.,"population_sd":0.,"minimum":25.,"maximum":25.}
        old = {"population_identity":{"target_count":2},"native_target_identity_sha256":"a"*64,
               "native_truth_history_sha256":"b"*64,"quantity":q,"quantity_bin_target_statistics":[{"true_quantity":tail}]}
        A.add_population_distribution(desc,native,old,"fixture","train")
        self.assertEqual(desc["true_quantity_population_variance"],100.)
        self.assertEqual(desc["distribution_source_sha256"], A.POPULATION)
        old["native_truth_history_sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"target-history identity"):
            A.add_population_distribution(desc,native,old,"fixture","train")


if __name__ == "__main__":
    unittest.main()
