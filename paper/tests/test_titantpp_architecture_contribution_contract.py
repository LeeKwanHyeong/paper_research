"""Offline contribution-contract guards; no model, population, or remote execution."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "search_artifacts/titantpp_architecture_contribution_contract_20261006_v1"
SCRIPT = BUNDLE / "control/validate_contract.py"
spec = importlib.util.spec_from_file_location("contribution_contract_under_test", SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ContributionContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bundle = Path(self.tmp.name) / "bundle"
        shutil.copytree(BUNDLE, self.bundle)
        self.contract = self.read("design_contract.json")
        self.pointer = self.read("current.json")

    def read(self, name):
        return json.loads((self.bundle / name).read_text())

    def write(self, name, value):
        (self.bundle / name).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        )

    def save_rehashed(self, **documents):
        """Pass integrity gates so negative cases actually exercise semantics."""
        self.write("design_contract.json", self.contract)
        self.pointer["contract_canonical_sha256"] = validator.canonical(self.contract)
        for name, value in documents.items():
            self.write(name + ".json", value)
            self.pointer[name + "_sha256"] = hashlib.sha256(
                (self.bundle / (name + ".json")).read_bytes()
            ).hexdigest()
        self.write("current.json", self.pointer)

    def rejected(self, message=None, **documents):
        self.save_rehashed(**documents)
        with self.assertRaises(validator.ContractError) as error:
            validator.validate_contract(ROOT, self.bundle)
        if message:
            self.assertIn(message, str(error.exception))

    def test_valid_contract_is_only_a_local_design_not_native_qualification(self):
        with mock.patch("subprocess.Popen", side_effect=AssertionError("No process launch")), \
             mock.patch("socket.socket", side_effect=AssertionError("No network")):
            result = validator.validate_contract(ROOT, self.bundle)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["scope"], "offline_contract_integrity_and_static_design_only")
        self.assertEqual(result["source_files_SHA_verified"], 123)
        self.assertEqual(result["planned_fits"], 54)
        self.assertEqual(result["new_training_calls"], 0)
        self.assertEqual(result["remote_calls"], 0)
        self.assertEqual(result["native_model_qualification"], "not_performed")
        self.assertEqual(result["new_independent_Test"], "not_performed")
        self.assertEqual(result["check_count"], len(result["checks"]))

    def test_modified_contract_with_stale_pointer_is_rejected(self):
        self.contract["representative"]["width"] = 12
        self.write("design_contract.json", self.contract)
        with self.assertRaisesRegex(validator.ContractError, "contract_canonical_pointer"):
            validator.validate_contract(ROOT, self.bundle)

    def test_representative_cannot_be_selected_from_test(self):
        self.contract["representative"]["selection_split"] = "Test"
        self.rejected("representative_Validation_only")

    def test_width_cannot_switch_to_test_favored_mlp12(self):
        self.contract["representative"]["width"] = 12
        self.contract["representative"]["backbone"] = "titantpp_history_mlp_width12"
        self.rejected("representative_Validation_only")

    def test_selection_must_use_the_dedicated_validation_file(self):
        selection = self.read("selection_receipt.json")
        selection["source_path"] = selection["source_path"].replace(
            "VALIDATION_TABLES.md", "metrics_aggregated.csv"
        )
        self.rejected("dedicated_Validation_source", selection_receipt=selection)

    def test_selection_metric_cannot_change_to_test_or_time_metric(self):
        baseline = self.read("selection_receipt.json")
        for metric in ("mean_3seed_Test_raw_quantity_RMSE", "mean_3seed_full_Validation_TimeNLL"):
            with self.subTest(metric=metric):
                selection = deepcopy(baseline)
                selection["metric"] = metric
                self.rejected(selection_receipt=selection)

    def test_historical_test_exposure_cannot_be_hidden(self):
        selection = self.read("selection_receipt.json")
        selection["previous_Test_exposure"] = False
        selection["independent_preregistration"] = True
        self.rejected("historical_exposure_disclosed", selection_receipt=selection)

    def test_quantity_checkpoint_selection_cannot_become_time_selection(self):
        self.contract["training"]["monitor"] = "validation_time_nll"
        self.rejected("training_monitor_unchanged")

    def test_quantity_primary_objective_cannot_become_joint_time_objective(self):
        self.contract["objective"] = "joint_time_quantity_superiority"
        self.rejected("user_quantity_objective")

    def test_mandatory_time_reporting_cannot_become_optional(self):
        self.contract["success_criteria"]["time_reporting"] = "report_only_if_favorable"
        self.rejected()

    def test_time_epoch_reselection_cannot_hide_quantity_selected_time_harm(self):
        self.contract["evaluation"]["best_time_epoch_reselection_allowed"] = True
        self.rejected("quantity_selector_Test_lock")

    def test_launch_authorizations_are_rejected_even_after_rehash(self):
        baseline = deepcopy(self.contract)
        for permission in ("new_model_implementation", "remote_qualification", "training", "Test_access"):
            with self.subTest(permission=permission):
                self.contract = deepcopy(baseline)
                self.contract["authorization"][permission] = True
                self.rejected("local_only_authorization")

    def test_approval_cannot_enable_new_training(self):
        approval = self.read("approval.json")
        approval["new_training_approved"] = True
        self.rejected("no_launch_permit", approval=approval)

    def test_missing_factorial_arm_is_rejected(self):
        self.contract["arms"] = [arm for arm in self.contract["arms"] if arm["id"] != "P0_H0"]
        self.rejected("six_arms_factorial_and_controls")

    def test_duplicate_canonical_job_cannot_replace_a_missing_condition(self):
        self.contract["jobs"][1] = deepcopy(self.contract["jobs"][0])
        self.rejected("canonical_job_matrix_54_unique")

    def test_one_seed_cannot_be_removed_as_an_outlier(self):
        self.contract["seeds"] = [42, 62]
        self.rejected("all_three_seeds_preserved")

    def test_ordinary_ffn_parameter_budget_cannot_be_inflated(self):
        arm = next(arm for arm in self.contract["arms"] if arm["id"] == "PF_H1")
        arm["prototype_nominal_parameters"] += 64
        arm["intervention_nominal_parameters"] += 64
        self.rejected("arm_PF_H1_real_parameter_budget")

    def test_ordinary_ffn_geometry_must_match_the_real_4096_parameter_budget(self):
        self.contract["architecture"]["prototype_ffn"]["dimensions"] = [64, 64, 64]
        self.rejected("prototype_FFN_same_context_4096")

    def test_dummy_parameter_padding_is_not_a_capacity_control(self):
        arm = next(arm for arm in self.contract["arms"] if arm["id"] == "PF_H1")
        arm["dummy_parameters"] = True
        self.rejected("arm_PF_H1_real_parameter_budget")

    def test_current_state_control_must_keep_the_same_parameter_budget_and_masks(self):
        self.contract["architecture"]["current_control"]["dimensions"] = [64, 8, 64]
        self.rejected("current_state_control_12288_same_masks")

    def test_old_fits_cannot_replace_fresh_paired_causal_cells(self):
        self.contract["jobs"][0]["reuse_old_fit"] = True
        self.rejected("all54_fresh_unlaunched_jobs")

    def test_large_quantity_threshold_cannot_be_changed_after_results(self):
        self.contract["evaluation"]["large_quantity_train_thresholds_strictly_greater_than"]["yellow_trip_hourly"] = 686
        self.rejected("frozen_Train_tail_thresholds")

    def test_matched_contrasts_must_succeed_in_the_same_datasets_not_their_union(self):
        self.contract["success_criteria"]["structural_support"] = (
            "each_required_matched_contrast_negative_RMSE_in_all3seeds_in_atleast2datasets; "
            "descriptive_consistency_not_significance; report_tail_and_MAE_even_if_unfavorable"
        )
        self.rejected("component_external_time_gates_separated")

    def test_native_prototype_tensor_shape_cannot_omit_the_leading_batch_dimension(self):
        self.contract["architecture"]["prototype"]["state_tensor_shape"] = [64, 64]
        self.rejected("static_prototype_scope")

    def test_paired_arm_order_cannot_be_reordered_after_freezing(self):
        block = self.contract["execution_plan"]["paired_blocks"][0]
        block["ordered_arms"][0], block["ordered_arms"][1] = (
            block["ordered_arms"][1], block["ordered_arms"][0]
        )
        self.rejected("nine_paired_block_orders_frozen")


if __name__ == "__main__":
    unittest.main()
