"""Focused diagnostic guards and synthetic inputs only; no real model inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
import torch

from paper.scripts import evaluate_titantpp_frozen_rows as e


class FrozenRowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def contract(self, *, family="component"):
        approval = {"authorized": True, "new_training": False, "frozen_selection": True,
                    "allowed_splits": list(e.SPLITS)}
        e.write_exclusive(self.root / "approval.json", approval)
        arm = "titantpp_cnn_gru54" if family == "component" else "rmtpp"
        job = {"id": "yellow_trip_hourly__42__" + arm, "dataset": "yellow_trip_hourly", "arm": arm,
               "seed": 42, "family": family, "allowed_splits": list(e.SPLITS) if family == "component" else ["train", "validation"],
               "expected_populations": {s: {"count": 3} for s in e.SPLITS},
               "reference_receipts": {s: {"path": "unused", "sha256": "0" * 64} for s in ("validation", "test")}}
        c = {"schema": e.SCHEMA, "new_training": False, "frozen_selection": True,
             "allowed_splits": list(e.SPLITS), "evaluator_sha256": e.sha(e.__file__), "deadline_unix": 200.,
             "approval": {"path": "approval.json", "sha256": e.sha(self.root / "approval.json")}, "jobs": [job]}
        e.write_exclusive(self.root / "contract.json", c)
        return c, job

    def replace(self, c):
        (self.root / "contract.json").write_text(json.dumps(c))
        return e.sha(self.root / "contract.json")

    def test_approval_identity_deadline_and_job_stage_precede_science(self):
        c, j = self.contract()
        digest = e.sha(self.root / "contract.json")
        self.assertEqual(e.verify_contract(self.root, digest, j["id"], "train", now=150.)[1], j)
        for overrides in ({"new_training": True}, {"frozen_selection": False}, {"deadline_unix": 100.},
                          {"evaluator_sha256": "0" * 64}, {"allowed_splits": ["test"]}, {"schema": "foreign"}):
            bad = {**c, **overrides}
            with self.assertRaises(ValueError):
                e.verify_contract(self.root, self.replace(bad), j["id"], "train", now=150.)
        self.replace(c)
        with self.assertRaises(ValueError):
            e.verify_contract(self.root, "0" * 64, j["id"], "train", now=150.)
        with self.assertRaises(ValueError):
            e.verify_contract(self.root, e.sha(self.root / "contract.json"), "foreign", "train", now=150.)
        (self.root / "approval.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "reference bytes"):
            e.verify_contract(self.root, e.sha(self.root / "contract.json"), j["id"], "train", now=150.)

    def test_external_test_new_inference_is_not_silently_authorized(self):
        c, j = self.contract(family="external")
        with self.assertRaisesRegex(ValueError, "not approved"):
            e.verify_contract(self.root, e.sha(self.root / "contract.json"), j["id"], "test", now=150.)

    def test_population_and_metric_conservation_keep_boundary_equalities_lower(self):
        q = np.asarray([0., 2., 3., 7., 11., 13., 19., 25.])
        p = q + np.asarray([1., 2., -1., 3., -2., 4., -3., 5.])
        temporal = np.arange(len(q), dtype=float) / 7
        m = e.RawMetrics([2, 7, 13, 19], np)
        m.add(q[:3], p[:3], temporal[:3])
        m.add(q[3:], p[3:], temporal[3:])
        result = m.finish()
        self.assertEqual([c["count"] for c in result["quantity_cells"]], [2, 2, 2, 1, 1])
        self.assertAlmostEqual(result["qty_sse"], float(((p - q) ** 2).sum()))
        self.assertAlmostEqual(result["qty_bias"], float((p - q).mean()))
        self.assertAlmostEqual(result["qty_rmse"], float(np.sqrt(((p - q) ** 2).mean())))
        self.assertAlmostEqual(sum(c["qty_sse"] for c in result["quantity_cells"]), result["qty_sse"])
        self.assertIsNone(e.RawMetrics([2, 7, 13, 19], np).finish()["qty_rmse"])
        invalid = e.RawMetrics([2, 7, 13, 19], np)
        with self.assertRaises(ValueError):
            invalid.add(np.asarray([1.]), np.asarray([float("nan")]), np.asarray([0.]))

    def test_prediction_parts_preserve_complete_order_and_exclusive_output(self):
        writer = e.PredictionWriter(self.root / "predictions", pl, part_rows=3)
        rows = [{"target_index": n, "target_id": "t" + str(n), "raw_quantity": float(n),
                 "predicted_raw_quantity": float(n + 1), "raw_error": 1., "squared_error": 1.} for n in range(8)]
        writer.add(rows[:4])
        writer.add(rows[4:])
        manifest = writer.finish()
        self.assertEqual([p["rows"] for p in manifest["parts"]], [3, 3, 2])
        self.assertEqual(manifest["prediction_rows"], 8)
        recovered = pl.concat([pl.read_parquet(self.root / "predictions" / p["path"]) for p in manifest["parts"]]).to_dicts()
        self.assertEqual(recovered, rows)
        with self.assertRaises(FileExistsError):
            e.PredictionWriter(self.root / "predictions", pl)
        with self.assertRaises(FileExistsError):
            e.write_exclusive(self.root / "predictions" / "part-00000.parquet", {})

    def test_synthetic_target_block_batch_consistency_and_model_nonmutation(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(.5)
        model.eval().requires_grad_(False)
        before = model.weight.clone()
        dts = torch.tensor([[0., 1., 2., 3.], [0., 0., 2., 1.]])
        mask = torch.tensor([[False, True, True, True], [False, False, True, True]])
        quantities = torch.tensor([[0., 2., 5., 7.], [0., 0., 8., 13.]])
        def causal(m, gaps, keep, q, *, lambda_log_qty):
            prediction = m(q[:, -2:-1]).squeeze(-1)
            return {"pred_qty": prediction, "time_loss": gaps[:, -1] + prediction * .1}
        with torch.inference_mode():
            output = causal(model, dts, mask, quantities, lambda_log_qty=1.)
            checks = e.qualify(torch, model, causal, dts, mask, quantities, output, None)
        self.assertTrue(all(checks.values()))
        self.assertTrue(torch.equal(before, model.weight))
        def leaking(m, gaps, keep, q, *, lambda_log_qty):
            prediction = m(q[:, -1:]).squeeze(-1)
            return {"pred_qty": prediction, "time_loss": gaps[:, -1]}
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "causality"):
            e.qualify(torch, model, leaking, dts, mask, quantities,
                      leaking(model, dts, mask, quantities, lambda_log_qty=1.), None)

    def test_validation_gate_needs_receipt_sha_and_actual_exit0(self):
        _, j = self.contract()
        digest = e.sha(self.root / "contract.json")
        base = self.root / "results" / j["id"]
        receipt = {"status": "complete", "split": "validation", "job_id": j["id"], "contract_raw_sha256": digest,
                   "reference_metrics_match": "PASS"}
        exit_receipt = {"job_id": j["id"], "exit_code": 0, "actual_child_exit_observed": True}
        e.write_exclusive(base / "validation" / "receipt.json", receipt)
        e.write_exclusive(base / "validation_worker_exit.json", exit_receipt)
        gate = {"status": "complete", "contract_sha256": digest, "actual_worker_exit0": True,
                "jobs": {j["id"]: {"receipt_sha256": e.sha(base / "validation" / "receipt.json"),
                                    "worker_exit_sha256": e.sha(base / "validation_worker_exit.json")}}}
        e.write_exclusive(self.root / "validation_gate.json", gate)
        e.verify_validation_gate(self.root, j, digest)
        exit_receipt["exit_code"] = 1
        (base / "validation_worker_exit.json").write_text(json.dumps(exit_receipt))
        gate["jobs"][j["id"]]["worker_exit_sha256"] = e.sha(base / "validation_worker_exit.json")
        (self.root / "validation_gate.json").write_text(json.dumps(gate))
        with self.assertRaisesRegex(ValueError, "exit0"):
            e.verify_validation_gate(self.root, j, digest)

    def test_confinement_and_reference_replay_cannot_hide_a_changed_metric(self):
        with self.assertRaises(ValueError):
            e.confined(self.root, "../outside")
        with self.assertRaises(ValueError):
            e.confined(self.root, "/private/tmp/foreign")
        metrics = e.RawMetrics([2, 7, 13, 19], np)
        metrics.add(np.asarray([1., 25.]), np.asarray([2., 26.]), np.asarray([3., 4.]))
        actual = metrics.finish()
        e.compare_metrics(actual, actual)
        changed = deepcopy(actual)
        changed["tail"]["qty_rmse"] += .01
        with self.assertRaisesRegex(ValueError, "metric differs"):
            e.compare_metrics(actual, changed)
        with self.assertRaisesRegex(ValueError, "population"):
            e.verify_population({"count": 3, "target_identity_sha256": "a"}, {"count": 3, "target_identity_sha256": "b"})

    def test_all_five_bin_sums_and_membership_are_reproduced_even_when_overall_metrics_match(self):
        metrics = e.RawMetrics([2, 7, 13, 19], np)
        metrics.add(np.asarray([1., 5., 11., 17., 25.]), np.asarray([2., 7., 14., 21., 30.]), np.asarray([1., 2., 3., 4., 5.]))
        actual = metrics.finish()
        for mutate, pattern in (
            (lambda r: r["quantity_boundaries"].__setitem__(0, 3.), "boundaries"),
            (lambda r: r["quantity_cells"][0].update(bin=1), "identity/order"),
            (lambda r: r["quantity_cells"][0].update(count=2), "target count"),
            (lambda r: r["quantity_cells"][0].update(qty_sse=2.), "sufficient sum"),
            (lambda r: r["quantity_cells"][0].update(qty_absolute_error_sum=2.), "sufficient sum"),
        ):
            changed = deepcopy(actual)
            mutate(changed)
            with self.assertRaisesRegex(ValueError, pattern):
                e.compare_metrics(actual, changed)
        # Historical external endpoints only store per-bin MAE and SSE.
        historical = deepcopy(actual)
        for cell in historical["quantity_cells"]:
            cell.pop("qty_absolute_error_sum")
        e.compare_metrics(actual, historical)
        historical["quantity_cells"][0]["qty_mae"] = 2.
        with self.assertRaisesRegex(ValueError, "sufficient sum"):
            e.compare_metrics(actual, historical)

    def reused_validation_fixture(self):
        c, j = self.contract()
        base = self.root / "results" / j["id"]
        metrics = e.RawMetrics([2, 7, 13, 19], np)
        metrics.add(np.asarray([1., 5., 25.]), np.asarray([2., 6., 26.]), np.asarray([1., 2., 3.]))
        aggregate = metrics.finish()
        e.write_exclusive(self.root / "prior_validation.json", {"metrics": aggregate})
        j["reference_receipts"]["validation"] = {"path": "prior_validation.json", "sha256": e.sha(self.root / "prior_validation.json")}
        writer = e.PredictionWriter(base / "validation" / "predictions", pl)
        writer.add([{"target_index": index, "raw_quantity": float(index + 1)} for index in range(3)])
        e.write_exclusive(base / "validation" / "predictions_manifest.json", writer.finish())
        qualification = {key: True for key in ("target_quantity_perturbation_preserves_prediction",
            "target_quantity_perturbation_preserves_time_loss", "target_gap_perturbation_preserves_quantity_prediction",
            "single_vs_batch_prediction_matches", "single_vs_batch_time_loss_matches", "parameters_unchanged")}
        old_sha = "old-contract-sha"
        receipt = {"status": "complete", "split": "validation", "job_id": j["id"], "contract_raw_sha256": old_sha,
            "reference_metrics_match": "PASS", **{key: j[key] for key in ("dataset", "arm", "seed", "family")},
            "checkpoint_file_sha256": "checkpoint", "state_sha256_before": "state", "state_sha256_after": "state",
            "selected_epoch": 7, "source_closure": "closure", "data_sha256": "data", "new_training": False,
            "checkpoint_reselection": False, "full_population": True, "raw_predictions_written": True,
            "qualification": qualification, "population": {"count": 3}, "metrics": aggregate,
            "runtime": {"native": "identity"}, "runtime_after": {"native": "identity"}, "prediction_rows": 3,
            "predictions_manifest_sha256": e.sha(base / "validation" / "predictions_manifest.json")}
        exit_receipt = {"job_id": j["id"], "split": "validation", "contract_sha256": old_sha,
                        "exit_code": 0, "actual_child_exit_observed": True}
        e.write_exclusive(base / "validation" / "receipt.json", receipt)
        e.write_exclusive(base / "validation_worker_exit.json", exit_receipt)
        bridge = {"receipt_sha256": e.sha(base / "validation" / "receipt.json"),
                  "worker_exit_sha256": e.sha(base / "validation_worker_exit.json"), "contract_sha256": old_sha}
        c["reused_validation"] = {j["id"]: bridge}
        current_sha = self.replace(c)
        gate = {"status": "complete", "contract_sha256": current_sha, "actual_worker_exit0": True,
                "jobs": {j["id"]: {key: bridge[key] for key in ("receipt_sha256", "worker_exit_sha256")}}}
        e.write_exclusive(self.root / "validation_gate.json", gate)
        spec = {"parent": self.root, "checkpoint": {"sha256": "checkpoint"}, "state_sha": "state",
                "selected_epoch": 7, "source_closure": "closure", "data": {"sha256": "data"}, "runtime": {"approved": True}}
        return c, j, gate, receipt, spec, current_sha

    def test_reused_validation_bridge_requires_explicit_old_contract_and_scientific_identity(self):
        c, j, gate, receipt, spec, current_sha = self.reused_validation_fixture()
        with patch.object(e, "resolve_job", return_value=spec), patch.object(e, "check_runtime") as native:
            e.verify_validation_gate(self.root, j, current_sha)
            native.assert_called_once_with(receipt["runtime"], spec["runtime"])
        # Even if the bridge and gate are repinned, a different model/selection cannot pass.
        for key, value in (("checkpoint_file_sha256", "foreign"), ("state_sha256_after", "mutated"),
                           ("selected_epoch", 9), ("data_sha256", "foreign"), ("arm", "foreign")):
            changed = {**receipt, key: value}
            p = self.root / "results" / j["id"] / "validation" / "receipt.json"
            p.write_text(json.dumps(changed))
            bad_contract = deepcopy(c)
            bad_contract["reused_validation"][j["id"]]["receipt_sha256"] = e.sha(p)
            digest = self.replace(bad_contract)
            bad_gate = deepcopy(gate)
            bad_gate["contract_sha256"] = digest
            bad_gate["jobs"][j["id"]]["receipt_sha256"] = e.sha(p)
            (self.root / "validation_gate.json").write_text(json.dumps(bad_gate))
            with patch.object(e, "resolve_job", return_value=spec), patch.object(e, "check_runtime"), self.assertRaises(ValueError):
                e.verify_validation_gate(self.root, j, digest)

    def test_reused_validation_missing_foreign_bridge_or_modified_bytes_are_rejected(self):
        c, j, gate, receipt, spec, current_sha = self.reused_validation_fixture()
        for altered_reuse in ({}, {"foreign-job": c["reused_validation"][j["id"]]}):
            bad = {**c, "reused_validation": altered_reuse}
            digest = self.replace(bad)
            (self.root / "validation_gate.json").write_text(json.dumps({**gate, "contract_sha256": digest}))
            with self.assertRaises(ValueError):
                e.verify_validation_gate(self.root, j, digest)
        digest = self.replace(c)
        (self.root / "validation_gate.json").write_text(json.dumps({**gate, "contract_sha256": digest}))
        part = self.root / "results" / j["id"] / "validation" / "predictions" / "part-00000.parquet"
        part.write_bytes(b"changed")
        with patch.object(e, "resolve_job", return_value=spec), patch.object(e, "check_runtime"), self.assertRaisesRegex(ValueError, "prediction bytes"):
            e.verify_validation_gate(self.root, j, digest)

    def test_restored_empty_layout_sentinel_resolves_the_frozen_source_without_source_edits(self):
        # The frozen helper is executed from its own bytes; it only checks directories.
        helper = Path(__file__).resolve().parents[2] / "simple_lab_test" / "common" / "pathing.py"
        copied = self.root / "source" / "simple_lab_test" / "common" / "pathing.py"
        copied.parent.mkdir(parents=True)
        copied.write_bytes(helper.read_bytes())
        for directory in ("models", "utils"):
            (self.root / "source" / directory).mkdir()
        # Require the empty directory before import; no Python source modification.
        original_sha = e.sha(copied)
        module = e.file_module(copied, "_empty_layout_pathing_test")
        with self.assertRaisesRegex(RuntimeError, "Could not locate"):
            module.resolve_project_root(copied)
        (self.root / "source" / "sample_data").mkdir()
        self.assertEqual(module.resolve_project_root(copied), self.root / "source")
        self.assertEqual(e.sha(copied), original_sha)


if __name__ == "__main__":
    unittest.main()
