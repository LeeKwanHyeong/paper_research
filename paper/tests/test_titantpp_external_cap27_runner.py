"""Cap external-encoder runner checks use fabricated data and temporary artifacts only."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import run_titantpp_external_cap27 as runner


def contract_at(root):
    data = {"dataset_id": "raf_spare_parts", "cap_value": 10.,
        "quantity_boundaries_all_train_rows": [3., 10., 10., 10.],
        "model": {"hidden_dim": 64, "quantity_variant": runner.VARIANT,
            "lambda_log_qty": 1., "lambda_tail": 0.,
            "time_head_mode": "heteroscedastic_lognormal_duration"}}
    return {"hosts": {"5090": {"root": str(root)}}, "datasets": [data],
        "jobs": [{"id": "raf__42__rmtpp", "host": "5090", "dataset": "raf_spare_parts",
                  "arm": runner.ARMS[0], "seed": 42}],
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
                     "batch_size": 128, "monitor": runner.MONITOR, "warm_start": False}}


class CapMetricTests(unittest.TestCase):
    def test_capped_ties_empty_strict_tail_and_cap_hit(self):
        acc = runner.MetricAccumulator(10)
        acc.add([2, 10, 10], [1, 8, 12], [.5, 1., 1.5])
        result = acc.finish()
        self.assertEqual(result["overall"]["count"], 3)
        self.assertAlmostEqual(result["overall"]["qty_rmse"], 3 ** .5)
        self.assertAlmostEqual(result["overall"]["qty_mae"], 5/3)
        self.assertEqual(result["tail"]["count"], 0)
        self.assertEqual(result["tail"]["status"], "N/A_empty")
        self.assertIsNone(result["tail"]["time_nll"])
        self.assertEqual(result["cap_hit"]["qty_mae"], 2.)
        self.assertEqual(result["cap_hit"]["qty_rmse"], 2.)
        self.assertEqual(result["cap_hit"]["time_nll"], 1.25)

    def test_prediction_is_not_clipped_to_cap(self):
        acc = runner.MetricAccumulator(10)
        acc.add([10], [30], [1.])
        self.assertEqual(acc.finish()["overall"]["qty_mae"], 20.)

    def test_nonfinite_and_uncapped_truth_are_rejected(self):
        for truth, pred, temporal in (([11], [1], [1]), ([1], [float("nan")], [1]),
                                       ([1], [1], [float("inf")])):
            with self.subTest(truth=truth, pred=pred, temporal=temporal), self.assertRaises(ValueError):
                runner.MetricAccumulator(10).add(truth, pred, temporal)

    def test_metric_replay_handles_empty_tail_and_detects_regression(self):
        acc = runner.MetricAccumulator(10)
        acc.add([2, 10], [1, 8], [.5, 1.])
        a = acc.finish()
        b = json.loads(json.dumps(a))
        runner.compare_metrics(a, b)
        b["overall"]["qty_mae"] += .1
        with self.assertRaises(ValueError):
            runner.compare_metrics(a, b)


class CapContractTests(unittest.TestCase):
    def test_three_original_external_models_only(self):
        self.assertEqual(runner.ARMS, ("rmtpp", "thp", "s2p2_matched_head"))
        self.assertEqual(runner.SCHEMA, "titantpp_external_cap27_stage_v1")
        self.assertEqual(runner.role_for("rmtpp"), "observed_time_joint_v1")
        self.assertEqual(runner.role_for("thp"), "observed_time_joint_v1")
        self.assertEqual(runner.role_for("s2p2_matched_head"), "additional_tpp_matched_head_v1")
        for arm in ("s2p2", "attnhp_matched_head", "titantpp_history_mlp", "titantpp_cnn", "new_encoder"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                runner.role_for(arm)

    def test_duplicate_cap_boundaries_are_admitted(self):
        c = contract_at("/tmp/cap-fixture")
        job, data, root, folder = runner.select_job(c, "5090", c["jobs"][0]["id"])
        self.assertEqual(data["cap_value"], 10.)
        self.assertEqual(folder, root / "run" / job["id"])

    def test_scope_budget_and_path_changes_are_rejected(self):
        changes = (("host", "5080"), ("seed", 32), ("arm", "new_shared_mlp"), ("id", "../escape"))
        for key, value in changes:
            c = contract_at("/tmp/cap-fixture")
            c["jobs"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.select_job(c, "5090", c["jobs"][0]["id"])
        c = contract_at("/tmp/cap-fixture")
        c["training"]["maximum_epochs"] = 301
        with self.assertRaises(ValueError):
            runner.select_job(c, "5090", c["jobs"][0]["id"])

    def test_canonical_contract_hash(self):
        data = {"한글": 1, "b": 2}
        expected = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(runner.sha_json(data), expected)

    def test_exclusive_receipt_and_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p = root / "receipt.json"
            runner.write_exclusive(p, {"status": "complete"})
            with self.assertRaises(FileExistsError):
                runner.write_exclusive(p, {"status": "changed"})
            inv = runner.file_inventory(root, [p])
            runner.verify_inventory(root, inv)
            p.write_text("changed")
            with self.assertRaises(ValueError):
                runner.verify_inventory(root, inv)
            with self.assertRaises(ValueError):
                runner.verify_inventory(root, {})
            with self.assertRaises(ValueError):
                runner.resolve_under(root, "../escape")

    def test_population_keeps_exact_identity_and_quantity(self):
        from types import SimpleNamespace
        dataset = SimpleNamespace(index=[(0, 0), (0, 1)], parts=["part"],
            seq_lists=[[1, 2, 3]], val_lists=[[2., 10., 10.]],
            split_lists=[["train", "test", "test"]])
        q, identity = runner.population_identity(dataset, "test")
        self.assertEqual(q.tolist(), [10., 10.])
        runner.verify_population(identity, {**identity, "split": "test"})
        dataset.val_lists[0][-1] = 9.
        changed = runner.population_identity(dataset, "test")[1]
        self.assertEqual(changed["target_identity_sha256"], identity["target_identity_sha256"])
        with self.assertRaises(ValueError):
            runner.verify_population(changed, identity)

    def test_test_gate_is_checked_before_any_input_or_runtime_access(self):
        with tempfile.TemporaryDirectory() as temp:
            c = contract_at(temp)
            with patch.object(runner, "verify_validation_gate", side_effect=ValueError("no gate")) as gate:
                with self.assertRaisesRegex(ValueError, "no gate"):
                    runner.run_stage(c, "5090", c["jobs"][0]["id"], "test")
                gate.assert_called_once()
            self.assertFalse((Path(temp) / "run" / c["jobs"][0]["id"] / "test.intent.json").exists())

    def test_gate_rejects_partial_host_job_set(self):
        with tempfile.TemporaryDirectory() as temp:
            c = contract_at(temp)
            runner.write_exclusive(Path(temp)/"validation_gate.json", {
                "status": "passed", "host": "5090", "contract_sha256": runner.sha_json(c), "jobs": {}})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.verify_validation_gate(c, "5090", Path(temp))

    def test_gate_checks_complete_validation_train_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = contract_at(root)
            job = c["jobs"][0]
            folder = root / "run" / job["id"]
            folder.mkdir(parents=True)
            checkpoint = folder / "selected.bin"
            checkpoint.write_bytes(b"synthetic checkpoint")
            tr = {**runner._base_receipt(c, "5090", job, "train"),
                "state_sha256": "state", "selected_checkpoint": str(checkpoint),
                "files": runner.file_inventory(root, [checkpoint])}
            runner.write_exclusive(folder/"train_receipt.json", tr)
            vr = {**runner._base_receipt(c, "5090", job, "validation"),
                "state_sha256": "state", "selected_checkpoint": str(checkpoint),
                "validation_replay_passed": True, "files": runner.file_inventory(root, [checkpoint])}
            runner.write_exclusive(folder/"validation_receipt.json", vr)
            runner.write_exclusive(root/"validation_gate.json", {"status": "passed", "host": "5090",
                "contract_sha256": runner.sha_json(c), "jobs": {job["id"]: {
                    "validation_receipt_sha256": runner.sha_file(folder/"validation_receipt.json")}}})
            runner.verify_validation_gate(c, "5090", root)
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                runner.verify_validation_gate(c, "5090", root)

    def test_prediction_replay_rejects_row_and_metric_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = {"target_index": 0, "entity_id": "a", "target_seq": 2, "raw_quantity": 10.,
                   "history_length": 1, "split": "validation", "predicted_raw_quantity": 9., "time_nll": .5}
            def save(path, records):
                with gzip.open(path, "wt") as f:
                    for record in records:
                        f.write(json.dumps(record)+"\n")
            a, b = root/"a.gz", root/"b.gz"
            save(a, [row]); save(b, [row])
            self.assertEqual(runner.compare_prediction_files(a, b), 1)
            save(b, [{**row, "target_seq": 3}])
            with self.assertRaises(ValueError):
                runner.compare_prediction_files(a, b)
            save(b, [{**row, "predicted_raw_quantity": 8.}])
            with self.assertRaises(ValueError):
                runner.compare_prediction_files(a, b)
            save(b, [])
            with self.assertRaises(ValueError):
                runner.compare_prediction_files(a, b)


class CapSyntheticTests(unittest.TestCase):
    def test_inactive_tail_constructor_has_no_effect_on_active_loss(self):
        import torch
        from paper.scripts.count_aware_tpp_backbone.core import target_outputs
        data = runner.fabricated_data()
        dt = torch.tensor([[1., 2., 3., 1.]])
        q = torch.tensor([[1., 3., 8., 10.]])
        mask = torch.ones_like(dt, dtype=torch.bool)
        for arm in runner.ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(42)
                standard, _ = runner.build_model(data, arm)
                with patch.dict(runner.INACTIVE_TAIL, {
                        "tail_threshold": 1., "tail_normalization_scale": 2.,
                        "tail_clip_cap": 20., "tail_huber_delta": .5}):
                    torch.manual_seed(42)
                    alternate, _ = runner.build_model(data, arm)
                self.assertEqual(set(standard.state_dict()), set(alternate.state_dict()))
                for key, value in standard.state_dict().items():
                    self.assertTrue(torch.equal(value, alternate.state_dict()[key]), key)
                standard.eval(); alternate.eval()
                a = target_outputs(standard, dt, mask, q, lambda_log_qty=1.)
                b = target_outputs(alternate, dt, mask, q, lambda_log_qty=1.)
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0., atol=0.)
                a["joint_loss"].sum().backward(); b["joint_loss"].sum().backward()
                for (name, p), (_, other) in zip(standard.named_parameters(), alternate.named_parameters()):
                    if p.grad is None:
                        self.assertIsNone(other.grad, name)
                    else:
                        torch.testing.assert_close(p.grad, other.grad, rtol=0., atol=0.)

    def test_external_identity_positive_integer_head_and_checkpoint_routes(self):
        import torch
        from models.TPPs.CountAwareFactory import validate_checkpoint_route
        from models.TPPs.CountAwareTPP import CountAwareRMTPP, CountAwareTHP
        from models.TPPs.CountAwareAdditionalTPP import CountAwareS2P2
        from paper.scripts.count_aware_tpp_backbone import constants
        classes = {"rmtpp": CountAwareRMTPP, "thp": CountAwareTHP, "s2p2_matched_head": CountAwareS2P2}
        data = runner.fabricated_data()
        for arm, cls in classes.items():
            with self.subTest(arm=arm):
                torch.manual_seed(42)
                model, meta = runner.build_model(data, arm)
                self.assertIs(type(model), cls)
                self.assertEqual(model.quantity_variant, runner.VARIANT)
                self.assertEqual(model.time_head_mode, "heteroscedastic_lognormal_duration")
                self.assertEqual(model.time_observation_contract,
                    {"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None})
                self.assertEqual(model.lambda_tail, 0.)
                constants.validate_model_role_contract(model_role=runner.role_for(arm), backbones=(arm,),
                    quantity_variants=(runner.VARIANT,), time_head_mode=model.time_head_mode, lambda_tail=0.)
                payload = {"backbone": arm, "encoder_config": meta, "variant": runner.VARIANT,
                           "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
                validate_checkpoint_route(payload, arm)
                if arm == "s2p2_matched_head":
                    with self.assertRaises(ValueError):
                        validate_checkpoint_route(payload, "rmtpp")
                    with self.assertRaises(ValueError):
                        validate_checkpoint_route({**payload, "backbone": "rmtpp"}, arm)
        with self.assertRaises(ValueError):
            runner.build_model(data, "titantpp_history_mlp")

    def test_all_three_real_models_on_fabricated_inputs(self):
        result = runner.synthetic_check(device="cpu")
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["real_data_loaded"])
        self.assertFalse(result["held_out_test_evaluated"])
        self.assertEqual(set(result["arms"]), set(runner.ARMS))
        self.assertEqual(result["parameter_count"], {"rmtpp": 25347, "thp": 100355, "s2p2_matched_head": 8003})
        self.assertEqual(result["real_scalar_parameter_count"],
                         {"rmtpp": 25347, "thp": 100355, "s2p2_matched_head": 13155})

    def test_selected_checkpoint_cannot_be_relabelled_as_another_arm(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "foreign.pt"
            for arm in runner.ARMS:
                with self.subTest(arm=arm):
                    foreign = next(other for other in runner.ARMS if other != arm)
                    torch.save({"backbone": foreign}, checkpoint)
                    receipt = {"selected_checkpoint": str(checkpoint),
                               "checkpoint_sha256": runner.sha_file(checkpoint), "selected_epoch": 1}
                    with self.assertRaisesRegex(ValueError, "route/selection"):
                        runner._load_selected(runner.fabricated_data(),
                            {"arm": arm, "seed": 42}, receipt, root, "cpu")

    def test_full_train_validation_test_path_on_fabricated_rows(self):
        """Exercise real loader/train/selection/evaluation with a two-epoch test-only budget."""
        import numpy as np
        import polars as pl
        import torch
        from paper.scripts.quantity_comparison_runtime import configure_runtime
        from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
        from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
        from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
        from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
        configure_runtime("cpu", 4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = contract_at(root)
            c["source"] = {"base_git_revision": "fabricated", "files_sha256": "fixture"}
            c["design_sha256"], c["architecture"] = "fixture", {"models": list(runner.ARMS)}
            c["jobs"] = [{"id": f"fixture_{i}", "host": "5090", "dataset": "raf_spare_parts", "seed": 42, "arm": a}
                         for i, a in enumerate(runner.ARMS)]
            records = [{"oper_part_no": entity, "seq": i+1,
                "delta_t": [0, 1, 2, 1, 3, 1, 2, 1][i], "demand_qty": float([1, 3, 8, 10, 10, 2, 4, 10][i]),
                "chronological_split": "train" if i < 4 else "validation" if i < 6 else "test"}
                for entity in ("fabricated_a", "fabricated_b") for i in range(8)]
            raw = pl.DataFrame(records)
            tv = raw.filter(pl.col("chronological_split") != "test")
            full_path, tv_path, manifest = root/"full.parquet", root/"tv.parquet", root/"manifest.json"
            raw.write_parquet(full_path); tv.write_parquet(tv_path)
            manifest.write_text('{"fabricated":true}')
            data = c["datasets"][0]
            data["loader"] = {"batch_size": 128, "lookback_weeks": 84, "max_seq_len": 84,
                              "num_workers": 0, "drop_last": False, "train_shuffle": True, "validation_shuffle": False}
            populations, full_populations = {}, {}
            for split in ("train", "validation", "test"):
                frame = prepare_count_frame(raw if split == "test" else tv)
                loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=84,
                                     max_seq_len=84, shuffle=False, generator=None)
                _, full_populations[split] = runner.population_identity(loader.dataset, split)
                if split != "test":
                    populations[split] = full_populations[split]
            data["inherited_data_identity"] = {"data": {"path": str(tv_path), "sha256": runner.sha_file(tv_path)},
                "split_manifest": {"path": str(manifest), "sha256": runner.sha_file(manifest)}, "populations": populations}
            data["parent_data_identity"] = {"data": {"path": str(full_path), "sha256": runner.sha_file(full_path)},
                "split_manifest": data["inherited_data_identity"]["split_manifest"], "populations": full_populations}
            train_q = tv.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype("<f8")
            train_loader = make_loader(prepare_count_frame(tv), target_split="train", batch_size=128,
                                       lookback_weeks=84, max_seq_len=84, shuffle=False, generator=None)
            target_q, _ = runner.population_identity(train_loader.dataset, "train")
            data["statistics"] = {"train_log_mean": float(np.log1p(train_q).mean()),
                "train_log_std": float(np.log1p(train_q).std()), "raw_scale": float(np.sqrt(np.square(target_q).mean())),
                "all_train_quantity_sha256": hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0"+train_q.tobytes()).hexdigest()}
            temporal = derive_train_time_contract(prepare_count_frame(tv), lookback_weeks=84, max_seq_len=84)
            data["model"] = {**data["model"], "time_scale": temporal["time_scale"],
                "time_initial_location": temporal["target_log_scaled_mean"],
                "time_initial_scale": temporal["target_log_scaled_std"], "time_sigma_floor": .001,
                "time_head_lr_multiplier": 1.,
                "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "month", "top_code": None}}
            data["time_statistics"] = {"train_time_scale": temporal["time_scale"],
                "train_log_scaled_mean": temporal["target_log_scaled_mean"], "train_log_scaled_std": temporal["target_log_scaled_std"]}
            for split in ("train", "validation"):
                ds = make_loader(prepare_count_frame(tv), target_split=split, batch_size=128,
                    lookback_weeks=84, max_seq_len=84, shuffle=False, generator=None).dataset
                values = torch.tensor([max(1., float(ds.dt_lists[p][e+1])) for p,e in ds.index], dtype=torch.float64)
                data["time_statistics"][f"expected_{split}_targets"] = len(ds)
                data["time_statistics"][f"expected_{split}_target_dt_sha256"] = target_dt_sha256(values)
            data["optimizer"] = {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.}
            actual_args, actual_eval = runner.training_args, runner.evaluate_selected
            def tiny_args(*args, **kwargs):
                result = actual_args(*args, **kwargs)
                self.assertEqual((result.epochs, result.min_epochs, result.early_stopping_patience), (300,40,40))
                result.epochs = result.min_epochs = result.early_stopping_patience = 2
                result.device = "cpu"
                return result
            def cpu_eval(*args, **kwargs):
                return actual_eval(*args, **kwargs, device="cpu")
            with patch("paper.scripts.quantity_comparison_runtime.configure_runtime", side_effect=lambda *a: configure_runtime("cpu",4)), \
                    patch.object(runner, "training_args", side_effect=tiny_args), \
                    patch.object(runner, "evaluate_selected", side_effect=cpu_eval):
                for job in c["jobs"]:
                    with self.subTest(arm=job["arm"]):
                        tr = runner.run_stage(c, "5090", job["id"], "train")
                        self.assertEqual(tr["summary"]["completed_epochs"], 2)
                        self.assertIsNone(tr["validation_reference"]["metrics"]["tail"]["qty_rmse"])
                        vr = runner.run_stage(c, "5090", job["id"], "validation")
                        self.assertTrue(vr["validation_replay_passed"])
                runner.write_exclusive(root/"validation_gate.json", {"status": "passed", "host": "5090",
                    "contract_sha256": runner.sha_json(c), "jobs": {job["id"]: {"validation_receipt_sha256":
                        runner.sha_file(root/"run"/job["id"]/"validation_receipt.json")} for job in c["jobs"]}})
                for job in c["jobs"]:
                    te = runner.run_stage(c, "5090", job["id"], "test")
                    self.assertEqual(te["prediction_rows"], 4)
                    self.assertTrue(te["held_out_test_evaluated"])
                    runner.verify_inventory(root, te["files"])


if __name__ == "__main__":
    unittest.main()
