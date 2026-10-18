"""Time-weight MLP tests use fabricated tensors, rows and temporary artifacts only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

try:
    from paper.scripts import run_titantpp_time_weight_mlp18 as worker
except ImportError:
    spec = importlib.util.spec_from_file_location("time_weight_worker_under_test", Path(__file__).with_name("run_titantpp_time_weight_mlp18.py"))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)


def contract_at(root):
    data = {"dataset_id": "raf_spare_parts", "cap_value": 10.,
        "quantity_boundaries_all_train_rows": [3., 10., 10., 10.],
        "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.},
        "model": {"hidden_dim": 64, "quantity_variant": worker.VARIANT,
            "lambda_log_qty": 1., "lambda_tail": 0., "time_head_lr_multiplier": 1.,
            "time_head_mode": "heteroscedastic_lognormal_duration"}}
    return {"schema": worker.CONTRACT_SCHEMA, "hosts": {"5090": {"root": str(root)}},
        "datasets": [data], "jobs": [{"id": "raf_mlp16_seed42_alpha0p1", "host": "5090",
            "dataset": "raf_spare_parts", "arm": worker.ARMS[0], "seed": 42, "time_alpha": .1}],
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "batch_size": 128, "monitor": worker.MONITOR, "warm_start": False}}


def bound_payload(alpha):
    identity = worker.weight_identity(alpha)
    interface = {"time_weight_identity": identity, "unchanged": True}
    return {"time_alpha": alpha, "time_weight_identity": identity, "interface_meta": interface,
        "resume_identity": {"interface_meta": copy.deepcopy(interface), "time_weight_identity": identity,
            "arguments": {"time_alpha": alpha}}}


class ExactObjectiveTests(unittest.TestCase):
    def test_alpha1_preserves_exact_original_joint_object(self):
        t, q = torch.tensor([1., 2.], requires_grad=True), torch.tensor([3., 4.], requires_grad=True)
        original = {"joint_loss": t + q, "time_loss": t, "quantity_train_loss": q, "log_qty_loss": q,
            "unchanged": object()}
        wrapped = worker.make_weighted_target_outputs(lambda *a, **k: original, 1.)
        result = wrapped(None, lambda_log_qty=1.)
        self.assertIs(result["joint_loss"], original["joint_loss"])
        self.assertIs(result["time_loss"], t)
        self.assertIs(result["quantity_train_loss"], q)
        self.assertIs(result["unchanged"], original["unchanged"])

    def test_only_time_gradient_changes_before_global_clipping(self):
        for alpha in worker.TIME_ALPHAS:
            with self.subTest(alpha=alpha):
                time = torch.tensor([.3, .8], dtype=torch.float64, requires_grad=True)
                quantity = torch.tensor([.2, .6], dtype=torch.float64, requires_grad=True)
                def original(*args, **kwargs):
                    tl, ql = time.square(), quantity.square()
                    return {"joint_loss": tl+ql, "time_loss": tl, "quantity_train_loss": ql, "log_qty_loss": ql}
                result = worker.make_weighted_target_outputs(original, alpha)(None, lambda_log_qty=1.)
                torch.testing.assert_close(result["joint_loss"], alpha*time.square()+quantity.square(), rtol=0., atol=0.)
                gt, gq = torch.autograd.grad(result["joint_loss"].mean(), (time, quantity))
                torch.testing.assert_close(gt, alpha*time, rtol=0., atol=0.)
                torch.testing.assert_close(gq, quantity, rtol=0., atol=0.)
                torch.testing.assert_close(result["time_loss"], time.square(), rtol=0., atol=0.)

    def test_scalar_and_gradient_alpha_rejects_foreign_weight_or_qty_loss(self):
        for alpha in (0., 1.1, float("nan"), True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                worker.weight_identity(alpha)
        wrong = {"joint_loss": torch.ones(2), "time_loss": torch.ones(2),
            "quantity_train_loss": torch.ones(2)*2, "log_qty_loss": torch.ones(2)}
        for lam in (1., .1):
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                worker.make_weighted_target_outputs(lambda *a, **k: wrong, .1)(None, lambda_log_qty=lam)

    def test_alpha_does_not_construct_a_different_model_or_consume_rng(self):
        states, rngs, metas = [], [], []
        for alpha in (*worker.TIME_ALPHAS, 1.):
            with worker.runtime_hooks(alpha):
                torch.manual_seed(42)
                model, metadata = worker.build_model(worker.fabricated_data(), worker.ARMS[0])
                states.append({k: v.clone() for k,v in model.state_dict().items()})
                rngs.append(torch.get_rng_state().clone()); metas.append(metadata)
        self.assertEqual(metas[0], metas[1]); self.assertEqual(metas[1], metas[2])
        for state, rng in zip(states[1:], rngs[1:]):
            self.assertTrue(torch.equal(rngs[0], rng))
            for key, value in states[0].items():
                self.assertTrue(torch.equal(value, state[key]), key)

    def test_alpha1_real_training_step_and_rng_are_bitwise_identical(self):
        from paper.scripts.count_aware_tpp_backbone import training
        from simple_lab_test.search.common.runner import canonical_state_dict_sha256
        worker.inherited.install_hooks()
        torch.set_num_threads(4)
        dts = torch.tensor([[1., 2., 1., 3.], [1., 1., 2., 1.]])
        qty = torch.tensor([[1., 3., 2., 4.], [2., 4., 1., 3.]])
        loader = [(None, dts, torch.ones_like(dts, dtype=torch.bool), None, qty)]
        torch.manual_seed(42)
        a, _ = worker.build_model(worker.fabricated_data(), worker.ARMS[0])
        torch.manual_seed(42)
        b, _ = worker.build_model(worker.fabricated_data(), worker.ARMS[0])
        optimizer_a = training.build_optimizer(a, lr=.001, time_head_lr_multiplier=1.)
        optimizer_b = training.build_optimizer(b, lr=.001, time_head_lr_multiplier=1.)
        rng = torch.get_rng_state().clone()
        base = training.train_epoch_with_telemetry(model=a, loader=loader, optimizer=optimizer_a,
            device="cpu", lambda_log_qty=1., grad_clip=1., max_batches=None)
        after_a = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        with worker.runtime_hooks(1.):
            alternate = training.train_epoch_with_telemetry(model=b, loader=loader, optimizer=optimizer_b,
                device="cpu", lambda_log_qty=1., grad_clip=1., max_batches=None)
        self.assertEqual(canonical_state_dict_sha256(a.state_dict()), canonical_state_dict_sha256(b.state_dict()))
        self.assertTrue(torch.equal(after_a, torch.get_rng_state()))
        for key, value in base.items():
            self.assertEqual(value, alternate[key], key)


class BindingAndIsolationTests(unittest.TestCase):
    def test_scope_allows_only_new_mlp16_alpha_point1_or10(self):
        for alpha in worker.TIME_ALPHAS:
            c = contract_at("/tmp/time-weight-fixture")
            c["jobs"][0]["time_alpha"] = alpha
            worker.select_job(c, "5090", c["jobs"][0]["id"])
        for key, value in (("time_alpha", 1.), ("time_alpha", .2), ("arm", "titantpp_cnn"), ("seed", 32)):
            c = contract_at("/tmp/time-weight-fixture"); c["jobs"][0][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                worker.select_job(c, "5090", c["jobs"][0]["id"])
        c = contract_at("/tmp/time-weight-fixture"); c["training"]["maximum_epochs"] = 301
        with self.assertRaises(ValueError):
            worker.select_job(c, "5090", c["jobs"][0]["id"])

    def test_nested_or_failed_hooks_restore_all_process_references(self):
        worker.inherited.install_hooks()
        from paper.scripts.count_aware_tpp_backbone import core, training
        from paper.scripts import run_multilag_detail_execution as base
        before = (core.target_outputs, training.target_outputs, training._resume_identity,
            training.save_json, training.atomic_torch_save, training.validate_checkpoint_route,
            base.time_interface, worker.inherited.training_args, worker.inherited.SCHEMA)
        with self.assertRaises(RuntimeError):
            with worker.runtime_hooks(.1):
                with self.assertRaises(ValueError):
                    with worker.runtime_hooks(10.):
                        pass
                raise RuntimeError("fabricated interrupted stage")
        after = (core.target_outputs, training.target_outputs, training._resume_identity,
            training.save_json, training.atomic_torch_save, training.validate_checkpoint_route,
            base.time_interface, worker.inherited.training_args, worker.inherited.SCHEMA)
        self.assertEqual(before, after)
        self.assertIsNone(worker._ACTIVE_ALPHA)

    def test_wrong_alpha_in_any_metadata_level_is_rejected(self):
        valid = bound_payload(.1)
        worker.validate_weight_binding(valid, .1)
        for target in ("top", "identity", "interface", "resume", "args"):
            p = copy.deepcopy(valid)
            if target == "top": p["time_alpha"] = 10.
            if target == "identity": p["time_weight_identity"]["time_alpha"] = 10.
            if target == "interface": p["interface_meta"]["time_weight_identity"]["time_alpha"] = 10.
            if target == "resume": p["resume_identity"]["time_weight_identity"]["time_alpha"] = 10.
            if target == "args": p["resume_identity"]["arguments"]["time_alpha"] = 10.
            with self.subTest(target=target), self.assertRaises(ValueError):
                worker.validate_weight_binding(p, .1)

    def test_weighted_and_unweighted_logs_are_separate(self):
        row = {"epoch": 1, "train_time_nll": 2., "train_quantity_loss": 3., "train_joint_objective": 3.2,
            "val_time_nll": 4., "val_quantity_train_loss": 5., "val_joint_objective": 5.4}
        worker.annotate_history_row(row, .1)
        self.assertEqual(row["train_joint_weighted"], 3.2)
        self.assertEqual(row["train_joint_unweighted"], 5.)
        self.assertEqual(row["val_joint_unweighted"], 9.)
        with tempfile.TemporaryDirectory() as temp:
            from paper.scripts.count_aware_tpp_backbone import training
            path = Path(temp)/"history.json"
            with worker.runtime_hooks(.1):
                training.save_json(path, {"history": [row]})
            saved = json.loads(path.read_text())
            self.assertEqual(saved["time_alpha"], .1)
            logged = json.loads((path.parent/"time_weight.log").read_text())
            self.assertEqual(logged["val_joint_weighted"], 5.4)
            self.assertEqual(logged["val_joint_unweighted"], 9.)

    def test_selector_remains_first_strict_finite_raw_validation_rmse(self):
        from paper.scripts.count_aware_tpp_backbone.training import earliest_strict_minimum
        history = [{"epoch": 1, "val_qty_rmse": 3., "val_joint_objective": .01},
            {"epoch": 2, "val_qty_rmse": 2., "val_joint_objective": 100.},
            {"epoch": 3, "val_qty_rmse": 2., "val_joint_objective": .001}]
        for alpha in worker.TIME_ALPHAS:
            with worker.runtime_hooks(alpha):
                best = earliest_strict_minimum(history, metric_key="val_qty_rmse")
                self.assertEqual(best["epoch"], 2)

    def test_test_stage_and_nonvalidation_replay_are_physically_rejected(self):
        with self.assertRaises(ValueError):
            worker.run_stage({}, "5090", "unused", "test")
        with self.assertRaises(ValueError):
            worker.evaluate_selected({}, {}, {}, "/tmp", None, "test", "/tmp/unwritten")

    def test_synthetic_admission_both_weights_same_model(self):
        result = worker.synthetic_check(device="cpu")
        self.assertEqual(result["time_alphas"], [.1, 10.])
        self.assertEqual(result["arms"], list(worker.ARMS))
        self.assertEqual(set(result["cuda_memory_peaks"]), set(worker.ARMS))
        self.assertFalse(result["real_data_loaded"])
        self.assertFalse(result["held_out_test_evaluated"])

    def test_fabricated_full_train_selected_checkpoint_validation_only(self):
        import numpy as np
        import polars as pl
        from paper.scripts.quantity_comparison_runtime import configure_runtime
        from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
        from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
        from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
        from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
        from simple_lab_test.search.common.runner import torch_load_checkpoint
        configure_runtime("cpu", 4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); c = contract_at(root)
            c["source"] = {"base_git_revision": "fabricated", "files_sha256": "fixture"}
            c["design_sha256"], c["architecture"] = "fixture", {"models": list(worker.ARMS)}
            c["jobs"] = [{"id": f"fabricated_alpha_{alpha}", "host": "5090", "dataset": "raf_spare_parts",
                "seed": 42, "arm": worker.ARMS[0], "time_alpha": alpha} for alpha in worker.TIME_ALPHAS]
            raw = pl.DataFrame([{"oper_part_no": entity, "seq": i+1,
                "delta_t": [0, 1, 2, 1, 3, 1][i], "demand_qty": float([1, 3, 8, 10, 10, 2][i]),
                "chronological_split": "train" if i < 4 else "validation"}
                for entity in ("fabricated_a", "fabricated_b") for i in range(6)])
            path, manifest = root/"train_validation.parquet", root/"manifest.json"
            raw.write_parquet(path); manifest.write_text('{"fabricated":true}')
            data = c["datasets"][0]
            data["loader"] = {"batch_size": 128, "lookback_weeks": 84, "max_seq_len": 84,
                "num_workers": 0, "drop_last": False, "train_shuffle": True, "validation_shuffle": False}
            frame = prepare_count_frame(raw)
            populations = {}
            for split in ("train", "validation"):
                loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=84,
                    max_seq_len=84, shuffle=False, generator=None)
                _, populations[split] = worker.population_identity(loader.dataset, split)
            data["inherited_data_identity"] = {"data": {"path": str(path), "sha256": worker.sha_file(path)},
                "split_manifest": {"path": str(manifest), "sha256": worker.sha_file(manifest)}, "populations": populations}
            train_q = raw.filter(pl.col("chronological_split")=="train")["demand_qty"].to_numpy().astype("<f8")
            train_ds = make_loader(frame, target_split="train", batch_size=128, lookback_weeks=84,
                max_seq_len=84, shuffle=False, generator=None).dataset
            target_q, _ = worker.population_identity(train_ds, "train")
            data["statistics"] = {"train_log_mean": float(np.log1p(train_q).mean()),
                "train_log_std": float(np.log1p(train_q).std()), "raw_scale": float(np.sqrt(np.square(target_q).mean())),
                "all_train_quantity_sha256": hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0"+train_q.tobytes()).hexdigest()}
            temporal = derive_train_time_contract(frame, lookback_weeks=84, max_seq_len=84)
            data["model"].update(time_scale=temporal["time_scale"], time_initial_location=temporal["target_log_scaled_mean"],
                time_initial_scale=temporal["target_log_scaled_std"], time_sigma_floor=.001,
                time_observation_contract={"mode": "positive_integer_round_clamp_v1", "unit": "month", "top_code": None})
            data["time_statistics"] = {"train_time_scale": temporal["time_scale"],
                "train_log_scaled_mean": temporal["target_log_scaled_mean"], "train_log_scaled_std": temporal["target_log_scaled_std"]}
            for split in ("train", "validation"):
                ds = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=84,
                    max_seq_len=84, shuffle=False, generator=None).dataset
                values = torch.tensor([max(1.,float(ds.dt_lists[p][e+1])) for p,e in ds.index], dtype=torch.float64)
                data["time_statistics"][f"expected_{split}_targets"] = len(ds)
                data["time_statistics"][f"expected_{split}_target_dt_sha256"] = target_dt_sha256(values)
            original_args, original_eval = worker.training_args, worker.evaluate_selected
            def tiny(*args, **kwargs):
                result = original_args(*args, **kwargs)
                self.assertEqual((result.epochs,result.min_epochs,result.early_stopping_patience), (300,40,40))
                result.epochs = result.min_epochs = result.early_stopping_patience = 2
                result.device = "cpu"
                return result
            def cpu_eval(*args, **kwargs):
                return original_eval(*args, **kwargs, device="cpu")
            with patch("paper.scripts.quantity_comparison_runtime.configure_runtime", side_effect=lambda *a: configure_runtime("cpu",4)), \
                patch.object(worker, "training_args", side_effect=tiny), patch.object(worker, "evaluate_selected", side_effect=cpu_eval):
                for job in c["jobs"]:
                    with self.subTest(alpha=job["time_alpha"]):
                        train = worker.run_stage(c, "5090", job["id"], "train")
                        self.assertEqual(train["summary"]["completed_epochs"], 2)
                        worker.validate_weight_binding(train["summary"], job["time_alpha"])
                        cp = torch_load_checkpoint(Path(train["selected_checkpoint"]), map_location="cpu")
                        worker.validate_weight_binding(cp, job["time_alpha"])
                        validation = worker.run_stage(c, "5090", job["id"], "validation")
                        self.assertTrue(validation["validation_replay_passed"])
                        self.assertEqual(validation["time_alpha"], job["time_alpha"])
                        self.assertFalse(validation["held_out_test_evaluated"])
                        with self.assertRaises(ValueError):
                            worker._load_selected(data, {**job,"time_alpha": 10. if job["time_alpha"]==.1 else .1}, train, root, "cpu")
                        # Even self-consistently rehashed bytes cannot relabel a
                        # foreign model-state identity as the approved MLP16.
                        from simple_lab_test.search.common.runner import canonical_state_dict_sha256
                        wrong = copy.deepcopy(cp)
                        wrong["model_state_dict"]["history_width_identity"][0] ^= 1
                        wrong["model_state_sha256"] = canonical_state_dict_sha256(wrong["model_state_dict"])
                        bad_path = root / (job["id"]+"_foreign_identity.pt")
                        torch.save(wrong, bad_path)
                        forged = {**train, "selected_checkpoint": str(bad_path),
                            "checkpoint_sha256": worker.sha_file(bad_path), "state_sha256": wrong["model_state_sha256"]}
                        with worker.runtime_hooks(job["time_alpha"]), self.assertRaises(ValueError):
                            worker._load_selected(data, job, forged, root, "cpu")
                        with self.assertRaises(ValueError):
                            worker.run_stage(c,"5090",job["id"],"train")


if __name__ == "__main__":
    unittest.main()
