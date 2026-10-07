"""Stdlib repair-contract and completed-epoch audit regression tests; no GPU/data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("repair", ROOT / "paper/scripts/titantpp_contextual_retrieval_repair_runtime.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def contract():
    parent = json.loads((ROOT / "search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/execution_contract.json").read_text())
    hosts = deepcopy(parent["hosts"])
    for host, value in hosts.items():
        folder = str(Path(value["root"]).parent / (repair.SCHEMA + "_" + host))
        value.update(root=folder, source_root=folder + "/source", operation_root=folder + "/operation", tmux=repair.SCHEMA + "_" + host)
        value["environment"].update(PYTHONPATH=folder + "/source", MPLCONFIGDIR=folder + "/cache/matplotlib", XDG_CACHE_HOME=folder + "/cache")
    jobs = [j for j in parent["jobs"] if j["host"] == "5080" or j["arm"] not in ("titantpp_retrieval_mean", "titantpp_retrieval_weighted")]
    reused = [j for j in parent["jobs"] if j["host"] == "5090" and j["arm"] in ("titantpp_retrieval_mean", "titantpp_retrieval_weighted")]
    files = {"operation/repair_runtime.py": "0" * 64}
    return {"schema": repair.SCHEMA, "parent_contract": parent, "parent_contract_sha256": repair.PARENT_SHA,
        "hosts": hosts, "limits": {"per_condition_seconds": 21600, "total_wall_seconds": 129600, "qualification_seconds": 1800},
        "jobs": jobs, "reused_jobs": reused, "resume": {repair.FIXED_ID: {"last_epoch": 148, "best_epoch": 120}},
        "operation": {"files": files, "files_sha256": repair.sha_json(files)}, "policy": {"automatic_retry": False}}


def population():
    return {"inherited_data_identity": {"populations": {"train": {"target_count": 384}, "validation": {"target_count": 128}}},
            "loader": {"batch_size": 128}}


def prefix(n):
    records = {s: [{"epoch": e, "split": s, "count": 384 if s == "train" else 128,
        "batches": 3 if s == "train" else 1, "batch_order_sha256": "0" * 64} for e in range(1, n + 1)]
        for s in ("train", "validation")}
    timings = {"epochs": [{"epoch": e, "elapsed_seconds": 1.0} for e in range(1, n + 1)]}
    return records, timings


class FakeExposureLoader:
    def __init__(self, loader, split, records, budget, on_start=None):
        self.split, self.records, self.on_start = split, records, on_start
    def __iter__(self):
        if self.on_start:
            self.on_start()
        yield None
        self.records.append({"epoch": len(self.records) + 1, "split": self.split,
            "count": 384 if self.split == "train" else 128, "batches": 3 if self.split == "train" else 1,
            "batch_order_sha256": "0" * 64})


class ContractTests(unittest.TestCase):
    def test_original_parent_and_exact_eight_admitted_jobs(self):
        repair.validate(contract(), verify_files=False)

    def test_parent_sha_drift_rejected(self):
        c = contract()
        c["parent_contract"]["training"]["patience"] = 39
        with self.assertRaisesRegex(ValueError, "Parent scientific"):
            repair.validate(c, verify_files=False)

    def test_duplicate_or_reused_job_readmission_rejected(self):
        for replacement in (contract()["jobs"][0], contract()["reused_jobs"][0]):
            c = contract()
            c["jobs"][-1] = replacement
            with self.assertRaisesRegex(ValueError, "Only original"):
                repair.validate(c, verify_files=False)

    def test_changed_runtime_or_budget_rejected(self):
        c = contract()
        c["hosts"]["5080"]["gpu_uuid"] = "foreign"
        with self.assertRaisesRegex(ValueError, "Qualified runtime"):
            repair.validate(c, verify_files=False)
        c = contract()
        c["limits"]["per_condition_seconds"] += 1
        with self.assertRaisesRegex(ValueError, "wall budget"):
            repair.validate(c, verify_files=False)
        for name, value in (("OMP_NUM_THREADS", "8"), ("PYTHONPATH", "/foreign/source")):
            c = contract()
            c["hosts"]["5080"]["environment"][name] = value
            with self.assertRaisesRegex(ValueError, "numerical environment"):
                repair.validate(c, verify_files=False)

    def test_exact_resume_epoch_prefix_and_malformed_prefix(self):
        rows, timing = prefix(148)
        repair.validate_prefix(rows, timing, 148, population())
        rows["validation"].pop()
        with self.assertRaisesRegex(ValueError, "exposure length"):
            repair.validate_prefix(rows, timing, 148, population())

    def test_wrong_file_digest_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            c = contract()
            with self.assertRaises(FileNotFoundError):
                repair.validate(c, bundle=folder)

    def test_expired_repair_lease_never_uses_parent_deadline(self):
        c = contract()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            approval = {"approved": True, "contract_sha256": repair.sha_json(c), "hosts": ["5080"], "user_instruction": "문제 해결해서 진행하자"}
            start = {"host": "5080", "contract_sha256": repair.sha_json(c), "approval_sha256": repair.sha_json(approval),
                     "started_at_unix": 1.0, "deadline_unix": 129601.0}
            repair.write(root / "approval.json", approval)
            repair.write(root / "start_permit.json", start)
            with patch.object(repair.time, "time", return_value=129602.0):
                with self.assertRaisesRegex(ValueError, "lease not active"):
                    repair.authorization(c, "5080", root)

    def test_only_two_resolved_input_paths_can_change(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder) / "old", Path(folder) / "new"
            relative = "inputs/data/yellow_trip_hourly/"
            names = {"local_data_path": "yellow_trip_hourly_with_split.parquet", "local_split_manifest_path": "yellow_trip_hourly_split_manifest.json"}
            original = {"resolved_inputs": {}, "population": 8268}
            actual = {"resolved_inputs": {}, "population": 8268}
            for key, name in names.items():
                for source in (old, new):
                    path = source / relative / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"same frozen input")
                original["resolved_inputs"][key] = str(old / relative / name)
                actual["resolved_inputs"][key] = str(new / relative / name)
            repair.verify_relocated_input_receipt(actual, original, old, new)
            actual["population"] = 8267
            with self.assertRaisesRegex(ValueError, "semantic input"):
                repair.verify_relocated_input_receipt(actual, original, old, new)


class AuditTests(unittest.TestCase):
    def trainer(self):
        saved = []
        training = SimpleNamespace(make_loader=lambda *a, **kw: [], atomic_torch_save=lambda value, path: saved.append(value))
        return training, saved

    def test_resume_continues_prefix_and_keeps_optimizer_step_identity(self):
        rows, timings = prefix(148)
        training, saved = self.trainer()
        previous_loader, previous_save = training.make_loader, training.atomic_torch_save
        callbacks = []
        with tempfile.TemporaryDirectory() as folder:
            with repair.audited_training(training, SimpleNamespace(ExposureLoader=FakeExposureLoader), lambda: None,
                    lambda e, r: callbacks.append((e, deepcopy(r))), prefix=rows, timing_prefix=timings["epochs"]) as records:
                list(training.make_loader(target_split="train"))
                list(training.make_loader(target_split="validation"))
                payload = {"epoch": 149, "optimizer_state_dict": {"state": {0: {"step": 447}}},
                    "history": [{"train_event_count": 384, "train_batch_count": 3}]}
                training.atomic_torch_save(payload, Path(folder) / "last_epoch_state.pt")
                self.assertEqual(len(records["train"]), 149)
                self.assertEqual(records["train"][:148], rows["train"])
            self.assertEqual(len(repair.read(Path(folder) / "epoch_timing.json")["epochs"]), 149)
        self.assertEqual(callbacks[0][0], 149)
        self.assertEqual(len(saved), 1)
        self.assertIs(training.make_loader, previous_loader)
        self.assertIs(training.atomic_torch_save, previous_save)


class NativeResumeTests(unittest.TestCase):
    trainer = AuditTests.trainer

    def test_frozen_native_resume_retains_best_rng_optimizer_and_original_prefix(self):
        """Actual frozen lifecycle and AdamW; fabricated one-row CPU batches only."""
        try:
            import torch
        except ImportError:
            self.skipTest("Local Torch unavailable; native GPU gate remains required")
        from contextlib import ExitStack
        from paper.scripts import run_observed_slot_partition as shared
        from simple_lab_test.search.common.runner import capture_rng_state, canonical_state_dict_sha256, torch_load_checkpoint
        frozen = ROOT / "search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1/source/paper/scripts/count_aware_tpp_backbone/training.py"
        native_spec = importlib.util.spec_from_file_location("frozen_retrieval_resume_lifecycle", frozen)
        training = importlib.util.module_from_spec(native_spec)
        native_spec.loader.exec_module(training)
        generator = training.set_seed(42)
        model = torch.nn.Linear(1, 1)
        initial = canonical_state_dict_sha256(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        best_state = None
        for epoch in range(1, 149):
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            torch.rand(1, generator=generator)
            if epoch == 120:
                best_state = training.clone_state_dict(model)
        history = [{"epoch": e, "val_qty_rmse": 1. if e == 120 else 2.,
            "val_joint_objective": .5, "val_qty_mae": .4, "val_time_nll": .2,
            "train_event_count": 1, "train_batch_count": 1, "train_all_finite": True}
            for e in range(1, 149)]
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            args = SimpleNamespace(**{k: None for k in training._RESUME_IDENTITY_ARGUMENTS})
            args.__dict__.update(output_dir=Path(folder), force_rerun=False, epochs=300,
                min_epochs=40, early_stopping_patience=40, device="cpu", checkpoint_monitor="validation_raw_quantity_rmse",
                source_revision="a" * 40, batch_size=128, max_seq_len=2, lookback_weeks=1,
                hidden_dim=1, lr=.001, time_head_lr_multiplier=1., lambda_log_qty=1.,
                max_val_batches=None, max_train_batches=None, grad_clip=1., model_role=None,
                quantity_sigma_floor=.001, lambda_location_huber=1., location_huber_delta=.25,
                lambda_tail=0., tail_threshold=1., tail_normalization_scale=1., tail_clip_cap=1.,
                tail_huber_delta=1., time_head_mode="legacy_clamped_rmtpp", time_scale=1.,
                time_w_max=1., time_intercept_limit=300., time_wd_safety_limit=40., time_sigma_floor=.001)
            interface = {"train_target_mean": 1., "train_target_std": 1., "time_head": {}}
            quantity = {"boundaries": [1, 2, 3, 4], "strata": []}
            variant = "count_only_log_regression"
            identity = training._resume_identity(args=args, backbone="offline_toy", quantity_variant=variant,
                seed=42, monitor="validation_raw_quantity_rmse", quantity_contract=quantity, interface_meta=interface)
            selection = training.checkpoint_monitor_spec("validation_raw_quantity_rmse")
            payload = {"checkpoint_type": "epoch_resume", "checkpoint_schema_version": 2,
                "epoch": 148, "backbone": "offline_toy", "variant": variant, "seed": 42,
                "resume_identity": identity, "checkpoint_monitor": "validation_raw_quantity_rmse",
                "checkpoint_monitor_history_key": "val_qty_rmse", "checkpoint_selection": selection["selection"],
                "initial_state_sha256": initial, "model_state_dict": training.clone_state_dict(model),
                "model_state_sha256": canonical_state_dict_sha256(model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(), "history": history,
                "best_selection_value": 1., "best_epoch": 120, "best_state_dict": best_state,
                "best_state_sha256": canonical_state_dict_sha256(best_state), "rng_state": capture_rng_state(),
                "train_loader_generator_state": generator.get_state(), "source_revision": args.source_revision,
                "source_revision_history": [args.source_revision]}
            training._validate_resume_payload(payload, expected_identity=identity,
                expected_initial_state_sha256=initial, monitor="validation_raw_quantity_rmse")
            bad = deepcopy(payload)
            del bad["rng_state"]["torch"]
            with self.assertRaisesRegex(ValueError, "RNG state"):
                training._validate_resume_payload(bad, expected_identity=identity,
                    expected_initial_state_sha256=initial, monitor="validation_raw_quantity_rmse")
            run = Path(folder) / "runs/offline_toy" / variant / "seed_42"
            run.mkdir(parents=True)
            training.atomic_torch_save(payload, run / "last_epoch_state.pt")
            repair.write(run / "history.json", {"history": history})
            # Native entry validation is replaced only for this fabricated toy route.
            stack.enter_context(patch.object(training, "validate_training_identity", lambda *a: None))
            stack.enter_context(patch.object(training, "validate_checkpoint_route", lambda *a: None))
            stack.enter_context(patch.object(training, "time_model_kwargs", lambda *a: {}))
            stack.enter_context(patch.dict(training.BACKBONE_LABELS, {"offline_toy": "offline toy"}))
            batch = (None, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), None, torch.ones(1, 1))
            captured = {}
            def loader(*a, **kw):
                if kw["target_split"] == "train":
                    captured["generator"] = kw["generator"]
                return [batch]
            stack.enter_context(patch.object(training, "make_loader", loader))
            stack.enter_context(patch.object(training, "build_model", lambda *a, **kw: (torch.nn.Linear(1, 1), {})))
            new_steps = []
            def train_epoch(**kw):
                if not new_steps:
                    self.assertTrue(torch.equal(captured["generator"].get_state(), payload["train_loader_generator_state"]))
                    self.assertTrue(torch.equal(torch.get_rng_state(), payload["rng_state"]["torch"]))
                    self.assertTrue(all(float(v["step"]) == 148 for v in kw["optimizer"].state_dict()["state"].values()))
                for _, x, *_ in kw["loader"]:
                    opt = kw["optimizer"]
                    opt.zero_grad()
                    kw["model"](x).sum().backward()
                    opt.step()
                new_steps.append(True)
                return {"train_joint_objective": .5, "train_time_nll": .2, "train_batch_joint_p99": .5,
                    "train_pre_clip_grad_norm_mean": 1., "train_gradient_clip_fraction": 0.,
                    "train_event_count": 1, "train_batch_count": 1, "train_all_finite": True}
            def evaluate(**kw):
                list(kw["loader"])
                is_best = canonical_state_dict_sha256(kw["model"].state_dict()) == payload["best_state_sha256"]
                return {"val_joint_objective": .5, "val_time_nll": .2, "val_quantity_train_loss": .5,
                    "val_log_qty_mse": .5, "val_quantity_distribution_nll": 0., "val_quantity_location_huber": 0.,
                    "val_tail_aux_loss": 0., "val_tail_count": 0, "val_quantity_scale_mean": 0.,
                    "qty_mae": .4, "qty_rmse": 1. if is_best else 2.}
            stack.enter_context(patch.object(training, "train_epoch_with_telemetry", train_epoch))
            stack.enter_context(patch.object(training, "evaluate", evaluate))
            exposure = {s: [{"epoch": e, "split": s, "count": 1, "batches": 1,
                "batch_order_sha256": "f" * 64} for e in range(1, 149)] for s in ("train", "validation")}
            timings = [{"epoch": e, "elapsed_seconds": 1.} for e in range(1, 149)]
            with repair.audited_training(training, shared, lambda: None, lambda *a: None,
                    prefix=exposure, timing_prefix=timings) as records:
                summary, _, _ = training.train_one(args=args, frame=None, quantity_contract=quantity,
                    interface_meta=interface, backbone="offline_toy", quantity_variant=variant, seed=42)
            self.assertEqual(len(new_steps), 12)
            self.assertEqual(summary["completed_epochs"], 160)
            self.assertEqual(summary["best_epoch"], 120)
            self.assertEqual(summary["checkpoint_state_sha256"], payload["best_state_sha256"])
            self.assertEqual(repair.read(run / "history.json")["history"][:148], history)
            self.assertEqual(records["train"][:148], exposure["train"])
            self.assertEqual(len(records["validation"]), 161)
            last = torch_load_checkpoint(run / "last_epoch_state.pt", map_location="cpu")
            self.assertTrue(all(float(v["step"]) == 160 for v in last["optimizer_state_dict"]["state"].values()))

    def test_optimizer_reset_is_rejected_before_checkpoint_save(self):
        training, saved = self.trainer()
        rows, timings = prefix(148)
        with repair.audited_training(training, SimpleNamespace(ExposureLoader=FakeExposureLoader), lambda: None,
                lambda e, r: None, prefix=rows, timing_prefix=timings["epochs"]):
            list(training.make_loader(target_split="train"))
            list(training.make_loader(target_split="validation"))
            with self.assertRaisesRegex(ValueError, "Optimizer steps"):
                training.atomic_torch_save({"epoch": 149, "optimizer_state_dict": {"state": {0: {"step": 3}}}}, "last_epoch_state.pt")
        self.assertEqual(saved, [])

    def test_incomplete_epoch_never_saves_and_hooks_restored(self):
        training, saved = self.trainer()
        previous_loader = training.make_loader
        rows, timings = prefix(148)
        with self.assertRaisesRegex(ValueError, "completed resumed epoch"):
            with repair.audited_training(training, SimpleNamespace(ExposureLoader=FakeExposureLoader), lambda: None,
                    lambda e, r: None, prefix=rows, timing_prefix=timings["epochs"]):
                list(training.make_loader(target_split="train"))
                training.atomic_torch_save({"epoch": 149}, "last_epoch_state.pt")
        self.assertEqual(saved, [])
        self.assertIs(training.make_loader, previous_loader)


if __name__ == "__main__":
    unittest.main()
