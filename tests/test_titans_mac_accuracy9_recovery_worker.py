"""Fabricated CPU continuation checks; no project held-out rows or SSH access."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from paper.scripts import run_titans_mac_accuracy9_recovery as recovery
from paper.scripts import run_titans_mac_accuracy9 as science


class InterruptedAfterSave(RuntimeError):
    pass


def make_fixture(tmp_path, monkeypatch, *, dataset="intermittent_frozen_5000"):
    from tests.test_titans_mac_accuracy9_worker import contract_fixture
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    old_root = tmp_path / "original"
    fixture = old_root / "fabricated"
    fixture.mkdir(parents=True)
    args, frame, fabricated, interface, _ = science.prepared._engine_fixture(fixture, science.prepared.ARMS[0])
    original = contract_fixture(old_root)
    original["source"].update(base_git_revision=args.source_revision,
                              files={"fabricated_source.py": "f" * 64})
    data = next(d for d in original["datasets"] if d["dataset_id"] == dataset)
    data.update({**fabricated, "dataset_id": dataset,
                 "quantity_boundaries_all_train_rows": [2., 5., 8., 10.]})
    lookback, maximum = (520, 256) if dataset == "intermittent_frozen_5000" else (84, 84)
    data["loader"].update(batch_size=128, lookback_weeks=lookback, max_seq_len=maximum,
                          num_workers=0, drop_last=False, train_shuffle=True, validation_shuffle=False)
    data["model"].update(lambda_log_qty=1., lambda_tail=0., time_head_lr_multiplier=1.)
    data["model"]["time_observation_contract"]["unit"] = "week" if dataset == "intermittent_frozen_5000" else "month"
    interface["time_head"]["observation_likelihood"] = deepcopy(data["model"]["time_observation_contract"])
    data["inherited_data_identity"] = {
        "data": {"path": str(args.data), "sha256": args.data_sha256},
        "split_manifest": {"path": str(args.split_manifest), "sha256": args.split_manifest_sha256},
        "populations": {}}
    for split in ("train", "validation"):
        loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=lookback,
                             max_seq_len=maximum, shuffle=False, generator=None)
        data["inherited_data_identity"]["populations"][split] = science.population_identity(loader.dataset, split)[1]
    model, encoder = science.build_model(data)
    original["frozen_conditions"] = {"models": {science.ARM: {
        "expected_encoder_config_by_dataset": {data["dataset_id"]: encoder},
        "expected_parameter_count_by_dataset": {data["dataset_id"]: sum(p.numel() for p in model.parameters())}}}}
    job = next(j for j in original["jobs"] if j["dataset"] == data["dataset_id"] and j["seed"] == 42)
    args.epochs = args.min_epochs = args.early_stopping_patience = 4
    args.model_role, args.time_alpha = science.role_for(science.ARM), 1.
    args.titans_memory_gradient_clip = 1.
    args.dataset_contract = data["dataset_id"]
    args.lookback_weeks, args.max_seq_len = lookback, maximum
    args.execution_role = "fresh_titans_mac_accuracy9_validation"
    for key, value in science.input_identity().items():
        setattr(args, key, value)
    interface.update(science.input_identity())
    interface.update(source_files_sha256=original["source"]["files_sha256"],
                     execution_contract_sha256=science.sha_json(original))

    def arguments(contract, selected_data, output, arm):
        assert contract == original and selected_data == data and arm == science.ARM
        result = deepcopy(args)
        result.output_dir = output
        return result

    monkeypatch.setattr(science, "training_args", arguments)
    monkeypatch.setattr(science, "training_interface", lambda *a: deepcopy(interface))
    monkeypatch.setattr(science, "check_native_runtime", lambda *a: {"fabricated_cpu_only": True})
    monkeypatch.setattr(base, "prepare_admitted_data", lambda *a: (frame, {"fabricated_cpu_only": True}))
    monkeypatch.setattr(science.mac_runtime, "configure_dynamo", lambda: None)
    configure_runtime("cpu", 4)
    science.write_exclusive(old_root / "scientific_contract.json", original)
    return original, data, job, frame, interface


def native_fit(original, data, job, frame, interface, output, *, stop_after=None):
    from paper.scripts.count_aware_tpp_backbone import training
    atomic = training.atomic_torch_save

    def save(payload, path):
        atomic(payload, path)
        if payload.get("checkpoint_type") == "epoch_resume" and payload["epoch"] == stop_after:
            raise InterruptedAfterSave("Fabricated durable interruption")

    with patch.object(training, "atomic_torch_save", side_effect=save):
        with science._engine_hooks(science.ARM, contract=original, data=data):
            return training.train_one(args=science.training_args(original, data, output, science.ARM),
                frame=frame, interface_meta=interface, quantity_contract=recovery.quantity_contract(data),
                backbone=science.ARM, quantity_variant=science.VARIANT, seed=job["seed"])


def last_path(root, job):
    return root / "runs" / science.ARM / science.VARIANT / f"seed_{job['seed']}" / "last_epoch_state.pt"


def operation(original, job, new_root, resume_path):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    old_root = Path(original["hosts"]["5090"]["root"])
    payload = torch_load_checkpoint(resume_path, map_location="cpu")
    source_contract = old_root / "scientific_contract.json"
    jobs = [{**j, "recovery_mode": "resume" if j["dataset"] == "intermittent_frozen_5000" else "reference"}
            for j in deepcopy(original["jobs"])]
    item = next(j for j in jobs if j["id"] == job["id"])
    item.update(origin_resume_checkpoint={"path": str(resume_path), "sha256": science.sha_file(resume_path)},
                resume_epoch=payload["epoch"])
    return {"schema": recovery.CONTRACT_SCHEMA, "host": "5090", "root": str(new_root),
        "source_root": str(new_root / "source"), "source": deepcopy(original["source"]),
        "scientific_contract": {"path": str(source_contract), "raw_sha256": science.sha_file(source_contract),
                                "canonical_sha256": science.sha_json(original)},
        "immutable_originals": {"root": str(old_root), "files": {
            str(p.relative_to(old_root)): science.sha_file(p) for p in old_root.rglob("*") if p.is_file()}},
        "jobs": jobs, "training_jobs": [j["id"] for j in jobs if j["recovery_mode"] == "resume"],
        "validation_jobs": [j["id"] for j in jobs], "test_jobs": [j["id"] for j in jobs]}


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            same_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            same_tree(x, y)
    else:
        assert a == b


def prepared_interruption(tmp_path, monkeypatch):
    original, data, job, frame, interface = make_fixture(tmp_path, monkeypatch)
    prefix = Path(original["hosts"]["5090"]["root"]) / "run" / job["id"]
    with pytest.raises(InterruptedAfterSave):
        native_fit(original, data, job, frame, interface, prefix, stop_after=2)
    contract = operation(original, job, tmp_path / "recovery", last_path(prefix, job))
    return original, data, job, frame, interface, contract


def scope_fixture(tmp_path, monkeypatch):
    original, data, job, frame, interface = make_fixture(tmp_path, monkeypatch)
    prefix = Path(original["hosts"]["5090"]["root"]) / "fabricated_scope_checkpoint.pt"
    torch.save({"epoch": 2}, prefix)
    return original, data, job, frame, interface, operation(original, job, tmp_path / "recovery", prefix)


def test_exact_interrupted_mac_continuity_and_immutable_originals(tmp_path, monkeypatch):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    original, data, job, frame, interface, contract = prepared_interruption(tmp_path, monkeypatch)
    uninterrupted = tmp_path / "uninterrupted"
    native_fit(original, data, job, frame, interface, uninterrupted)
    baseline = torch_load_checkpoint(last_path(uninterrupted, job), map_location="cpu")
    old_root = Path(contract["immutable_originals"]["root"])
    before = {str(p): science.sha_file(p) for p in old_root.rglob("*") if p.is_file()}
    result = recovery.train(contract, job["id"], device="cpu")
    resumed = torch_load_checkpoint(last_path(Path(contract["root"]) / "run" / job["id"], job), map_location="cpu")
    for key in ("model_state_dict", "model_state_sha256", "optimizer_state_dict", "rng_state",
                "train_loader_generator_state", "history", "best_state_dict", "best_state_sha256",
                "initial_state_sha256", "resume_identity", "source_revision_history", "best_epoch",
                "best_selection_value", "time_head_training_verification"):
        same_tree(baseline[key], resumed[key])
    assert result["schema"] == recovery.SCHEMA
    assert result["summary"]["completed_epochs"] == 4
    assert result["lineage"]["resume_validation"]["prefix_optimizer_steps"] == 2
    assert result["lineage"]["new_optimizer_steps"] == 2
    assert result["time_head_training_verification"]["optimizer_steps_checked"] == 4
    assert before == {str(p): science.sha_file(p) for p in old_root.rglob("*") if p.is_file()}
    effective = recovery.effective_scientific(contract)
    scientific = science.read(Path(contract["root"]) / "run" / job["id"] / "train_receipt.json")
    science.verify_train_receipt(effective, "5090", job, Path(contract["root"]), scientific)
    assert result["scientific_contract_sha256"] == science.sha_json(original)
    assert scientific["contract_sha256"] == science.sha_json(effective)
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    evaluator = science.evaluate_selected
    with patch("paper.scripts.quantity_comparison_runtime.configure_runtime",
               side_effect=lambda *_: configure_runtime("cpu", 4)), \
         patch.object(science, "evaluate_selected", side_effect=lambda *a, **kw:
                      evaluator(*a, **{**kw, "device": "cpu"})):
        replay = recovery.run_stage(contract, job["id"], "validation")
    assert replay["validation_replay_passed"] and replay["individual_predictions_match"]
    assert replay["metrics"] == result["validation_reference"]["metrics"]
    assert replay["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert before == {str(p): science.sha_file(p) for p in old_root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        recovery.train(contract, job["id"], device="cpu")


def test_completed_raf_is_only_copied_with_original_actual_exit_and_selected_bytes(tmp_path, monkeypatch):
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import training
    original, _, job, _, _ = make_fixture(tmp_path, monkeypatch, dataset="raf_spare_parts")
    evaluator = science.evaluate_selected
    with patch("paper.scripts.quantity_comparison_runtime.configure_runtime",
               side_effect=lambda *_: configure_runtime("cpu", 4)), \
         patch.object(science, "evaluate_selected", side_effect=lambda *a, **kw:
                      evaluator(*a, **{**kw, "device": "cpu"})):
        trained = science.run_stage(original, "5090", job["id"], "train")
    old_root = Path(original["hosts"]["5090"]["root"])
    old_folder = old_root / "run" / job["id"]
    exited = {"job_id": job["id"], "stage": "train", "actual_os_returncode": 0,
              "timeout": False, "pid": 12, "start_ticks": 123,
              "observed_exit_at": "fabricated-original-observation"}
    science.write_exclusive(old_folder / "train_exit.json", exited)
    contract = operation(original, job, tmp_path / "reference", last_path(old_folder, job))
    item = next(j for j in contract["jobs"] if j["id"] == job["id"])
    item.pop("origin_resume_checkpoint")
    item.pop("resume_epoch")
    item.update(origin_train_receipt={"path": str(old_folder / "train_receipt.json"),
                                      "sha256": science.sha_file(old_folder / "train_receipt.json")},
                origin_train_exit={"path": str(old_folder / "train_exit.json"),
                                   "sha256": science.sha_file(old_folder / "train_exit.json")})
    before = {str(p): science.sha_file(p) for p in old_root.rglob("*") if p.is_file()}
    with patch.object(training, "train_one", side_effect=AssertionError("Completed fit retrained")), \
         patch.object(science, "evaluate_selected", side_effect=AssertionError("Reference reevaluated")):
        result = recovery.run_stage(contract, job["id"], "reference")
    assert result["lineage"]["reference_retrained"] is False
    assert result["checkpoint_sha256"] == trained["checkpoint_sha256"]
    assert science.sha_file(result["selected_checkpoint"]) == trained["checkpoint_sha256"]
    assert science.read(Path(contract["root"]) / "run" / job["id"] / "train_exit.json") == exited
    assert before == {str(p): science.sha_file(p) for p in old_root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        recovery.reference(contract, job["id"])


def test_prefix_with_no_additional_epochs_exports_best_without_claiming_new_steps(tmp_path, monkeypatch):
    from paper.scripts.count_aware_tpp_backbone import training
    _, _, job, _, _, contract = prepared_interruption(tmp_path, monkeypatch)
    with patch.object(training, "early_stopping_exhausted", return_value=True):
        result = recovery.train(contract, job["id"], device="cpu")
    assert result["summary"]["completed_epochs"] == 2
    assert result["lineage"]["new_optimizer_steps"] == 0
    assert result["time_head_training_verification"]["optimizer_steps_checked"] == 2


@pytest.mark.parametrize("mutation", ["proof_state", "initial", "counter", "flags"])
def test_invalid_joint_prefix_proof_is_rejected_before_any_new_optimizer_step(tmp_path, monkeypatch, mutation):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    from paper.scripts.count_aware_tpp_backbone import training
    _, _, job, _, _, contract = prepared_interruption(tmp_path, monkeypatch)
    selected = next(j for j in contract["jobs"] if j["id"] == job["id"])
    path = Path(selected["origin_resume_checkpoint"]["path"])
    value = torch_load_checkpoint(path, map_location="cpu")
    proof = value["time_head_training_verification"]
    if mutation == "proof_state":
        proof["checkpoint_time_head_sha256"] = "f" * 64
    elif mutation == "initial":
        proof["initial_time_head_sha256"] = "f" * 64
    elif mutation == "counter":
        proof["optimizer_steps_checked"] = 0
    else:
        proof["nonzero_time_head_gradient_observed"] = False
    # This is explicitly a fabricated corrupt source and a freshly resealed
    # binding, so validation must inspect content rather than just its bytes.
    torch.save(value, path)
    selected["origin_resume_checkpoint"]["sha256"] = science.sha_file(path)
    contract["immutable_originals"]["files"][str(path.relative_to(Path(contract["immutable_originals"]["root"])))] = science.sha_file(path)
    with patch.object(training, "train_epoch_with_telemetry", side_effect=AssertionError("New optimizer step")):
        with pytest.raises(ValueError):
            recovery.train(contract, job["id"], device="cpu")


@pytest.mark.parametrize("mutation", ["alpha", "mode", "seed", "source", "old_root", "duplicate", "matrix"])
def test_changed_scope_rejected_before_native_data(tmp_path, monkeypatch, mutation):
    original, _, job, _, _, contract = scope_fixture(tmp_path, monkeypatch)
    if mutation == "alpha":
        contract["jobs"][0]["time_alpha"] = 0
    elif mutation == "mode":
        next(j for j in contract["jobs"] if j["id"] == job["id"])["recovery_mode"] = "reference"
    elif mutation == "seed":
        contract["jobs"][0]["seed"] = 72
    elif mutation == "source":
        contract["source"]["files_sha256"] = "changed"
    elif mutation == "old_root":
        contract["root"] = contract["immutable_originals"]["root"]
        contract["source_root"] = str(Path(contract["root"]) / "source")
    elif mutation == "duplicate":
        contract["jobs"][0] = deepcopy(contract["jobs"][1])
    else:
        contract["test_jobs"].pop()
    with patch.object(science, "check_native_runtime", side_effect=AssertionError("Native data accessed")):
        with pytest.raises(ValueError):
            recovery.select_job(contract, job["id"])


def test_tampered_origin_and_unsafe_destination_rejected(tmp_path, monkeypatch):
    _, _, job, _, _, contract = scope_fixture(tmp_path, monkeypatch)
    selected = next(j for j in contract["jobs"] if j["id"] == job["id"])
    binding = selected["origin_resume_checkpoint"]
    with pytest.raises(ValueError, match="escapes"):
        recovery.copy_origin(contract, binding, Path(contract["immutable_originals"]["root"]) / "forbidden.pt")
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        recovery.verify_origin(contract, binding)


def test_test_stage_requires_complete_original_gate_before_native_or_input(tmp_path, monkeypatch):
    _, _, job, _, _, contract = scope_fixture(tmp_path, monkeypatch)
    with patch.object(science, "check_native_runtime", side_effect=AssertionError("CUDA accessed")), \
         patch.object(science, "prepare_evaluation_frame", side_effect=AssertionError("Held-out data accessed")):
        with pytest.raises(FileNotFoundError):
            recovery.run_stage(contract, job["id"], "test")


def test_cli_rejects_contract_tamper_before_stage(tmp_path, capsys):
    path = tmp_path / "contract.json"
    science.write_exclusive(path, {})
    with patch.object(recovery, "run_stage", side_effect=AssertionError("stage started")):
        with pytest.raises(ValueError, match="SHA"):
            recovery.main(("--contract", str(path), "--contract-sha", "0" * 64,
                           "--phase", "train", "--job-id", "fabricated"))
