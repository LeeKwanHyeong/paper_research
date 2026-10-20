"""New execution worker checks, with fabricated rows and CPU fits only."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from paper.scripts import run_titantpp_shared_history_raw9 as worker


def contract_fixture(root):
    parent = json.loads((worker.ROOT / "search_artifacts/titantpp_mlp16_late_path18_20261019_v2/execution_contract.json").read_text())
    result = {"schema": worker.CONTRACT_SCHEMA, "scope": "execution",
        "execution_authorization": {"real_data_training": True, "real_data_inference": True,
            "remote_execution": True, "test_evaluation": False},
        "hosts": {"5080": {"root": str(root)}}, "datasets": deepcopy(parent["datasets"]),
        "training": deepcopy(parent["training"]), "source": {
            "base_git_revision": "fabricated_execution_only", "files_sha256": "a" * 64},
        "architecture": {"scope": "fabricated_execution_contract_only"}, "design_sha256": "b" * 64,
        "common_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
        "data_variant": "original_uncapped", "quantity_capping_applied": False,
        "jobs": [{"id": f"{dataset}__{arm}__seed{seed}__losslog_mse", "host": "5080",
                  "dataset": dataset, "arm": arm, "seed": seed, "time_input": "log1p",
                  "loss_mode": "log_mse", "time_alpha": 1.0}
                 for dataset in worker.DATASETS for arm in worker.ARMS for seed in worker.SEEDS]}
    for data in result["datasets"]:
        data.pop("cap_value", None)
        data["quantity_boundaries_all_train_rows"] = {
            "yellow_trip_hourly": [7, 686, 1562, 3449],
            "raf_spare_parts": [2, 30, 60, 200],
            "intermittent_frozen_5000": [2, 31, 46, 187]}[data["dataset_id"]]
    return result


@pytest.fixture
def contract(tmp_path):
    return contract_fixture(tmp_path / "isolated_execution")


def test_approved_matrix_and_common_condition(contract):
    for job in contract["jobs"]:
        selected, data, root, folder = worker.select_job(contract, "5080", job["id"])
        assert selected == job and data["dataset_id"] == job["dataset"]
        assert folder == root / "run" / job["id"]
    assert len(contract["jobs"]) == 9
    assert worker.candidate_labels(worker.ARMS[0])["architecture_metadata"]["active_prefix_parameter_counts"] == [
        0, 2880, 3840, 4800, 5760, 6720, 7680, 8640, 9600]


@pytest.mark.parametrize("key,value", [("time_input", "raw"), ("loss_mode", "raw_mae"), ("time_alpha", 0.0)])
def test_unapproved_common_input_or_loss_rejected_before_data_access(contract, key, value):
    contract["common_condition"][key] = value
    with patch.object(Path, "read_bytes", side_effect=AssertionError("Dataset read attempted")):
        with pytest.raises(ValueError):
            worker.select_job(contract, "5080", contract["jobs"][0]["id"])


@pytest.mark.parametrize("mutation", ["host", "seed", "duplicate", "baseline", "schema", "scope", "test"])
def test_matrix_host_or_authorization_tampering_rejected(contract, mutation):
    if mutation == "host":
        contract["jobs"][0]["host"] = "5090"
    elif mutation == "seed":
        contract["jobs"][0]["seed"] = 32
    elif mutation == "duplicate":
        contract["jobs"][1] = deepcopy(contract["jobs"][0])
    elif mutation == "baseline":
        contract["jobs"][0]["arm"] = "titantpp_history_mlp_width16"
    elif mutation in ("schema", "scope"):
        contract[mutation] = "preparation_only"
    else:
        contract["execution_authorization"]["test_evaluation"] = True
    with pytest.raises(ValueError):
        worker.select_job(contract, "5080", contract["jobs"][0]["id"])


@pytest.mark.parametrize("stage", ["test", "resume", "deploy", None])
def test_unsupported_stage_rejected_before_dependencies_or_data(contract, stage):
    with patch.object(worker, "verify_frozen_dependencies", side_effect=AssertionError("Unexpected read")):
        with pytest.raises(ValueError):
            worker.run_stage(contract, "5080", contract["jobs"][0]["id"], stage)


def test_cli_rejects_5090_and_test_before_contract_read():
    with patch.object(worker, "read", side_effect=AssertionError("Unexpected contract read")):
        for extra in (("--host", "5090", "--stage", "train"),
                      ("--host", "5080", "--stage", "test")):
            with pytest.raises(SystemExit):
                worker.main(("--contract", "unused", *extra))


def test_cli_stdout_is_stable_json_and_engine_output_is_stderr(contract, capsys, tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    def synthetic(*args):
        print("[fabricated synthetic progress]")
        return {"status": "passed", "real_data_loaded": False}
    with patch.object(worker, "synthetic_check", side_effect=synthetic):
        worker.main(("--contract", str(path), "--host", "5080", "--stage", "synthetic"))
    streams = capsys.readouterr()
    assert json.loads(streams.out) == {"status": "passed", "real_data_loaded": False}
    assert streams.err.strip() == "[fabricated synthetic progress]"


def test_checkpoint_budget_and_optimizer_are_unchanged(contract, tmp_path):
    job, data, _, _ = worker.select_job(contract, "5080", contract["jobs"][0]["id"])
    with worker.candidate_factory(job["arm"]):
        args = worker.training_args(contract, data, tmp_path, job["arm"])
    assert (args.epochs, args.min_epochs, args.early_stopping_patience) == (300, 40, 40)
    assert (args.lr, args.weight_decay, args.grad_clip, args.batch_size) == (.001, .01, 1., 128)
    assert args.time_alpha == 1 and args.loss_mode == "log_mse"
    assert args.model_role == worker.prepared.role_for_arm(job["arm"])
    assert args.checkpoint_monitor == worker.MONITOR
    assert args.data_sha256 == data["inherited_data_identity"]["data"]["sha256"]


def test_dependencies_frozen():
    worker.verify_frozen_dependencies()


@pytest.fixture(scope="module")
def synthetic():
    return worker.synthetic_check(device="cpu")


def test_synthetic_qualification_joint_time_learning_and_eval_state(synthetic):
    assert synthetic["status"] == "passed" and synthetic["device"] == "cpu"
    assert set(synthetic["arms"]) == set(worker.ARMS)
    assert all(synthetic["checks"].values())
    assert not synthetic["real_data_loaded"] and not synthetic["held_out_test_evaluated"]
    assert len(synthetic["parameter_count"]) == 1
    for arm in worker.ARMS:
        proof = worker.verify_time_head_proof(synthetic["time_head_training_verification"][arm])
        assert proof["optimizer_steps_checked"] == 2
        assert synthetic["cuda_memory_peaks"][arm]["full_training_shape"] is None


def test_per_arm_synthetic_runtime_interface(contract):
    result = worker.synthetic_condition(contract, "5080", arm=worker.ARMS[0], device="cpu")
    assert result["arms"] == [worker.ARMS[0]] and result["candidate_name"] == worker.ARMS[0]
    assert result["cuda_memory_peak"] == result["cuda_memory_peaks"][worker.ARMS[0]]
    assert result["history_relation_sharing"] == "shared"
    assert not result["quantity_only"] and result["time_head_trained"]
    with pytest.raises(ValueError):
        worker.synthetic_condition(contract, "5080", "raw_mse", arm=worker.ARMS[0], device="cpu")


@pytest.mark.parametrize("arm", worker.ARMS)
def test_actual_new_stage_fabricated_engine_save_and_fixed_checkpoint_validation(tmp_path, arm):
    """Exercise new stage and replay with actual train_one; no Cap parquet read."""
    import polars as pl
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    root = tmp_path / "execution"
    root.mkdir()
    fixture = root / "fabricated"
    fixture.mkdir()
    args, _, data, interface, _ = worker.prepared._engine_fixture(fixture, arm)
    # Fabricated quantities deliberately exceed the tail boundary; no Cap.
    frame = prepare_count_frame(pl.read_parquet(args.data))
    args.time_alpha = 1.0
    for name, value in worker.input_identity().items():
        setattr(args, name, value)
    interface.update(worker.input_identity())
    contract = contract_fixture(root)
    contract["source"]["base_git_revision"] = args.source_revision
    base_data = next(value for value in contract["datasets"] if value["dataset_id"] == "raf_spare_parts")
    data = {**base_data, **data, "dataset_id": "raf_spare_parts",
            "quantity_boundaries_all_train_rows": [2.0, 5.0, 8.0, 10.0],
            "loader": {**base_data["loader"], **data["loader"]}}
    data["model"].update(lambda_log_qty=1., lambda_tail=0., time_head_lr_multiplier=1.)
    data["inherited_data_identity"] = {
        "data": {"path": str(args.data), "sha256": args.data_sha256},
        "split_manifest": {"path": str(args.split_manifest), "sha256": args.split_manifest_sha256},
        "populations": {}}
    for split in ("train", "validation"):
        loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=84,
                             max_seq_len=84, shuffle=False, generator=None)
        data["inherited_data_identity"]["populations"][split] = worker.population_identity(loader.dataset, split)[1]
    contract["datasets"] = [data if value["dataset_id"] == "raf_spare_parts" else value
                            for value in contract["datasets"]]
    job = next(value for value in contract["jobs"]
               if value["dataset"] == "raf_spare_parts" and value["arm"] == arm and value["seed"] == 42)
    folder = root / "run" / job["id"]
    args.output_dir = folder
    interface.update(source_files_sha256=contract["source"]["files_sha256"],
                     execution_contract_sha256=worker.sha_json(contract))
    # CPU/2epochs is a deliberate synthetic fixture, never a production route.
    with patch("paper.scripts.quantity_comparison_runtime.configure_runtime",
               side_effect=lambda device, threads: configure_runtime("cpu", threads)), \
         patch.object(worker, "training_args", return_value=args), \
         patch.object(worker, "training_interface", return_value=interface), \
         patch.object(base, "prepare_admitted_data", return_value=(frame, {"fabricated_only": True})):
        original_evaluate = worker.evaluate_selected
        with patch.object(worker, "evaluate_selected",
                          side_effect=lambda *values, **kw: original_evaluate(*values, **{**kw, "device": "cpu"})):
            trained = worker.run_stage(contract, "5080", job["id"], "train")
            validated = worker.run_stage(contract, "5080", job["id"], "validation")
        for stage in ("train", "validation"):
            with pytest.raises(ValueError):
                worker.run_stage(contract, "5080", job["id"], stage)
    assert trained["summary"]["completed_epochs"] == 2
    assert trained["selected_epoch"] == validated["selected_epoch"]
    assert validated["validation_replay_passed"] and validated["parameters_unchanged"]
    assert trained["validation_reference"]["metrics"] == validated["metrics"]
    assert validated["metrics"]["overall"]["count"] == 8
    assert validated["metrics"]["tail"]["count"] == 4
    assert "cap_hit" not in validated["metrics"]
    assert trained["checkpoint_sha256"] == validated["checkpoint_sha256"]
    worker.verify_train_receipt(contract, "5080", job, root, trained)
    worker.verify_inventory(root, validated["files"])
    assert worker.sha_file(trained["selected_checkpoint"]) == trained["checkpoint_sha256"]
    assert worker.verify_time_head_proof(trained["time_head_training_verification"])["optimizer_steps_checked"] == 2
    assert not trained["held_out_test_evaluated"] and not validated["held_out_test_evaluated"]


def test_completed_receipt_and_prior_intent_cannot_be_overwritten(contract):
    job = contract["jobs"][0]
    folder = Path(contract["hosts"]["5080"]["root"]) / "run" / job["id"]
    folder.mkdir(parents=True)
    for stage in ("train", "validation"):
        path = folder / f"{stage}.intent.json"
        path.write_text("preserved original attempt")
        with pytest.raises(FileExistsError):
            worker.run_stage(contract, "5080", job["id"], stage)
        assert path.read_text() == "preserved original attempt"


@pytest.mark.parametrize("mutation", ["cap_value", "cap_variant", "capping", "independent"])
def test_uncapped_shared_only_contract_rejects_capping_or_other_arm(contract, mutation):
    if mutation == "cap_value":
        contract["datasets"][0]["cap_value"] = 65.0
    elif mutation == "cap_variant":
        contract["data_variant"] = "cap"
    elif mutation == "capping":
        contract["quantity_capping_applied"] = True
    else:
        contract["jobs"][0]["arm"] = "titantpp_history_mlp_prefix_capacity_v1"
    with pytest.raises(ValueError):
        worker.select_job(contract, "5080", contract["jobs"][0]["id"])


def test_uncapped_raw_metrics_hand_calculation_and_tail():
    import math
    metric = worker.MetricAccumulator(200)
    metric.add([1, 200, 1000], [3, 190, 800], [1, 2, 3])
    result = metric.finish()
    assert result["overall"]["qty_mae"] == pytest.approx(212/3)
    assert result["overall"]["qty_rmse"] == pytest.approx(math.sqrt(40104/3))
    assert result["overall"]["time_nll"] == 2
    assert result["tail"]["count"] == 1
    assert result["tail"]["qty_rmse"] == result["tail"]["qty_mae"] == 200
    assert result["tail"]["time_nll"] == 3
    assert result["tail_threshold"] == 200
    worker.compare_metrics(result, result)
    altered = deepcopy(result)
    altered["tail"]["qty_mae"] += .1
    with pytest.raises(ValueError):
        worker.compare_metrics(result, altered)


@pytest.mark.parametrize("values", [([-1], [0], [1]), ([1], [float("inf")], [1]),
                                    ([1], [1], [float("nan")]), ([1, 2], [1], [1])])
def test_uncapped_metric_nonfinite_negative_and_shape_rejection(values):
    metric = worker.MetricAccumulator(200)
    with pytest.raises(ValueError):
        metric.add(*values)


def test_quantity_only_guards_are_never_entered_and_hooks_restore():
    from paper.scripts.count_aware_tpp_backbone import core, training
    from models.TPPs import CountAwareFactory as factory
    originals = core.target_outputs, training.target_outputs, training.build_model, factory.build_count_aware_model
    with patch.object(worker.prepared, "TimeHeadGuard", side_effect=AssertionError("Quantity-only guard entered")), \
         patch.object(worker.prepared, "runtime_hooks", side_effect=AssertionError("Cap hook entered")):
        with worker.runtime_hooks(worker.ARMS[0]):
            model, encoder = worker.build_model(worker.prepared.fabricated_data(), worker.ARMS[0])
            assert encoder["time_alpha"] == 1.0 and encoder["time_head_trained"]
            batch = worker.prepared.fabricated_batch()
            result = core.target_outputs(model, *[batch[index] for index in (0, 2, 1)], lambda_log_qty=1.)
            torch.testing.assert_close(result["joint_loss"], result["time_loss"] + result["log_qty_loss"])
            assert result["joint_loss"] is not result["quantity_train_loss"]
    assert (core.target_outputs, training.target_outputs, training.build_model, factory.build_count_aware_model) == originals
    with pytest.raises(RuntimeError):
        with worker.runtime_hooks(worker.ARMS[0]):
            raise RuntimeError("fabricated interrupted stage")
    assert (core.target_outputs, training.target_outputs, training.build_model, factory.build_count_aware_model) == originals


def test_joint_proof_rejects_unchanged_or_forged_parameter_state(synthetic):
    arm = worker.ARMS[0]
    proof = synthetic["time_head_training_verification"][arm]
    assert proof["initial_time_head_sha256"] != proof["checkpoint_time_head_sha256"]
    assert proof["finite_time_head_gradient_observed"] and proof["optimizer_state_present"]
    for key, value in (("time_alpha", 0), ("optimizer_steps_checked", 0),
                       ("joint_loss_includes_time", False), ("nonzero_time_head_gradient_observed", False),
                       ("checkpoint_time_head_sha256", proof["initial_time_head_sha256"])):
        altered = {**proof, key: value}
        with pytest.raises(ValueError):
            worker.verify_time_head_proof(altered)


@pytest.mark.parametrize('seed', worker.SEEDS)
def test_same_shared15_architecture_initialization_and_only_joint_metadata_change(seed):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    data, arm = worker.prepared.fabricated_data(), worker.ARMS[0]
    torch.manual_seed(seed)
    cap_model, cap_meta = worker.prepared.build_model(data, arm)
    torch.manual_seed(seed)
    raw_model, raw_meta = worker.build_model(data, arm)
    assert canonical_state_dict_sha256(cap_model.state_dict()) == canonical_state_dict_sha256(raw_model.state_dict())
    assert cap_meta['time_alpha'] == 0 and raw_meta['time_alpha'] == 1
    assert {key: cap_meta[key] for key in worker.metadata(mode='shared')} == worker.metadata(mode='shared')
    assert {key: raw_meta[key] for key in worker.metadata(mode='shared')} == worker.metadata(mode='shared')
    assert sum(p.numel() for p in raw_model.multilag_detail.parameters()) == 9600
