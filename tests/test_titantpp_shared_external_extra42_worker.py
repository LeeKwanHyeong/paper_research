"""Extra42 admission, native-gate boundaries and fabricated CPU engine integration."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from paper.scripts import run_titantpp_shared_external_extra42 as worker

COUNTS = dict(zip(worker.ARMS, (99459, 25347, 100355, 30211, 112835, 8003, 31491)))


def contract_fixture(root):
    parent = json.loads((worker.ROOT / "search_artifacts/titantpp_mlp16_late_path18_20261019_v2/execution_contract.json").read_text())
    result = {"schema": worker.CONTRACT_SCHEMA, "scope": "execution",
        "execution_authorization": {"real_data_training": True, "real_data_inference": True,
            "remote_execution": True, "test_evaluation": True},
        "hosts": {"5080": {"root": str(root)}}, "datasets": deepcopy(parent["datasets"]),
        "training": deepcopy(parent["training"]), "source": {
            "base_git_revision": "fabricated_execution_only", "files_sha256": "a" * 64},
        "architecture": {"scope": "fabricated_only"}, "design_sha256": "b" * 64,
        "common_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
        "data_variant": "original_uncapped", "quantity_capping_applied": False,
        "jobs": [{"id": f"{dataset}__{arm}__seed{seed}", "host": "5080", "dataset": dataset,
                  "arm": arm, "seed": seed, "time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0}
                 for dataset in worker.DATASETS for arm in worker.ARMS for seed in worker.SEEDS]}
    for data in result["datasets"]:
        data.pop("cap_value", None)
        data["quantity_boundaries_all_train_rows"] = {
            "yellow_trip_hourly": [7, 686, 1562, 3449], "raf_spare_parts": [2, 30, 60, 200],
            "intermittent_frozen_5000": [2, 31, 46, 187]}[data["dataset_id"]]
        data["full_data_identity"] = {"data": {"path": "never-read-full-data", "sha256": "f" * 64},
            "populations": {"test": {"target_count": 1, "target_identity_sha256": "c" * 64,
                                     "target_quantity_sha256": "d" * 64}}}
    return result


@pytest.fixture
def contract(tmp_path):
    return contract_fixture(tmp_path / "isolated_extra42")


def test_42_new_seeds_only_and_native_scientific_settings(contract, tmp_path):
    for job in contract["jobs"]:
        selected, data, root, folder = worker.select_job(contract, "5080", job["id"])
        assert selected == job and data["dataset_id"] == job["dataset"]
        assert folder == root / "run" / job["id"]
        with worker.candidate_factory(job["arm"]):
            args = worker.training_args(contract, data, tmp_path, job["arm"])
        assert (args.epochs, args.min_epochs, args.early_stopping_patience) == (300, 40, 40)
        assert (args.batch_size, args.lr, args.weight_decay, args.grad_clip) == (128, .001, .01, 1.)
        assert args.model_role == worker.role_for(job["arm"])
        assert args.time_alpha == 1. and args.loss_mode == "log_mse"
    assert len(contract["jobs"]) == 42 and worker.SEEDS == (72, 82)


@pytest.mark.parametrize("mutation", ["old_seed", "duplicate", "host", "cap", "alpha", "warm", "window", "test_permission"])
def test_tampered_scientific_scope_rejected_before_data(contract, mutation):
    if mutation == "old_seed":
        contract["jobs"][0]["seed"] = 42
    elif mutation == "duplicate":
        contract["jobs"][1] = deepcopy(contract["jobs"][0])
    elif mutation == "host":
        contract["jobs"][0]["host"] = "5090"
    elif mutation == "cap":
        contract["datasets"][0]["cap_value"] = 12
    elif mutation == "alpha":
        contract["common_condition"]["time_alpha"] = 0
    elif mutation == "warm":
        contract["training"]["warm_start"] = True
    elif mutation == "window":
        next(d for d in contract["datasets"] if d["dataset_id"] == contract["jobs"][0]["dataset"])["loader"]["max_seq_len"] = 32
    else:
        contract["execution_authorization"]["test_evaluation"] = False
    with patch.object(Path, "read_bytes", side_effect=AssertionError("Unexpected data access")):
        with pytest.raises(ValueError):
            worker.select_job(contract, "5080", contract["jobs"][0]["id"])


def test_missing_validation_gate_blocks_test_before_input_access(contract):
    with patch.object(worker, "prepare_evaluation_frame", side_effect=AssertionError("Held-out input opened")), \
         patch.object(worker, "check_native_runtime", side_effect=AssertionError("GPU initialized")):
        with pytest.raises(FileNotFoundError):
            worker.run_stage(contract, "5080", contract["jobs"][0]["id"], "test")


def test_incomplete_gate_blocks_test_and_forged_exit_rejected(contract):
    root = Path(contract["hosts"]["5080"]["root"])
    root.mkdir()
    gate = {"status": "passed", "host": "5080", "contract_sha256": worker.sha_json(contract), "jobs": {}}
    worker.write_exclusive(root / "validation_gate.json", gate)
    with pytest.raises(ValueError, match="All 42"):
        worker.verify_validation_gate(contract, "5080", root)
    gate["jobs"] = {job["id"]: {} for job in contract["jobs"]}
    folder = root / "run" / contract["jobs"][0]["id"]
    folder.mkdir(parents=True)
    receipt_path, exit_path = folder / "validation_receipt.json", folder / "validation_exit.json"
    worker.write_exclusive(receipt_path, {})
    worker.write_exclusive(exit_path, {"job_id": contract["jobs"][0]["id"], "stage": "validation",
        "actual_os_returncode": False, "timeout": False, "pid": 123, "start_ticks": 12,
        "observed_exit_at": "fabricated"})
    gate["jobs"][contract["jobs"][0]["id"]] = {
        "validation_receipt_sha256": worker.sha_file(receipt_path),
        "validation_exit_sha256": worker.sha_file(exit_path)}
    (root / "validation_gate.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="actual child exit0"):
        worker.verify_validation_gate(contract, "5080", root)


@pytest.fixture(scope="module")
def synthetic():
    return worker.synthetic_check(device="cpu")


def test_all7_synthetic_gradients_causality_checkpoint_and_state(synthetic):
    assert synthetic["status"] == "passed" and all(synthetic["checks"].values())
    assert synthetic["parameter_count"] == COUNTS
    assert not synthetic["real_data_loaded"] and not synthetic["held_out_test_evaluated"]
    for arm in worker.ARMS:
        proof = worker.verify_time_head_proof(synthetic["time_head_training_verification"][arm])
        assert proof["optimizer_steps_checked"] == 2
        assert proof["all_trainable_parameters_in_optimizer"]
        assert set(worker.TIME_HEAD_NAMES) <= set(proof["trainable_parameter_names"])


def test_external_routes_preserve_existing_constructors_and_metadata():
    from models.TPPs import CountAwareFactory as factory
    old = factory.build_count_aware_model
    for arm in worker.EXTERNAL_ARMS:
        data = worker.prepared.fabricated_data()
        with worker.runtime_hooks(arm):
            assert factory.build_count_aware_model is old
            model, meta = worker.build_model(data, arm)
        assert type(model).__name__ == {"rmtpp": "CountAwareRMTPP", "thp": "CountAwareTHP", "nhp": "CountAwareNHP",
            "sahp": "CountAwareSAHP", "s2p2_matched_head": "CountAwareS2P2", "attnhp_matched_head": "CountAwareAttNHP"}[arm]
        assert sum(p.numel() for p in model.parameters()) == COUNTS[arm]
        assert meta["time_head"]["mode"] == "heteroscedastic_lognormal_duration"


def test_optimizer_omitting_backbone_parameter_rejected():
    model, _ = worker.build_model(worker.prepared.fabricated_data(), "rmtpp")
    tracker = worker.JointTimeHeadTracker(model)
    optimizer = torch.optim.AdamW([p for _, p in model.time_head_named_parameters()], lr=.001)
    with pytest.raises(ValueError, match="omitted or added"):
        tracker.bind_optimizer(optimizer)


def test_cli_alias_returns_single_stable_json(contract, capsys, tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    def synthetic(*args, **kwargs):
        print("fabricated progress")
        return {"arm": kwargs["arm"], "status": "passed"}
    with patch.object(worker, "synthetic_check", side_effect=synthetic):
        worker.main(("--contract", str(path), "--host", "5080", "--mode", "synthetic", "--arm", "nhp"))
    streams = capsys.readouterr()
    assert json.loads(streams.out) == {"arm": "nhp", "status": "passed"}
    assert streams.err.strip() == "fabricated progress"


@pytest.mark.parametrize("arm", worker.ARMS)
def test_fabricated_actual_engine_checkpoint_selection_and_native_replay(tmp_path, arm):
    import polars as pl
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    root = tmp_path / "execution"
    fixture = root / "fabricated"
    fixture.mkdir(parents=True)
    args, _, fixture_data, interface, _ = worker.prepared._engine_fixture(fixture, worker.SHARED_ARM)
    frame = prepare_count_frame(pl.read_parquet(args.data))
    args.model_role = worker.role_for(arm)
    args.time_alpha = 1.0
    for key, value in worker.input_identity().items():
        setattr(args, key, value)
    interface.update(worker.input_identity())
    contract = contract_fixture(root)
    contract["source"]["base_git_revision"] = args.source_revision
    data = next(value for value in contract["datasets"] if value["dataset_id"] == "raf_spare_parts")
    data.update({**fixture_data, "dataset_id": "raf_spare_parts",
                 "quantity_boundaries_all_train_rows": [2., 5., 8., 10.]})
    data["loader"].update(batch_size=128, lookback_weeks=84, max_seq_len=84, num_workers=0,
                          drop_last=False, train_shuffle=True, validation_shuffle=False)
    data["model"].update(lambda_log_qty=1., lambda_tail=0., time_head_lr_multiplier=1.)
    data["inherited_data_identity"] = {
        "data": {"path": str(args.data), "sha256": args.data_sha256},
        "split_manifest": {"path": str(args.split_manifest), "sha256": args.split_manifest_sha256},
        "populations": {}}
    for split in ("train", "validation"):
        loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=84,
                             max_seq_len=84, shuffle=False, generator=None)
        data["inherited_data_identity"]["populations"][split] = worker.population_identity(loader.dataset, split)[1]
    # Fabricated held-out rows exercise the frozen Test path without opening
    # project held-out data or weakening the production all-42 gate.
    raw = pl.read_parquet(args.data)
    extra = pl.DataFrame([{"oper_part_no": entity, "seq": seq, "delta_t": 2,
        "demand_qty": quantity, "chronological_split": "test"}
        for entity in raw["oper_part_no"].unique().to_list()
        for seq, quantity in ((11, 13.), (12, 7.))])
    full_frame = prepare_count_frame(pl.concat([raw, extra]).sort(["oper_part_no", "seq"]))
    full_path = fixture / "fabricated_full.parquet"
    full_frame.write_parquet(full_path)
    full_loader = make_loader(full_frame, target_split="test", batch_size=128,
                             lookback_weeks=84, max_seq_len=84, shuffle=False, generator=None)
    data["full_data_identity"] = {"data": {"path": str(full_path), "sha256": worker.sha_file(full_path)},
        "populations": {"test": worker.population_identity(full_loader.dataset, "test")[1]}}
    model, encoder = worker.build_model(data, arm)
    contract["frozen_conditions"] = {"models": {arm: {
        "expected_encoder_config_by_dataset": {"raf_spare_parts": encoder},
        "expected_parameter_count_by_dataset": {"raf_spare_parts": sum(p.numel() for p in model.parameters())}}}}
    job = next(value for value in contract["jobs"] if value["dataset"] == "raf_spare_parts"
               and value["arm"] == arm and value["seed"] == 72)
    args.output_dir = root / "run" / job["id"]
    interface.update(source_files_sha256=contract["source"]["files_sha256"],
                     execution_contract_sha256=worker.sha_json(contract))
    with patch("paper.scripts.quantity_comparison_runtime.configure_runtime",
               side_effect=lambda device, threads: configure_runtime("cpu", threads)), \
         patch.object(worker, "check_native_runtime", return_value={"fabricated_only": True}), \
         patch.object(worker, "training_args", return_value=args), \
         patch.object(worker, "training_interface", return_value=interface), \
         patch.object(base, "prepare_admitted_data", return_value=(frame, {"fabricated_only": True})):
        evaluate = worker.evaluate_selected
        with patch.object(worker, "evaluate_selected", side_effect=lambda *values, **kw:
                          evaluate(*values, **{**kw, "device": "cpu"})):
            trained = worker.run_stage(contract, "5080", job["id"], "train")
            replay = worker.run_stage(contract, "5080", job["id"], "validation")
            with patch.object(worker, "verify_validation_gate", return_value={"fabricated_only": True}) as gate:
                tested = worker.run_stage(contract, "5080", job["id"], "test")
                assert gate.call_count == 2
    assert trained["summary"]["completed_epochs"] == 2
    assert trained["selected_epoch"] == replay["selected_epoch"]
    assert trained["validation_reference"]["metrics"] == replay["metrics"]
    assert replay["prediction_rows"] == 8 and replay["metrics"]["tail"]["count"] == 4
    assert replay["parameters_unchanged"] and replay["context_unchanged"] and replay["native_runtime_checked"]
    assert replay["validation_replay_passed"] and replay["individual_predictions_match"]
    assert not trained["held_out_test_evaluated"] and not replay["held_out_test_evaluated"]
    assert trained["checkpoint_sha256"] == replay["checkpoint_sha256"]
    assert tested["held_out_test_evaluated"] and tested["stage"] == "test"
    assert tested["prediction_rows"] == 8 and tested["metrics"]["tail"]["count"] == 4
    assert tested["selected_epoch"] == trained["selected_epoch"]
    assert tested["checkpoint_sha256"] == trained["checkpoint_sha256"]
    assert tested["state_sha256"] == trained["state_sha256"]
    assert not tested["validation_replay_passed"] and tested["parameters_unchanged"]
    proof = worker.verify_time_head_proof(trained["time_head_training_verification"])
    assert proof["all_trainable_parameters_in_optimizer"]
    worker.verify_train_receipt(contract, "5080", job, root, trained)
    worker.verify_inventory(root, replay["files"])
    for stage in ("train", "validation"):
        with pytest.raises(ValueError):
            worker.run_stage(contract, "5080", job["id"], stage)


def test_frozen_raw9_dependency_bytes_remain_unchanged():
    worker.verify_frozen_dependencies()
