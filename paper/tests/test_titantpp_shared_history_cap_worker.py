"""Fabricated-only checks of routing, objective and the real training engine."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from paper.scripts import run_titantpp_shared_history_cap as worker


@pytest.fixture(scope="module")
def completed_synthetic(tmp_path_factory):
    return worker.synthetic_check(output=tmp_path_factory.mktemp("shared_worker") / "checks")


@pytest.mark.parametrize("arm", worker.ARMS)
def test_factory_is_exact_candidate_with_unchanged_inputs_heads_and_memory(arm):
    model, meta = worker.build_model(worker.fabricated_data(), arm)
    assert type(model) is worker.CountAwareTitanSharedHistoryMLP
    assert model.history_sharing_mode == worker.MODE_BY_ARM[arm]
    assert meta["candidate_name"] == arm
    assert meta["history_relation_parameter_count"] == sum(p.numel() for p in model.multilag_detail.parameters())
    assert meta["time_head"] == model.time_head_contract()
    assert any(name.startswith("lmm.") for name in model.state_dict())
    dt = torch.tensor([[1.0, 32.0]])
    quantity = torch.tensor([[2.0, 10.0]])
    features = model.continuous_features(dt, quantity, torch.ones_like(dt, dtype=torch.bool))
    assert torch.equal(features[..., 0], torch.log1p(dt))
    assert torch.equal(features[..., 1], torch.log1p(quantity))
    assert worker.input_identity()["time_loss_graph_excluded"]


@pytest.mark.parametrize("dataset", worker.DATASETS)
@pytest.mark.parametrize("arm", worker.ARMS)
def test_three_cap_configurations_preserve_common_mlp16_tensors_rng_and_heads(dataset, arm):
    """Read only frozen configuration metadata; never load a Cap data file."""
    from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth
    root = Path(__file__).resolve().parents[2]
    contract_path = root / "search_artifacts/titantpp_mlp16_late_path18_20261019_v2/execution_contract.json"
    assert worker.sha_file(contract_path) == "67bfcad872b68886bf1de09879e031e055c70063bafbf4aa0db42b4ae5ca6fc7"
    contract = json.loads(contract_path.read_text())
    data = next(value for value in contract["datasets"] if value["dataset_id"] == dataset)
    config = {key: value for key, value in data["model"].items()
              if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    config.update(tail_threshold=46.0, tail_normalization_scale=46.0, tail_clip_cap=187.0, tail_huber_delta=1.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(52)
        baseline = CountAwareTitanHistoryWidth(**config,
            train_log_mean=data["statistics"]["train_log_mean"],
            train_log_std=data["statistics"]["train_log_std"],
            max_seq_len=data["loader"]["max_seq_len"], history_mlp_width=16)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(52)
        candidate, metadata = worker.build_model(data, arm)
        assert torch.equal(torch.get_rng_state(), expected_rng)
    expected, actual = baseline.state_dict(), candidate.state_dict()
    common = [name for name in expected if not name.startswith("multilag_detail.")
              and name != "history_width_identity"]
    assert common
    assert all(torch.equal(expected[name], actual[name]) for name in common)
    assert set(actual) - set(common) == {"shared_history_identity", *(
        name for name in actual if name.startswith("multilag_detail."))}
    assert candidate.time_head_contract() == baseline.time_head_contract() == metadata["time_head"]
    assert metadata["time_head"]["observation_likelihood"] == data["model"]["time_observation_contract"]
    assert metadata["time_head"]["time_scale"] == data["model"]["time_scale"]
    assert metadata["max_len"] == data["loader"]["max_seq_len"]
    assert metadata["time_input_transform"] == metadata["quantity_input_transform"] == worker.INPUT_TRANSFORM


@pytest.mark.parametrize("arm", worker.ARMS)
def test_actual_engine_two_epoch_selected_binary_same_target_validation_replay(completed_synthetic, arm):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    result = completed_synthetic["per_arm"][arm]
    assert result["actual_training_engine_integration_passed"]
    assert result["training_engine"].endswith("training.train_one")
    assert result["fabricated_epochs"] == 2
    assert result["selected_epoch"] == worker.select_best_epoch(result["history"])
    assert result["same_target_prediction_and_time_loss_replay_passed"]
    assert result["parameter_and_buffer_unchanged_during_replay"]
    assert result["selected_binary_sha256_unchanged"]
    checkpoint = Path(result["checkpoint"])
    assert worker.sha_file(checkpoint) == result["checkpoint_sha256"]
    payload = torch_load_checkpoint(checkpoint, map_location="cpu")
    assert worker.validate_checkpoint(payload, arm)
    assert payload["model_state_sha256"] == result["checkpoint_state_sha256"]
    assert payload["encoder_config"]["history_relation_sharing"] == worker.MODE_BY_ARM[arm]
    assert payload["resume_identity"]["arguments"]["model_role"] == worker.role_for_arm(arm)
    assert payload["interface_meta"]["loss_mode"] == "log_mse"
    assert len(result["six_metrics"]) == 6
    assert result["target_identity"]["target_count"] == 8
    proof = worker.verify_time_head_proof(result["time_head_no_training_verification"])
    assert proof["optimizer_steps_checked"] == 2
    assert proof["all_time_head_gradients_none"] and proof["optimizer_state_absent"]
    assert not result["real_data_loaded"] and not result["held_out_test_evaluated"]
    assert result["research_training_budget"] == {"epochs": 300, "min_epochs": 40, "patience": 40}


def test_preparation_check_has_zero_real_training_and_remote_execution(completed_synthetic):
    assert completed_synthetic["status"] == "passed"
    assert completed_synthetic["scope"] == "fabricated_cpu_preparation_only"
    assert set(completed_synthetic["per_arm"]) == set(worker.ARMS)
    assert completed_synthetic["new_real_data_training_count"] == 0
    assert completed_synthetic["remote_execution_count"] == 0
    assert not completed_synthetic["real_data_loaded"]
    assert not completed_synthetic["held_out_test_evaluated"]


@pytest.mark.parametrize("stage", ("train", "validation", "test", "synthetic", None))
def test_real_stages_fail_before_any_dataset_or_checkpoint_read(stage):
    contract = {"schema": worker.CONTRACT_SCHEMA, "scope": "preparation_only",
                "execution_authorization": {key: False for key in
                    ("real_data_training", "real_data_inference", "remote_execution", "test_evaluation")}}
    with patch.object(Path, "read_bytes", side_effect=AssertionError("Input read attempted")), \
         patch.object(Path, "read_text", side_effect=AssertionError("Input read attempted")):
        with pytest.raises(PermissionError):
            worker.run_stage(contract, "5090", "unused", stage)
        # Manually flipping flags cannot create an unimplemented execution path.
        contract["execution_authorization"] = {key: True for key in contract["execution_authorization"]}
        with pytest.raises(PermissionError):
            worker.run_stage(contract, "5090", "unused", stage)


def test_only_synthetic_cli_exists_and_real_training_options_rejected():
    for argv in (("--mode", "train"), ("--mode", "deploy"), ("--mode", "test"),
                 ("--mode", "synthetic-check", "--contract", "unused")):
        with pytest.raises(SystemExit):
            worker.main(argv)
    assert "subprocess" not in inspect.getsource(worker)


def test_cli_emits_one_json_value_and_keeps_engine_progress_on_stderr(capsys, tmp_path):
    output = tmp_path / "new_checks"
    expected = {"schema": worker.SCHEMA, "status": "passed", "real_data_loaded": False}
    def check(*, output):
        assert output == tmp_path / "new_checks"
        print("[epoch 001] fabricated engine progress")
        print("[epoch 002] fabricated engine progress")
        return expected
    with patch.object(worker, "synthetic_check", side_effect=check):
        worker.main(("--mode", "synthetic-check", "--output", str(output)))
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert captured.out.count('"status"') == 1
    assert "[epoch" not in captured.out
    assert captured.err.splitlines() == ["[epoch 001] fabricated engine progress",
                                         "[epoch 002] fabricated engine progress"]


def test_quantity_loss_uses_same_tensor_and_excludes_time_graph_exactly():
    z = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    time_parameter = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
    truth = torch.tensor([1.0, 10.0], dtype=torch.float64)
    quantity_loss = (torch.nn.functional.softplus(z) - torch.log1p(truth)).square()
    time_loss = time_parameter.square()
    original = {"quantity_train_loss": quantity_loss, "log_qty_loss": quantity_loss,
                "joint_loss": time_loss + quantity_loss, "time_loss": time_loss}
    value = worker.quantity_only_outputs(lambda *args, **kwargs: original)(None, lambda_log_qty=1.0)
    assert value["joint_loss"] is quantity_loss
    time_gradient, quantity_gradient = torch.autograd.grad(value["joint_loss"].mean(),
        (time_parameter, z), allow_unused=True)
    assert time_gradient is None
    expected = ((torch.nn.functional.softplus(z.detach()) - torch.log1p(truth))
                * torch.sigmoid(z.detach()))
    torch.testing.assert_close(quantity_gradient, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        worker.quantity_only_outputs(lambda **kwargs: original)(lambda_log_qty=0.1)
    changed = {**original, "quantity_train_loss": quantity_loss + 1}
    with pytest.raises(ValueError):
        worker.quantity_only_outputs(lambda **kwargs: changed)(lambda_log_qty=1.0)


@pytest.mark.parametrize("arm", worker.ARMS)
def test_actual_candidate_quantity_head_matches_literal_log_mse_without_time_gradient(arm):
    from paper.scripts.count_aware_tpp_backbone import core
    model, _ = worker.build_model(worker.fabricated_data(), arm)
    model.eval()
    dt, quantities, mask = worker.fabricated_batch()
    padded_dt, padded_q, padded_mask, lengths = core.right_pad_batch(dt, quantities, mask)
    rows = torch.arange(padded_dt.size(0))
    target_position = lengths - 1
    observed_q, write_mask = padded_q.clone(), padded_mask.clone()
    observed_q[rows, target_position] = 0.0
    write_mask[rows, target_position] = False
    _, hidden = model.encode_task_states(padded_dt, observed_q, padded_mask, memory_write_mask=write_mask)
    log_location = torch.nn.functional.softplus(model.quantity_head(hidden[rows, lengths - 2]).squeeze(-1))
    expected_loss = (log_location - torch.log1p(padded_q[rows, target_position])).square()
    actual = worker.quantity_only_outputs(core.target_outputs)(
        model, dt, mask, quantities, lambda_log_qty=1.0)
    assert torch.equal(actual["joint_loss"], expected_loss)
    assert torch.equal(actual["pred_qty"], torch.expm1(log_location))
    actual["joint_loss"].mean().backward()
    assert all(parameter.grad is None for _, parameter in model.time_head_named_parameters())


def test_hooks_restore_shared_aliases_after_exception_and_keep_frozen_sources():
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import core, training, constants, observed_time
    names = ((factory, "build_count_aware_model"), (factory, "validate_checkpoint_route"),
             (training, "build_count_aware_model"), (training, "build_model"),
             (training, "validate_checkpoint_route"), (training, "build_optimizer"),
             (training, "atomic_torch_save"), (training, "save_json"),
             (training, "target_outputs"), (core, "target_outputs"),
             (observed_time, "validate_launch"), (observed_time, "ROLES"),
             (training, "OBSERVED_TIME_ROLES"), (constants, "validate_model_role_contract"),
             (constants, "BACKBONE_LABELS"), (training, "BACKBONE_LABELS"),
             (constants, "MODEL_ROLES"), (constants, "SUPPORTED_BACKBONES"))
    before = [getattr(module, name) for module, name in names]
    frozen_sha = worker.sha_file(worker._ZERO_PATH)
    with pytest.raises(RuntimeError):
        with worker._engine_hooks(worker.ARMS[0]):
            assert factory.build_count_aware_model is not before[0]
            assert worker._ACTIVE_ARM == worker.ARMS[0]
            with pytest.raises(ValueError):
                with worker.candidate_factory(worker.ARMS[1]):
                    pass
            raise RuntimeError("fabricated restoration check")
    assert all(getattr(module, name) is value for (module, name), value in zip(names, before))
    assert worker._SCOPE_DEPTH == 0 and worker._ACTIVE_ARM is None
    assert frozen_sha == worker.sha_file(worker._ZERO_PATH)


def test_selected_candidate_cannot_be_relabelled_as_old_or_other_model(completed_synthetic):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    result = completed_synthetic["per_arm"][worker.ARMS[0]]
    payload = torch_load_checkpoint(Path(result["checkpoint"]), map_location="cpu")
    for wrong_arm in (worker.ARMS[1], "titantpp_history_mlp_width16", "titantpp"):
        with pytest.raises(ValueError):
            worker.validate_checkpoint(payload, wrong_arm)
    changed = deepcopy(payload)
    changed["backbone"] = worker.ARMS[1]
    with pytest.raises(ValueError):
        worker.validate_checkpoint(changed, worker.ARMS[1])


def test_first_strict_finite_raw_rmse_minimum_and_ties():
    history = [{"epoch": 1, "val_qty_rmse": float("nan")},
               {"epoch": 2, "val_qty_rmse": 3.0}, {"epoch": 3, "val_qty_rmse": 2.0},
               {"epoch": 4, "val_qty_rmse": 2.0}, {"epoch": 5, "val_qty_rmse": float("inf")}]
    assert worker.select_best_epoch(history) == 3
    with pytest.raises(ValueError):
        worker.select_best_epoch(history[:1])


def test_synthetic_output_cannot_overwrite_existing_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        worker.synthetic_check(output=output)
    assert sentinel.read_text() == "preserve"


def test_independent_six_metric_arithmetic(completed_synthetic):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    result = completed_synthetic["per_arm"][worker.ARMS[0]]
    payload = torch_load_checkpoint(Path(result["checkpoint"]), map_location="cpu")
    # Build a fresh instance under the saved model's Train-statistic values.
    data = worker.fabricated_data()
    head = payload["encoder_config"]["time_head"]
    data["loader"]["max_seq_len"] = payload["encoder_config"]["max_len"]
    data["model"].update(time_scale=head["time_scale"],
        time_initial_location=head["time_initial_location"], time_initial_scale=head["time_initial_scale"],
        time_observation_contract=head["observation_likelihood"])
    model, _ = worker.build_model(data, worker.ARMS[0])
    model.load_state_dict(payload["model_state_dict"])
    value = worker.evaluate_fabricated(model)
    prediction, truth, time = (torch.tensor(value[key], dtype=torch.float64)
                               for key in ("prediction", "truth", "time_nll"))
    error = prediction - truth
    tail = truth > 5
    expected = [float(error.square().mean().sqrt()), float(error.abs().mean()),
                float(error[tail].square().mean().sqrt()), float(error[tail].abs().mean()),
                float(time.mean()), float(time[tail].mean())]
    assert list(value["metrics"].values()) == expected
