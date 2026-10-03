import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.quantity_comparison_engine import _validate_device, run_case
from paper.scripts.quantity_objective_comparison import QuantityCase, fit_train_quantity_statistics
from paper.scripts.time_quantity_diagnostic import run_arm, update_selectors
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def stats():
    return fit_train_quantity_statistics(torch.tensor([0., 1., 2., 4., 3., 2.]), torch.tensor([4., 3., 5., 2.]), split="train")


def make_model(seed=31):
    torch.manual_seed(seed)
    model, _ = build_count_aware_model("titantpp", hidden_dim=8, train_log_mean=stats().mu, max_seq_len=8, time_intercept_limit=300.0)
    return model


def batch():
    return (
        torch.tensor([[0., 0., 1., 2.], [0., 1., 1., 3.], [0., 0., 2., 2.], [0., 1., 2., 4.]]),
        torch.tensor([[False, False, True, True], [False, True, True, True], [False, False, True, True], [False, True, True, True]]),
        torch.tensor([[0., 0., 2., 4.], [0., 1., 2., 3.], [0., 0., 1., 5.], [0., 2., 1., 2.]]),
    )


def loaders():
    dataset = TensorDataset(*batch())
    return (
        DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(41)),
        DataLoader(dataset, batch_size=2, shuffle=False, generator=torch.Generator().manual_seed(42)),
    )


def identity():
    return {"source": {"revision": "synthetic"}, "data": {"sha256": "synthetic"}, "runtime": {"name": "synthetic-cpu"}, "execution_contract_sha256": "a" * 64}


def run(path, case, **kwargs):
    train, validation = loaders()
    return run_case(model=make_model(), train_loader=train, validation_loader=validation, case=case, statistics=stats(), output_dir=path, epochs=3, seed=42, identity=identity(), **kwargs)


@pytest.mark.parametrize("case", list(QuantityCase))
def test_all_cases_share_initialization_batch_order_and_exact_epoch_resume(tmp_path, case):
    full = run(tmp_path / f"{case.value}-full", case)
    partial = run(tmp_path / f"{case.value}-resume", case, stop_after_epochs=1)
    resumed = run(tmp_path / f"{case.value}-resume", case, resume=True)
    assert partial["status"] == "paused_at_epoch_boundary"
    assert full["history"] == resumed["history"]
    assert full["last_state_sha256"] == resumed["last_state_sha256"]
    assert full["selectors"] == resumed["selectors"]
    assert full["global_step"] == 6
    assert all(row["train_count"] == row["validation_count"] == 4 for row in full["history"])
    assert all(row["train_batch_order_sha256"] for row in full["history"])
    uninterrupted = torch.load(tmp_path / f"{case.value}-full" / "last_epoch_state.pt", weights_only=False)
    restarted = torch.load(tmp_path / f"{case.value}-resume" / "last_epoch_state.pt", weights_only=False)
    assert uninterrupted["optimizer_state_sha256"] == restarted["optimizer_state_sha256"]
    assert uninterrupted["rng_state_sha256"] == restarted["rng_state_sha256"]
    timing = json.loads((tmp_path / f"{case.value}-resume" / "timing.json").read_text())
    assert [row["epoch"] for row in timing["epochs"]] == [1, 2, 3]
    diagnostic = full["history"][0]["gradient_diagnostic"]
    assert diagnostic["actual_clip_factor"] > 0 and diagnostic["global_joint_preclip_norm"] >= 0


def test_cases_have_same_initial_state_and_loader_order(tmp_path):
    summaries = [run(tmp_path / case.value, case) for case in QuantityCase]
    assert len({summary["initial_state_sha256"] for summary in summaries}) == 1
    for epoch in range(3):
        assert len({summary["history"][epoch]["train_batch_order_sha256"] for summary in summaries}) == 1


def test_raw_and_log_cases_record_log_mse_separately_and_publish_condition(tmp_path):
    raw = run(tmp_path / "raw", QuantityCase.RAW_ORIGINAL, stop_after_epochs=1)
    log = run(tmp_path / "log", QuantityCase.LOG_SOFTPLUS, stop_after_epochs=1)
    assert raw["condition"]["case"] == QuantityCase.RAW_ORIGINAL.value
    assert raw["condition"]["raw_scale"] == stats().raw_scale
    assert raw["history"][0]["quantity_train_loss"] != raw["history"][0]["log_quantity_mse"]
    assert log["history"][0]["quantity_train_loss"] == pytest.approx(log["history"][0]["log_quantity_mse"])
    checkpoint = torch.load(tmp_path / "raw" / "best_raw_quantity_rmse_model.pt", weights_only=False)
    assert checkpoint["condition"] == raw["condition"]


def test_resume_rejects_identity_and_tampered_checkpoint(tmp_path):
    path = tmp_path / "resume"
    run(path, QuantityCase.RAW_ORIGINAL, stop_after_epochs=1)
    changed = identity()
    changed["data"] = {"sha256": "other"}
    train, validation = loaders()
    with pytest.raises(ValueError, match="identity"):
        run_case(model=make_model(), train_loader=train, validation_loader=validation, case=QuantityCase.RAW_ORIGINAL, statistics=stats(), output_dir=path, epochs=3, seed=42, identity=changed, resume=True)
    checkpoint = path / "last_epoch_state.pt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["model_state_dict"][next(iter(payload["model_state_dict"]))].add_(1)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="model state hash"):
        run(path, QuantityCase.RAW_ORIGINAL, resume=True)


def test_selectors_keep_earliest_tie_and_saved_selected_prediction_restores(tmp_path):
    selectors = {"raw_quantity_rmse": {"applicable": True, "best_value": None}}
    state = {"x": torch.ones(1)}
    for epoch, value in ((1, 2.0), (2, 1.0), (3, 1.0)):
        update_selectors(selectors, {"raw_quantity_rmse": value}, epoch=epoch, global_step=epoch, state=state)
    assert selectors["raw_quantity_rmse"]["best_epoch"] == 2
    run(tmp_path, QuantityCase.LOG_SOFTPLUS)
    selected = torch.load(tmp_path / "best_raw_quantity_rmse_model.pt", weights_only=False)
    restored = make_model()
    restored.load_state_dict(selected["model_state_dict"], strict=True)
    restored.eval()
    dts, mask, quantities = batch()
    with torch.no_grad():
        expected = restored.quantity_outputs(restored.encode_task_states(dts, quantities, mask)[1][:, -2], quantities[:, -1])["point_prediction"]
    repeated = make_model()
    repeated.load_state_dict(selected["model_state_dict"], strict=True)
    repeated.eval()
    with torch.no_grad():
        actual = repeated.quantity_outputs(repeated.encode_task_states(dts, quantities, mask)[1][:, -2], quantities[:, -1])["point_prediction"]
    assert torch.equal(expected, actual)


def test_b_case_matches_original_engine_one_epoch_step(tmp_path):
    train, validation = loaders()
    original = run_arm(model=make_model(), train_loader=train, validation_loader=validation, objective="joint", output_dir=tmp_path / "original", epochs=1, seed=42, identity={"source": "synthetic", "data": "synthetic"})
    candidate = run(tmp_path / "candidate", QuantityCase.B_LOG_ORIGINAL, stop_after_epochs=1)
    first = torch.load(tmp_path / "original" / "last_epoch_state.pt", weights_only=False)
    second = torch.load(tmp_path / "candidate" / "last_epoch_state.pt", weights_only=False)
    assert canonical_state_dict_sha256(first["model_state_dict"]) == canonical_state_dict_sha256(second["model_state_dict"])
    assert original["history"][0]["train_batch_order_sha256"] == candidate["history"][0]["train_batch_order_sha256"]


def test_budget_callback_and_cuda_guard_are_not_bypassed(tmp_path):
    stages = []
    run(tmp_path / "budget", QuantityCase.B_LOG_ORIGINAL, stop_after_epochs=1, budget_check=stages.append)
    assert stages.count("train_batch") == 2 and stages.count("validation_batch") == 2 and stages[-1] == "before_epoch_commit"
    train, validation = loaders()
    with pytest.raises(ValueError, match="explicit device"):
        run_case(model=make_model(), train_loader=train, validation_loader=validation, case=QuantityCase.B_LOG_ORIGINAL, statistics=stats(), output_dir=tmp_path / "cuda", epochs=1, seed=42, identity=identity(), device="cuda")


def synthetic_cuda_guard(model, train, validation, epochs=2):
    runtime = {"device": "cuda:0", "name": "synthetic-guard-only"}
    scope = {**identity(), "runtime": runtime, "purpose": "synthetic_qualification"}
    receipt = {"purpose": "synthetic_probe", "runtime": runtime,
               "execution_contract_sha256": scope["execution_contract_sha256"]}
    # Validate the actual CUDA branch on CPU tensors; do not mock model metadata
    # or transfer any tensor to CUDA. Bootstrap validation has no runtime calls.
    _validate_device(torch.device("cuda:0"), scope, receipt, model, train, validation, epochs)


def test_cuda_bootstrap_accepts_actual_launcher_model_and_tensor_shapes():
    from paper.scripts.run_quantity_comparison import synthetic_inputs

    model, train, validation = synthetic_inputs()
    assert model.hidden_dim == model.encoder.max_len == 8
    assert not hasattr(model, "max_seq_len")
    assert tuple(model.encoder.pos_emb.shape) == (1, 8, 8)
    assert all(tuple(tensor.shape) == (8, 8) for tensor in train.dataset.tensors)
    synthetic_cuda_guard(model, train, validation)


@pytest.mark.parametrize("change, message", [
    ("hidden16", "hidden_dim=max_seq_len"),
    ("configured_sequence9", "hidden_dim=max_seq_len"),
    ("sequence9", "aligned sequence tensors"),
    ("targets9", "budget exceeded"),
    ("research_loader", "TensorDataset loaders"),
    ("named_tensor_dataset", "TensorDataset loaders"),
    ("unaligned_tensors", "aligned sequence tensors"),
    ("epochs3", "budget exceeded"),
    ("epochs0", "budget exceeded"),
])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_cuda_bootstrap_rejects_scope_expansion_without_gpu(change, message, split):
    from paper.scripts.run_quantity_comparison import synthetic_inputs

    model, train, validation = synthetic_inputs()
    epochs = 2
    if change in {"hidden16", "configured_sequence9"}:
        model, _ = build_count_aware_model(
            "titantpp", hidden_dim=16 if change == "hidden16" else 8,
            train_log_mean=1., max_seq_len=9 if change == "configured_sequence9" else 8,
            time_intercept_limit=300.)
    elif change.startswith("epochs"):
        epochs = int(change.removeprefix("epochs"))
    else:
        tensors = list(train.dataset.tensors)
        if change == "sequence9":
            tensors = [torch.cat((tensor, tensor[:, :1]), dim=1) for tensor in tensors]
        elif change == "targets9":
            tensors = [torch.cat((tensor, tensor[:1]), dim=0) for tensor in tensors]
        elif change == "unaligned_tensors":
            tensors[1] = tensors[1][:, :-1]
        dataset = TensorDataset(*tensors)
        if change == "research_loader":
            dataset = torch.utils.data.Subset(dataset, list(range(8)))
        elif change == "named_tensor_dataset":
            # A class name is not sufficient evidence of the synthetic loader.
            dataset = type("TensorDataset", (TensorDataset,), {})(*tensors)
        replacement = DataLoader(dataset, batch_size=4)
        if split == "train":
            train = replacement
        else:
            validation = replacement
    with pytest.raises(ValueError, match=message):
        synthetic_cuda_guard(model, train, validation, epochs)
