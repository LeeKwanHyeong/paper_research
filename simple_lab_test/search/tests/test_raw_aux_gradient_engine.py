"""Small real-Titan CPU integration, never reads research data/checkpoints."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from paper.scripts import quantity_comparison_engine as engine
from paper.scripts.mixed_quantity_objective import MixedQuantityObjective
from paper.scripts.quantity_objective_comparison import QuantityCase, QuantityStatistics
from paper.scripts.raw_aux_gradient_control import RawAuxGradientControl
from paper.scripts.run_quantity_comparison import synthetic_inputs


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "search_artifacts/mixed_quantity_preparation_v1/snapshots/3ac65c42b35a329ede32ad95862c594579dfca9c3ae4ebf6458560cec1d0cbd8/source/paper/scripts/quantity_comparison_engine.py"


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def run(path, *, implementation=engine, capped=True, objective_name="mixed_original", **kwargs):
    model, train, validation = synthetic_inputs()
    objective = None if objective_name is None else MixedQuantityObjective(objective_name,
        0. if objective_name == "B_log_original" else 1.769152228020146, 1., "c" * 64)
    options = {"raw_aux_control": RawAuxGradientControl()} if capped else {}
    options.update(kwargs)
    return implementation.run_case(model=model, train_loader=train, validation_loader=validation,
        case=QuantityCase.B_LOG_ORIGINAL, mixed_objective=objective, statistics=QuantityStatistics(1., 5.),
        output_dir=path, epochs=2, seed=42, device="cpu",
        identity={"source": {"scope": "cpu-synthetic"}, "data": {"synthetic_only": True},
                  "runtime": {"device": "cpu"}, "execution_contract_sha256": "a" * 64}, **options)


def payload(path):
    # These checkpoints are created from the tiny synthetic inputs in this test.
    return torch.load(path / "last_epoch_state.pt", weights_only=False, map_location="cpu")


@pytest.mark.parametrize("objective_name", [None, "B_log_original", "mixed_original"])
def test_disabled_controller_matches_frozen_engine_states_history_and_selectors(tmp_path, objective_name):
    specification = importlib.util.spec_from_file_location("frozen_mixed_engine_for_test", FROZEN)
    original = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(original)
    old = run(tmp_path / "old", implementation=original, capped=False, objective_name=objective_name)
    new = run(tmp_path / "new", capped=False, objective_name=objective_name)
    assert old["history"] == new["history"]
    assert old["selectors"] == new["selectors"]
    first, second = payload(tmp_path / "old"), payload(tmp_path / "new")
    for key in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
        assert first[key] == second[key]
    before, after = dict(first["contract"]), dict(second["contract"])
    # Editing an engine necessarily changes its source identity, never its legacy schema.
    assert before.pop("engine_source_sha256") != after.pop("engine_source_sha256")
    assert before == after
    assert not list((tmp_path / "new").glob("raw_aux_batches*"))
    assert (tmp_path / "old" / "history.json").read_bytes() == (tmp_path / "new" / "history.json").read_bytes()


def test_capped_full_vs_epoch_resume_restores_all_states_and_batch_records(tmp_path):
    full = run(tmp_path / "full")
    partial = run(tmp_path / "resumed", stop_after_epochs=1)
    resumed = run(tmp_path / "resumed", resume=True)
    assert partial["status"] == "paused_at_epoch_boundary"
    assert full == resumed
    first, second = payload(tmp_path / "full"), payload(tmp_path / "resumed")
    for key in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
        assert first[key] == second[key]
    for epoch in (1, 2):
        name = f"raw_aux_batches_epoch_{epoch:04d}.json"
        assert (tmp_path / "full" / name).read_bytes() == (tmp_path / "resumed" / name).read_bytes()
    assert full["condition"]["raw_aux_control"] == RawAuxGradientControl().to_dict()
    assert full["condition"]["mixed_objective"]["name"] == "mixed_original"
    timing = json.loads((tmp_path / "full" / "timing.json").read_text())
    assert [r["raw_auxiliary_work"]["component_gradient_queries"] for r in timing["epochs"]] == [4, 4]
    assert [r["raw_auxiliary_work"]["sparse_diagnostic_gradient_queries"] for r in timing["epochs"]] == [10, 0]
    assert all(r["raw_auxiliary_work"]["peak_cuda_allocated_bytes"] is None for r in timing["epochs"])


def test_three_arms_have_same_exposure_and_capped_validation_is_reference_only(tmp_path, monkeypatch):
    import paper.scripts.raw_aux_gradient_control as controller
    original = controller.apply_raw_aux_control
    calls = []
    def checked(*args, **kwargs):
        assert kwargs["training"] and torch.is_grad_enabled()
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(controller, "apply_raw_aux_control", checked)
    summaries = [run(tmp_path / "B", capped=False, objective_name="B_log_original"),
                 run(tmp_path / "uncapped", capped=False), run(tmp_path / "capped")]
    assert len(calls) == 4
    assert len({summary["initial_state_sha256"] for summary in summaries}) == 1
    assert {summary["global_step"] for summary in summaries} == {4}
    for epoch in range(2):
        assert len({s["history"][epoch]["train_batch_order_sha256"] for s in summaries}) == 1
        row = summaries[-1]["history"][epoch]
        assert row["train_count"] == row["validation_count"] == 8
        assert "adaptive_objective_loss" not in row
        assert row["train"]["adaptive_objective_loss"] <= row["train"]["objective_loss"] + 1e-6
        assert row["objective_loss"] == pytest.approx(row["legacy_time_loss"] + row["log_quantity_mse"]
            + 1.769152228020146 * row["raw_quantity_scaled_mse"], rel=1e-6)
        diagnostic = row["raw_aux_control"]
        assert diagnostic["count"] == 8 and diagnostic["component_gradient_queries"] == 4
        records = json.loads((tmp_path / "capped" / diagnostic["batches_file"]).read_text())["batches"]
        assert [r["global_step"] for r in records] == [epoch * 2 + 1, epoch * 2 + 2]
        assert all(r["norm_bound_passed"] and r["component_gradient_queries"] == 2 for r in records)


@pytest.mark.parametrize("tamper", ["controller", "source", "batch_record"])
def test_capped_resume_rejects_controller_source_or_batch_record_tamper(tmp_path, tamper):
    run(tmp_path, stop_after_epochs=1)
    checkpoint = tmp_path / "last_epoch_state.pt"
    saved = payload(tmp_path)
    if tamper == "batch_record":
        path = tmp_path / "raw_aux_batches_epoch_0001.json"
        record = json.loads(path.read_text())
        record["batches"][0]["s"] = .12345
        path.write_text(json.dumps(record))
    else:
        saved["contract"]["condition"]["raw_aux_control"]["source_sha256" if tamper == "source" else "cap_ratio"] = "changed" if tamper == "source" else .5
        saved["contract_sha256"] = engine._sha_json(saved["contract"])
        torch.save(saved, checkpoint)
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="identity|record hash"):
        run(tmp_path, resume=True)
    assert checkpoint.read_bytes() == before


def test_resume_cannot_switch_from_uncapped_to_capped(tmp_path):
    run(tmp_path, capped=False, stop_after_epochs=1)
    before = (tmp_path / "last_epoch_state.pt").read_bytes()
    with pytest.raises(ValueError, match="identity"):
        run(tmp_path, resume=True)
    assert (tmp_path / "last_epoch_state.pt").read_bytes() == before


def test_capped_strict_earliest_selector_ties_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: None)
    summary = run(tmp_path)
    for selector in engine.SELECTORS:
        assert len({row[selector] for row in summary["history"]}) == 1
        assert summary["selectors"][selector]["best_epoch"] == 1
    assert run(tmp_path, resume=True) == summary


@pytest.mark.parametrize("options", [{"objective_name": "B_log_original"}, {"objective_name": None}, {"grad_clip": 2.}])
def test_invalid_capped_combination_rejected_before_output(tmp_path, options):
    with pytest.raises(ValueError, match="mixed_original|grad_clip"):
        run(tmp_path / "invalid", **options)
    assert not (tmp_path / "invalid").exists()


def test_nonfinite_combined_gradient_rejected_before_optimizer_step(tmp_path, monkeypatch):
    # Inject an infinite accumulated gradient after the real synthetic backward.
    # The production finite check must reject it before AdamW can run.
    original_inputs = synthetic_inputs
    models = []
    def inputs():
        values = original_inputs()
        models.append(values[0])
        return values
    monkeypatch.setitem(globals(), "synthetic_inputs", inputs)
    original_backward = torch.Tensor.backward
    def backward(*args, **kwargs):
        original_backward(*args, **kwargs)
        next(p.grad for p in models[0].parameters() if p.grad is not None).fill_(float("inf"))
    steps = []
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: steps.append(1))
    with pytest.raises(ValueError, match="Nonfinite.*combined gradient"):
        run(tmp_path)
    assert not steps and not (tmp_path / "last_epoch_state.pt").exists()
