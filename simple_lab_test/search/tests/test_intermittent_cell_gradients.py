"""CPU-only synthetic checks; no project models, checkpoints, or datasets."""
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[3] / "paper/scripts/intermittent_cell_gradients.py"
SPEC = importlib.util.spec_from_file_location("cell_gradient_test_engine", SOURCE)
engine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = engine
SPEC.loader.exec_module(engine)


class SyntheticModel(torch.nn.Module):
    def __init__(self, dropout=0.0, dtype=torch.float64):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 3, dtype=dtype)
        self.drop = torch.nn.Dropout(dropout)
        self.quantity_head = torch.nn.Linear(3, 1, dtype=dtype)
        self.v_t = torch.nn.Linear(3, 1, dtype=dtype)
        self.b_t = torch.nn.Parameter(torch.tensor(0.4, dtype=dtype))
        self.w_raw = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))
        self.unused = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
        self.register_buffer("persistent_marker", torch.tensor(7.0, dtype=dtype))
        self.contextual_mem_size = 0
        self.use_context_update = False
        self._ctx_mem = None

    def outputs(self, x, truth, alpha=0.5, scale=0.8):
        hidden = self.drop(self.encoder(x))
        location = torch.nn.functional.softplus(self.quantity_head(hidden).squeeze(-1))
        prediction = location.expm1()
        time = (self.v_t(hidden).squeeze(-1) + self.b_t + self.w_raw).square()
        log = (location - truth.log1p()).square()
        raw = ((prediction - truth) / 3.0).square()
        quantity = scale * (log + alpha * raw)
        return {"time_loss": time, "log_qty_loss": log, "raw_qty_loss": raw,
                "objective_loss": time + quantity, "quantity_train_loss": quantity,
                "true_qty": truth, "pred_qty": prediction}


def inputs(n=5):
    x = torch.linspace(-0.4, 1.2, n * 2, dtype=torch.float64).reshape(n, 2)
    truth = torch.linspace(0, 4, n, dtype=torch.float64)
    return x, truth


def probe(model=None, n=5, cells=None, alpha=0.5, scale=0.8):
    model = model or SyntheticModel()
    x, truth = inputs(n)
    return engine.probe_batch(model, lambda: model.outputs(x, truth, alpha, scale),
                              alpha=alpha, quantity_scale=scale,
                              cell_indices=torch.arange(n) % 3 if cells is None else cells)


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_one_forward_uneven_empty_cells_and_independent_joint_audit():
    torch.manual_seed(7)
    model = SyntheticModel()
    x, truth = inputs()
    calls = []
    def forward():
        calls.append(1)
        return model.outputs(x, truth)
    result = engine.probe_batch(model, forward, alpha=0.5, quantity_scale=0.8,
                               cell_indices=lambda outputs: torch.tensor([0, 0, 1, 2, 2]))
    assert len(calls) == 1
    assert result.summary["autograd_calls"] == 13
    assert result.summary["conservation"]["passed"]
    assert result.summary["immutability"]["passed"]
    assert [cell["count"] for cell in result.summary["cells"][:4]] == [2, 1, 2, 0]
    assert torch.allclose(result._vectors[1:].sum(0), result._vectors[0], atol=1e-12)
    assert result._vectors.device.type == "cpu" and not result._vectors.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
    empty = result.summary["cells"][3]
    assert empty["loss_sums"]["joint"] == 0
    assert empty["loss_means"]["joint"] is None
    assert empty["preclip"]["all"]["log_raw"]["cosine"] is None
    assert empty["preclip"]["all"]["joint_projection"] is None
    assert sum(c["preclip"]["all"]["joint_projection"] or 0 for c in result.summary["cells"]) == pytest.approx(1)
    json.dumps(result.summary, allow_nan=False)


def test_maximum_49_autograd_calls_and_unused_zero():
    result = probe(n=15, cells=torch.arange(15))
    assert result.summary["autograd_calls"] == 49
    unused = next(item for item in result._layout if item[0] == "unused")
    assert torch.count_nonzero(result._vectors[..., unused[2]:unused[3]]) == 0


def test_B_measures_raw_but_uses_zero_weighted_raw_and_correct_groups():
    result = probe(alpha=0, scale=1)
    full = result.summary["overall"]["preclip"]
    assert full["all"]["norms"]["raw_scaled"] > 0
    assert full["all"]["norms"]["weighted_raw"] == 0
    assert full["all"]["weighted_raw_joint_projection"] == 0
    assert full["time_head"]["norms"]["quantity"] == 0
    assert full["quantity_head"]["norms"]["time"] == 0
    assert full["time_head"]["time_quantity"]["cosine"] is None


def test_common_clipping_every_cell_and_task():
    result = probe()
    k = result.summary["common_clip_factor"]
    assert k < 1
    for row in [result.summary["overall"], *result.summary["cells"]]:
        for group in engine.GROUPS:
            for task, norm in row["preclip"][group]["norms"].items():
                assert row["postclip"][group]["norms"][task] == pytest.approx(k * norm)


def test_dropout_rng_mode_requires_grad_and_existing_gradients_preserved():
    model = SyntheticModel(dropout=0.4)
    model.v_t.eval()
    first = next(model.parameters())
    first.requires_grad_(False)
    first.grad = torch.full_like(first, 9)
    original_grad = first.grad
    before = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    rng = rng_snapshot()
    left = probe(model)
    assert_rng_equal(rng, rng_snapshot())
    right = probe(model)
    assert torch.equal(left._vectors, right._vectors)
    assert first.grad is original_grad and torch.equal(first.grad, torch.full_like(first, 9))
    assert first.requires_grad is False
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("attribute,value", [("_ctx_mem", torch.ones(1)), ("contextual_mem_size", 1), ("use_context_update", True)])
def test_reject_nonstatic_memory_before_forward(attribute, value):
    model = SyntheticModel()
    setattr(model, attribute, value)
    with pytest.raises(ValueError, match="Static probe"):
        probe(model)


@pytest.mark.parametrize("mutation", ["parameter", "buffer", "context", "grad"])
def test_detect_and_restore_mutation(mutation):
    model = SyntheticModel()
    parameter = next(model.parameters())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = rng_snapshot()
    with pytest.raises(ValueError, match="mutated model state"):
        with engine.preserved_model_state(model):
            random.random()
            np.random.random()
            torch.rand(4)
            model.eval()
            parameter.requires_grad_(False)
            if mutation == "parameter":
                with torch.no_grad():
                    parameter.add_(1)
            elif mutation == "buffer":
                model.persistent_marker.add_(1)
            elif mutation == "context":
                model._ctx_mem = torch.ones(1)
            else:
                parameter.grad = torch.ones_like(parameter)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert model._ctx_mem is None
    assert parameter.grad is None and parameter.requires_grad
    assert model.training
    assert_rng_equal(rng, rng_snapshot())


def test_exception_restores_rng_flags_and_outer_seed_reset():
    model = SyntheticModel()
    rng = rng_snapshot()
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with engine.preserved_model_state(model):
            random.seed(9042)
            np.random.seed(9042)
            torch.manual_seed(9042)
            model.eval()
            next(model.parameters()).requires_grad_(False)
            raise RuntimeError("synthetic failure")
    assert model.training and next(model.parameters()).requires_grad
    assert_rng_equal(rng, rng_snapshot())


@pytest.mark.parametrize("bad", ["nan_output", "nan_gradient", "objective", "indices", "mode"])
def test_reject_nonfinite_and_bad_contract(bad):
    model = SyntheticModel()
    x, truth = inputs()
    def forward():
        out = model.outputs(x, truth)
        if bad == "nan_output":
            out["pred_qty"] = out["pred_qty"] * float("nan")
        elif bad == "nan_gradient":
            next(model.parameters()).register_hook(lambda gradient: gradient * float("nan"))
        elif bad == "objective":
            out["objective_loss"] = out["objective_loss"] * 2
        return out
    if bad == "mode":
        model.eval()
    cells = torch.zeros(5, dtype=torch.long) if bad != "indices" else torch.full((5,), 15, dtype=torch.long)
    with pytest.raises(ValueError):
        engine.probe_batch(model, forward, alpha=0.5, quantity_scale=0.8, cell_indices=cells)


def test_budget_interrupt_preserves_model_and_rng():
    model = SyntheticModel(dropout=0.4)
    rng = rng_snapshot()
    x, truth = inputs()
    calls = []
    def budget():
        calls.append(1)
        if len(calls) == 4:
            raise TimeoutError("fixed budget")
    with pytest.raises(TimeoutError, match="fixed budget"):
        engine.probe_batch(model, lambda: model.outputs(x, truth), alpha=0.5, quantity_scale=0.8,
                           cell_indices=torch.zeros(5, dtype=torch.long), budget_callback=budget)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert_rng_equal(rng, rng_snapshot())


class LinearLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def outputs(self, slopes):
        time = slopes * self.weight
        zero = time * 0
        return {"time_loss": time, "log_qty_loss": zero, "raw_qty_loss": zero,
                "objective_loss": time}


def test_clip_epsilon_boundary_is_distinct_from_norm_threshold():
    model = LinearLossModel()
    result = engine.probe_batch(model, lambda: model.outputs(torch.ones(2, dtype=torch.float64)),
        alpha=0., quantity_scale=1., cell_indices=torch.zeros(2, dtype=torch.int64))
    assert result.summary['global_joint_preclip_norm'] == 1.
    assert result.summary['common_clip_factor'] == pytest.approx(1. / (1. + 1e-6))
    assert result.summary['clipped'] is True
    assert result.summary['norm_above_max'] is False


def linear_probe(model, slopes, cells):
    return engine.probe_batch(model, lambda: model.outputs(torch.tensor(slopes, dtype=torch.float64)),
                              alpha=0, quantity_scale=1, cell_indices=torch.tensor(cells), cell_count=2)


def test_cancellation_zero_reference_and_nonadditive_norms():
    result = linear_probe(LinearLossModel(), [3, -3], [0, 1])
    assert result.summary["global_joint_preclip_norm"] == 0
    assert result.summary["overall"]["preclip"]["all"]["joint_projection"] is None
    assert all(c["preclip"]["all"]["joint_projection"] is None for c in result.summary["cells"])
    assert sum(c["preclip"]["all"]["norms"]["joint"] for c in result.summary["cells"]) == 3


def test_streaming_unequal_batches_energy_and_postclip_mean():
    model = LinearLossModel()
    left = linear_probe(model, [4, 4], [0, 1])
    right = linear_probe(model, [-2], [0])
    accumulator = engine.GradientAccumulator()
    accumulator.add(left, count=2)
    accumulator.add(right, count=1)
    result = accumulator.finalize()
    overall = result["overall"]
    # E[g]=(2*4-2)/3=2; E[g^2]=(2*16+4)/3=12.
    energy = overall["gradient_energy"]["preclip"]["all"]
    assert energy["mean_batch_squared_norm"]["joint"] == pytest.approx(12)
    assert energy["squared_norm_of_mean_gradient"]["joint"] == pytest.approx(4)
    expected_post = (2 * left.summary["common_clip_factor"] * 4 - right.summary["common_clip_factor"] * 2) / 3
    assert overall["postclip"]["all"]["norms"]["joint"] == pytest.approx(abs(expected_post))
    assert overall["postclip"]["all"]["norms"]["joint"] != pytest.approx(1.0)
    assert result["count"] == 3 and result["cells"][0]["count"] == 2
    # Projection of mean is not the mean of batch projections.
    assert result["cells"][0]["preclip"]["all"]["joint_projection"] == pytest.approx(1 / 3)
    assert result["conservation"]["passed"]
    json.dumps(result, allow_nan=False)


def test_accumulator_rejects_count_and_parameter_identity_drift():
    accumulator = engine.GradientAccumulator()
    result = probe()
    with pytest.raises(ValueError, match="count mismatch"):
        accumulator.add(result, count=99)
    accumulator.add(result)
    with pytest.raises(ValueError, match="identity drift"):
        accumulator.add(probe(alpha=0.6))
    with pytest.raises(ValueError, match="empty"):
        engine.GradientAccumulator().finalize()


def test_added_unregistered_context_attribute_detected_and_removed():
    model = LinearLossModel()
    with pytest.raises(ValueError, match="mutated model state"):
        with engine.preserved_model_state(model):
            model._ctx_mem = torch.ones(1)
    assert not hasattr(model, "_ctx_mem")


def test_joint_gradient_audit_catches_value_preserving_wrong_gradient():
    model = SyntheticModel()
    x, truth = inputs()
    def forward():
        outputs = model.outputs(x, truth)
        actual = outputs["objective_loss"]
        outputs["objective_loss"] = 2 * actual - actual.detach()
        return outputs
    with pytest.raises(ValueError, match="Gradient reconstruction failed: actual objective"):
        engine.probe_batch(model, forward, alpha=0.5, quantity_scale=0.8,
                           cell_indices=torch.zeros(5, dtype=torch.long))


def test_dropout_child_eval_mode_rejected():
    model = SyntheticModel(dropout=0.4)
    model.drop.eval()
    with pytest.raises(ValueError, match="dropout modules"):
        probe(model)


def test_accumulator_rejects_different_checkpoint_weights():
    model = SyntheticModel()
    accumulator = engine.GradientAccumulator()
    accumulator.add(probe(model))
    with torch.no_grad():
        model.encoder.weight.add_(0.1)
    with pytest.raises(ValueError, match="identity drift"):
        accumulator.add(probe(model))


def test_float32_full_batch_dropout_reconstruction():
    torch.manual_seed(9042)
    model = SyntheticModel(dropout=0.1, dtype=torch.float32)
    x, truth = (value.float() for value in inputs(128))
    result = engine.probe_batch(model, lambda: model.outputs(x, truth, 1.769152228020146, 0.8090744629002368),
                               alpha=1.769152228020146, quantity_scale=0.8090744629002368,
                               cell_indices=torch.arange(128) % 15)
    assert result.summary["autograd_calls"] == 49
    assert result.summary["conservation"]["passed"]
    accumulator = engine.GradientAccumulator()
    accumulator.add(result)
    assert accumulator.finalize()["conservation"]["passed"]
