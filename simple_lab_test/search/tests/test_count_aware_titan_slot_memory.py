"""CPU synthetic contracts for prefix-local observed-slot memory."""

from __future__ import annotations

import copy
import io
import math
from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanSlotMemory import (
    CountAwareTitanSlotMemoryTPP,
    ObservedSlotMemory,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def model_kwargs() -> dict[str, object]:
    return {
        "hidden_dim": 16,
        "train_log_mean": 1.5,
        "max_seq_len": 12,
        "quantity_variant": "count_only_log_regression",
        "lambda_tail": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp",
    }


def build_pair(seed: int = 47) -> tuple[CountAwareTitanTPP, CountAwareTitanSlotMemoryTPP]:
    torch.manual_seed(seed)
    control = CountAwareTitanTPP(**model_kwargs())
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    candidate = CountAwareTitanSlotMemoryTPP(**model_kwargs())
    assert torch.equal(torch.get_rng_state(), control_rng)
    return control, candidate


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor([[0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 3.0, 6.0, 0.0]])
    quantities = torch.tensor([[2.0, 5.0, 9.0, 12.0, 20.0], [1.0, 7.0, 13.0, 19.0, 0.0]])
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, True, False]])
    return dts, quantities, mask


def activate(candidate: CountAwareTitanSlotMemoryTPP) -> None:
    with torch.no_grad():
        candidate.slot_memory_alpha_raw.fill_(math.atanh(0.5))
        # The B quantity head starts at a constant prediction; use a trained-like
        # nonuniform readout to expose the memory path to quantity gradients.
        candidate.quantity_head.weight.copy_(torch.linspace(-0.3, 0.4, 16)[None])


def slow_prefix_reference(
    memory: ObservedSlotMemory,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    write_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute each prefix independently, without cumulative tensor scans."""
    rows = []
    width = hidden.shape[-1]
    for row in range(hidden.shape[0]):
        steps = []
        for end in range(hidden.shape[1]):
            observed = [
                index
                for index in range(end + 1)
                if bool(mask[row, index]) and bool(write_mask[row, index])
            ]
            if not bool(mask[row, end]) or len(observed) <= 1:
                steps.append(torch.zeros_like(hidden[row, end]))
                continue
            masses = hidden.new_zeros(8)
            totals = hidden.new_zeros(8, width)
            for index in observed:
                value = hidden[row, index]
                normalized = F.layer_norm(value, (width,), eps=1e-5)
                address = F.linear(normalized, memory.write_projection.weight)
                weights = torch.softmax(memory.addresses @ address / math.sqrt(8), dim=0)
                masses = masses + weights
                totals = totals + weights[:, None] * value[None]
            # Fixed numerical support threshold, independent of implementation
            # helpers and unrelated to any data-dependent model selection.
            supported = masses > 1e-8
            slots = totals[supported] / masses[supported, None]
            query = F.linear(
                F.layer_norm(hidden[row, end], (width,), eps=1e-5),
                memory.query_projection.weight,
            )
            keys = F.linear(
                F.layer_norm(slots, (width,), eps=1e-5),
                memory.key_projection.weight,
            )
            read_weights = torch.softmax(keys @ query / math.sqrt(8), dim=0)
            context = (read_weights[:, None] * slots).sum(dim=0)
            steps.append(context - hidden[row, end])
        rows.append(torch.stack(steps))
    return torch.stack(rows)


@pytest.mark.parametrize("seed", [13, 29])
def test_vectorized_memory_matches_independent_prefix_oracle_and_gradients(seed: int) -> None:
    torch.manual_seed(seed)
    memory = ObservedSlotMemory(16).double()
    reference_memory = copy.deepcopy(memory)
    hidden = torch.randn(3, 7, 16, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    mask = torch.tensor([
        [False, True, True, False, True, True, False],
        [True, False, True, True, True, False, True],
        [False, False, False, False, False, False, False],
    ])
    write_mask = torch.tensor([
        [True, True, False, True, True, False, True],
        [True, True, True, False, True, True, False],
        [True, True, True, True, True, True, True],
    ])
    actual = memory(hidden, mask, memory_write_mask=write_mask)
    expected = slow_prefix_reference(reference_memory, reference_hidden, mask, write_mask)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=2e-10, atol=2e-11)
    for name, parameter in memory.named_parameters():
        reference_parameter = dict(reference_memory.named_parameters())[name]
        assert parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=2e-9, atol=2e-11)


def test_uniform_assignments_use_count_normalized_raw_values() -> None:
    memory = ObservedSlotMemory(8).double()
    with torch.no_grad():
        memory.write_projection.weight.zero_()
        memory.query_projection.weight.zero_()
        memory.key_projection.weight.zero_()
    hidden = torch.arange(40, dtype=torch.float64).reshape(1, 5, 8) * 3.0 + 17.0
    mask = torch.ones(1, 5, dtype=torch.bool)
    write_mask = torch.tensor([[True, False, True, True, False]])

    actual = memory(hidden, mask, memory_write_mask=write_mask)

    assert torch.count_nonzero(actual[:, :2]) == 0
    for end, observed in ((2, [0, 2]), (3, [0, 2, 3]), (4, [0, 2, 3])):
        expected = hidden[:, observed].mean(dim=1) - hidden[:, end]
        torch.testing.assert_close(actual[:, end], expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("address_scale", [100.0, 1e6])
def test_tiny_positive_and_zero_mass_slots_have_finite_backward(address_scale: float) -> None:
    torch.manual_seed(23)
    memory = ObservedSlotMemory(8)
    with torch.no_grad():
        memory.write_projection.weight.zero_()
        memory.write_projection.weight[0, 0] = 1.0
        memory.addresses.zero_()
        memory.addresses[0, 0] = address_scale
    hidden = torch.randn(1, 4, 8)
    hidden[:, :, 0] += 8.0
    hidden.requires_grad_()
    mask = torch.ones(1, 4, dtype=torch.bool)

    correction = memory(hidden, mask)
    assert torch.isfinite(correction).all()
    # The extreme scores leave only slot zero numerically supported. Its value
    # must remain the raw observed-prefix mean after the other slots are removed.
    for end in range(1, hidden.shape[1]):
        expected = hidden[:, :end + 1].mean(dim=1) - hidden[:, end]
        torch.testing.assert_close(correction[:, end], expected, rtol=1e-6, atol=1e-6)
    correction.square().sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    for name, parameter in memory.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("observed_count", [0, 1])
def test_zero_or_one_observation_has_zero_finite_correction_and_gradient(observed_count: int) -> None:
    torch.manual_seed(73)
    memory = ObservedSlotMemory(16)
    hidden = torch.randn(2, 6, 16, requires_grad=True)
    mask = torch.tensor([[False, True, True, False, True, True], [True, True, True, True, True, True]])
    write_mask = torch.zeros_like(mask)
    if observed_count:
        write_mask[:, 1] = True

    correction = memory(hidden, mask, memory_write_mask=write_mask)
    assert torch.isfinite(correction).all()
    assert torch.count_nonzero(correction) == 0
    correction.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad) == 0
    for parameter in memory.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("sentinel", [-99999.0, 1e30, float("nan"), float("inf")])
def test_padding_values_never_write_or_contaminate_memory(sentinel: float) -> None:
    torch.manual_seed(127)
    memory = ObservedSlotMemory(16)
    hidden = torch.randn(2, 6, 16)
    mask = torch.tensor([[False, True, True, False, True, False], [True, False, True, True, False, True]])
    changed = hidden.clone()
    changed[~mask] = sentinel
    with torch.no_grad():
        expected = memory(hidden, mask)
        actual = memory(changed, mask)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[~mask]) == 0


def test_nonwriting_tokens_and_future_tokens_do_not_change_later_memory_reads() -> None:
    torch.manual_seed(149)
    memory = ObservedSlotMemory(16).double()
    hidden = torch.randn(1, 7, 16, dtype=torch.float64)
    mask = torch.ones(1, 7, dtype=torch.bool)
    write_mask = torch.tensor([[True, True, False, True, False, True, False]])
    changed = hidden.clone()
    changed[:, 2] = 7e5
    changed[:, 5:] = -9e5
    with torch.no_grad():
        expected = memory(hidden, mask, memory_write_mask=write_mask)
        actual = memory(changed, mask, memory_write_mask=write_mask)
    assert torch.equal(actual[:, :2], expected[:, :2])
    assert torch.equal(actual[:, 3:5], expected[:, 3:5])


def test_earlier_observed_values_change_read_with_identical_final_query() -> None:
    torch.manual_seed(157)
    memory = ObservedSlotMemory(16).double()
    hidden = torch.randn(1, 5, 16, dtype=torch.float64)
    mask = torch.ones(1, 5, dtype=torch.bool)
    changed = hidden.clone()
    changed[:, 0] += torch.linspace(-2.0, 4.0, 16)
    with torch.no_grad():
        expected = memory(hidden, mask)
        actual = memory(changed, mask)
    assert not torch.allclose(actual[:, -1], expected[:, -1], rtol=1e-6, atol=1e-7)


def test_memory_batch_isolation_permutation_and_no_persistent_mutation() -> None:
    torch.manual_seed(163)
    memory = ObservedSlotMemory(16).double()
    hidden = torch.randn(3, 6, 16, dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False, True, True], [False, True, True, True, True, False], [True, True, True, True, True, True]])
    write_mask = mask.clone()
    write_mask[:, -1] = False
    snapshot = {name: value.clone() for name, value in memory.state_dict().items()}
    with torch.no_grad():
        together = memory(hidden, mask, memory_write_mask=write_mask)
        separately = torch.cat([
            memory(hidden[i:i+1], mask[i:i+1], memory_write_mask=write_mask[i:i+1])
            for i in range(3)
        ])
        order = torch.tensor([2, 0, 1])
        permuted = memory(hidden[order], mask[order], memory_write_mask=write_mask[order])
        memory(hidden * 1e3, mask, memory_write_mask=write_mask)
        repeated = memory(hidden, mask, memory_write_mask=write_mask)
    torch.testing.assert_close(together, separately, rtol=1e-13, atol=1e-13)
    assert torch.equal(permuted, together[order])
    assert torch.equal(repeated, together)
    for name, value in memory.state_dict().items():
        assert torch.equal(value, snapshot[name]), name


def test_zero_gate_preserves_all_b_weights_and_caller_rng() -> None:
    control, candidate = build_pair()
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    expected_extra = {
        "slot_memory_alpha_raw",
        "slot_memory.addresses",
        "slot_memory.write_projection.weight",
        "slot_memory.query_projection.weight",
        "slot_memory.key_projection.weight",
    }
    assert set(candidate_state) - set(control_state) == expected_extra
    assert candidate.slot_memory_alpha.item() == 0.0
    for name, value in control_state.items():
        assert torch.equal(value, candidate_state[name]), name
    added = sum(parameter.numel() for parameter in candidate.parameters()) - sum(
        parameter.numel() for parameter in control.parameters()
    )
    assert added == 3 * 16 * 8 + 8 * 8 + 1


def test_zero_gate_preserves_train_outputs_losses_rng_and_shared_gradients_exactly() -> None:
    control, candidate = build_pair()
    control.train()
    candidate.train()
    dts, quantities, mask = batch()
    torch.manual_seed(211)
    expected = target_outputs(control, dts, mask, quantities, lambda_log_qty=1.0)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(211)
    actual = target_outputs(candidate, dts, mask, quantities, lambda_log_qty=1.0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
    expected["joint_loss"].mean().backward()
    actual["joint_loss"].mean().backward()
    candidate_parameters = dict(candidate.named_parameters())
    for name, parameter in control.named_parameters():
        candidate_gradient = candidate_parameters[name].grad
        if parameter.grad is None:
            assert candidate_gradient is None, name
        else:
            assert torch.equal(candidate_gradient, parameter.grad), name
    gate_gradient = candidate.slot_memory_alpha_raw.grad
    assert gate_gradient is not None
    assert torch.isfinite(gate_gradient).all()
    assert torch.count_nonzero(gate_gradient) > 0
    for name, parameter in candidate.slot_memory.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) == 0, name


def test_open_gate_changes_predictions_retains_final_hard_lmm_and_shared_heads() -> None:
    _, candidate = build_pair(seed=227)
    candidate.eval()
    activate(candidate)
    dts, quantities, mask = batch()
    final_read_calls = []

    def capture_final_read(_module: torch.nn.Module, _args: tuple, output: torch.Tensor) -> None:
        final_read_calls.append(output.detach().clone())

    handle = candidate.lmm.register_forward_hook(capture_final_read)
    with torch.no_grad():
        try:
            open_time, open_quantity = candidate.encode_task_states(dts, quantities, mask)
        finally:
            handle.remove()
        open_prediction = candidate.predict_quantity(open_quantity)[1]
        candidate.slot_memory_alpha_raw.zero_()
        closed_state = candidate.encode(dts, quantities, mask)
        closed_prediction = candidate.predict_quantity(closed_state)[1]
    assert len(final_read_calls) == 1
    assert torch.equal(open_time, open_quantity)
    assert not torch.equal(open_time[mask], closed_state[mask])
    assert not torch.equal(open_prediction[mask], closed_prediction[mask])
    assert torch.count_nonzero(open_time[~mask]) == 0


@pytest.mark.parametrize("objective", ["time_loss", "quantity_train_loss"])
def test_open_gate_all_address_parameters_receive_finite_nonzero_task_gradients(objective: str) -> None:
    _, candidate = build_pair(seed=233)
    candidate.eval()
    activate(candidate)
    dts, quantities, mask = batch()
    outputs = target_outputs(candidate, dts, mask, quantities, lambda_log_qty=1.0)
    outputs[objective].mean().backward()
    for name, parameter in candidate.slot_memory.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert candidate.slot_memory_alpha_raw.grad is not None
    assert torch.isfinite(candidate.slot_memory_alpha_raw.grad).all()
    assert torch.count_nonzero(candidate.slot_memory_alpha_raw.grad) > 0
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in candidate.encoder.parameters())
    assert candidate.lmm.mem.grad is not None
    assert torch.isfinite(candidate.lmm.mem.grad).all()
    assert torch.count_nonzero(candidate.lmm.mem.grad) > 0


def test_open_candidate_future_values_and_next_target_do_not_leak_into_predictions() -> None:
    _, candidate = build_pair(seed=241)
    candidate.eval()
    activate(candidate)
    dts, quantities, mask = batch()
    future_dts, future_quantities = dts.clone(), quantities.clone()
    future_dts[:, 3:] = 90000.0
    future_quantities[:, 3:] = 700000.0
    with torch.no_grad():
        reference = candidate.encode(dts, quantities, mask)
        future_changed = candidate.encode(future_dts, future_quantities, mask)
    assert torch.equal(reference[:, :3], future_changed[:, :3])

    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    target = mask.sum(dim=1) - 1
    rows = torch.arange(len(dts))
    changed_dts[rows, target] += 50000.0
    changed_quantities[rows, target] += 70000.0
    with torch.no_grad():
        expected = target_outputs(candidate, dts, mask, quantities, lambda_log_qty=1.0)
        actual = target_outputs(candidate, changed_dts, mask, changed_quantities, lambda_log_qty=1.0)
    assert torch.equal(actual["pred_qty"], expected["pred_qty"])
    assert not torch.equal(actual["true_qty"], expected["true_qty"])
    assert not torch.equal(actual["time_loss"], expected["time_loss"])
    assert not torch.equal(actual["quantity_train_loss"], expected["quantity_train_loss"])


def test_target_prediction_has_zero_target_input_gradient_and_live_history_gradients() -> None:
    _, candidate = build_pair(seed=251)
    candidate.eval()
    activate(candidate)
    dts, quantities, mask = batch()
    dts.requires_grad_()
    quantities.requires_grad_()
    outputs = target_outputs(candidate, dts, mask, quantities, lambda_log_qty=1.0)
    outputs["pred_qty"].sum().backward()
    target = mask.sum(dim=1) - 1
    rows = torch.arange(len(dts))
    for tensor in (dts, quantities):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad[rows, target]) == 0
        assert torch.count_nonzero(tensor.grad[:, 1:3]) > 0
        assert torch.count_nonzero(tensor.grad[~mask]) == 0


def test_model_padding_batch_permutation_and_forward_preserve_state() -> None:
    _, candidate = build_pair(seed=263)
    candidate.eval()
    activate(candidate)
    dts, quantities, mask = batch()
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    changed_dts[~mask], changed_quantities[~mask] = -99999.0, 1e30
    snapshot = {name: value.clone() for name, value in candidate.state_dict().items()}
    order = torch.tensor([1, 0])
    with torch.no_grad():
        expected = candidate.encode(dts, quantities, mask)
        padding_changed = candidate.encode(changed_dts, changed_quantities, mask)
        separate = torch.cat([candidate.encode(dts[i:i+1], quantities[i:i+1], mask[i:i+1]) for i in range(2)])
        permuted = candidate.encode(dts[order], quantities[order], mask[order])
        repeated = candidate.encode(dts, quantities, mask)
    assert torch.equal(padding_changed, expected)
    torch.testing.assert_close(separate, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(permuted, expected[order], rtol=1e-6, atol=1e-6)
    assert torch.equal(repeated, expected)
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, snapshot[name]), name


def test_model_optimizer_rng_roundtrip_reproduces_next_training_step() -> None:
    _, candidate = build_pair(seed=269)
    candidate.train()
    activate(candidate)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    dts, quantities, mask = batch()

    def train_step(model: CountAwareTitanSlotMemoryTPP, optim: torch.optim.Optimizer) -> torch.Tensor:
        optim.zero_grad(set_to_none=True)
        outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        loss = outputs["joint_loss"].mean()
        loss.backward()
        optim.step()
        return loss.detach().clone()

    train_step(candidate, optimizer)
    archive = io.BytesIO()
    torch.save({
        "model": candidate.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(),
    }, archive)
    expected_loss = train_step(candidate, optimizer)
    expected_rng = torch.get_rng_state().clone()
    expected_state = {name: value.clone() for name, value in candidate.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    archive.seek(0)
    checkpoint = torch.load(archive, weights_only=True)
    restored = CountAwareTitanSlotMemoryTPP(**model_kwargs()).train()
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    actual_loss = train_step(restored, restored_optimizer)
    assert torch.equal(actual_loss, expected_loss)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected_state[name]), name
    actual_optimizer = restored_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, expected_values in expected_optimizer["state"].items():
        for name, value in expected_values.items():
            actual_value = actual_optimizer["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(actual_value, value), (parameter_id, name)
            else:
                assert actual_value == value, (parameter_id, name)
