from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.Titan.common.titans_mac import (
    TitansMACEncoder,
    TitansMemoryState,
    TitansNeuralMemory,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "paper/contracts/count_aware_titans_backbone_reproduction_v1.json"
)


def build_encoder(*, segment_size: int = 2) -> TitansMACEncoder:
    torch.manual_seed(123)
    return TitansMACEncoder(
        input_dim=2,
        d_model=8,
        n_layers=1,
        n_heads=2,
        d_ff=16,
        persistent_memory_size=2,
        segment_size=segment_size,
        max_len=32,
        dropout=0.0,
    ).eval()


def build_model() -> CountAwareTitanTPP:
    torch.manual_seed(123)
    model, _ = build_count_aware_model(
        "titantpp_titans_mac",
        hidden_dim=8,
        train_log_mean=1.5,
        max_seq_len=16,
    )
    assert isinstance(model, CountAwareTitanTPP)
    return model


def assert_state_allclose(
    left: TitansMemoryState,
    right: TitansMemoryState,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> None:
    for left_tensor, right_tensor in zip(
        (*left.memory_tensors(), *left.momentum_tensors()),
        (*right.memory_tensors(), *right.momentum_tensors()),
        strict=True,
    ):
        assert torch.allclose(left_tensor, right_tensor, atol=atol, rtol=rtol)
    assert torch.equal(left.positions, right.positions)
    if left.series_ids is None or right.series_ids is None:
        assert left.series_ids is right.series_ids
    else:
        assert torch.equal(left.series_ids, right.series_ids)


def assert_state_finite(state: TitansMemoryState) -> None:
    assert all(
        torch.isfinite(tensor).all()
        for tensor in (*state.memory_tensors(), *state.momentum_tensors())
    )


def test_reproduction_contract_separates_b0_b1_and_b2() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())

    assert contract["variants"]["B0"]["implementation"] == (
        "HardLocalMemoryMatcher"
    )
    assert contract["variants"]["B0"]["faithful_titans_ltm"] is False
    assert contract["variants"]["B1"]["name"] == "Faithful Titans-MAC"
    assert contract["variants"]["B1"]["test_time_weight_update"] is True
    assert contract["variants"]["B2"]["faithful_titans_ltm"] is False
    assert contract["b2_status"] == "implemented_pending_matched_validation"
    assert contract["matched_t0_boundary"]["held_out_test"] == "locked"


def test_associative_gradients_match_autograd_exactly() -> None:
    torch.manual_seed(5)
    memory = TitansNeuralMemory(d_model=4, hidden_expansion=2)
    state = memory.initial_state(
        2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    state = TitansMemoryState(
        *(tensor.detach().clone().requires_grad_(True) for tensor in state.memory_tensors()),
        *(tensor.detach().clone() for tensor in state.momentum_tensors()),
        positions=state.positions,
    )
    keys = torch.randn(2, 4, dtype=torch.float64)
    values = torch.randn(2, 4, dtype=torch.float64)

    analytical, losses = memory.associative_gradients(state, keys, values)
    autograd = torch.autograd.grad(losses.sum(), state.memory_tensors())

    for expected, actual in zip(autograd, analytical, strict=True):
        assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)


def test_prediction_state_is_built_before_current_event_write() -> None:
    encoder = build_encoder(segment_size=1)
    inputs = torch.tensor([[[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    all_writes = mask.clone()
    no_writes = torch.zeros_like(mask)

    with torch.no_grad():
        written, written_state, written_diag = encoder.forward_with_state(
            inputs,
            mask=mask,
            write_mask=all_writes,
            segment_size=1,
        )
        unwritten, unwritten_state, unwritten_diag = encoder.forward_with_state(
            inputs,
            mask=mask,
            write_mask=no_writes,
            segment_size=1,
        )

    assert torch.equal(written[:, 0], unwritten[:, 0])
    assert not torch.allclose(written[:, 1:], unwritten[:, 1:])
    assert torch.equal(written_diag["write_applied"], mask.float())
    assert torch.equal(unwritten_diag["write_applied"], no_writes.float())
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            written_state.memory_tensors(),
            unwritten_state.memory_tensors(),
            strict=True,
        )
    )


def test_future_event_cannot_change_prefix_predictions() -> None:
    encoder = build_encoder(segment_size=4)
    inputs = torch.tensor(
        [[[0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [3.0, 8.0]]]
    )
    changed = inputs.clone()
    changed[:, -1] = torch.tensor([1_000_000.0, -1_000_000.0])
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        original = encoder(inputs, mask=mask)
        mutated = encoder(changed, mask=mask)

    assert torch.allclose(original[:, :-1], mutated[:, :-1], atol=1e-6)
    assert not torch.allclose(original[:, -1], mutated[:, -1])


def test_target_write_mask_blocks_target_from_memory_state() -> None:
    model = build_model().eval()
    dts = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    quantities = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[:, -1] = 1_000_000.0
    changed_quantities[:, -1] = 1_000_000.0
    mask = torch.ones(1, 4, dtype=torch.bool)
    write_mask = mask.clone()
    write_mask[:, -1] = False

    with torch.no_grad():
        original, original_state, original_diag = model.encode_with_memory_state(
            dts,
            quantities,
            mask,
            memory_write_mask=write_mask,
        )
        changed, changed_state, changed_diag = model.encode_with_memory_state(
            changed_dts,
            changed_quantities,
            mask,
            memory_write_mask=write_mask,
        )

    assert torch.allclose(original[:, :-1], changed[:, :-1], atol=1e-6)
    assert_state_allclose(original_state, changed_state)
    assert original_diag["write_applied"][0, -1] == 0.0
    assert changed_diag["write_applied"][0, -1] == 0.0


def test_target_outputs_marks_each_final_target_as_non_writable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_model().eval()
    dts = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 0.0],
        ]
    )
    quantities = torch.tensor(
        [
            [1.0, 2.0, 4.0, 8.0],
            [1.0, 3.0, 9.0, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    captured: dict[str, torch.Tensor] = {}
    original = model.encode_task_states

    def capture_write_mask(
        batch_dts: torch.Tensor,
        batch_quantities: torch.Tensor,
        batch_mask: torch.Tensor,
        *,
        memory_write_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert memory_write_mask is not None
        captured["mask"] = memory_write_mask.detach().clone()
        return original(
            batch_dts,
            batch_quantities,
            batch_mask,
            memory_write_mask=memory_write_mask,
        )

    monkeypatch.setattr(model, "encode_task_states", capture_write_mask)
    target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )

    assert captured["mask"].tolist() == [
        [True, True, True, False],
        [True, True, False, False],
    ]


def test_padding_changes_neither_output_nor_online_state() -> None:
    encoder = build_encoder(segment_size=2)
    inputs = torch.tensor(
        [[[0.0, 1.0], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    changed = inputs.clone()
    changed[:, 2:] = 1_000_000.0
    mask = torch.tensor([[True, True, False, False]])

    with torch.no_grad():
        original, original_state, original_diag = encoder.forward_with_state(
            inputs,
            mask=mask,
        )
        mutated, mutated_state, mutated_diag = encoder.forward_with_state(
            changed,
            mask=mask,
        )

    assert torch.equal(original, mutated)
    assert_state_allclose(original_state, mutated_state)
    for name in original_diag:
        assert torch.equal(original_diag[name], mutated_diag[name])
        assert torch.count_nonzero(original_diag[name][:, 2:]) == 0


def test_batch_rows_have_independent_memory_state() -> None:
    encoder = build_encoder(segment_size=1)
    inputs = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]],
            [[0.0, 3.0], [1.0, 6.0], [2.0, 12.0]],
        ]
    )
    changed = inputs.clone()
    changed[1] = 1_000.0
    mask = torch.ones(2, 3, dtype=torch.bool)

    with torch.no_grad():
        original, original_state, _ = encoder.forward_with_state(inputs, mask=mask)
        mutated, mutated_state, _ = encoder.forward_with_state(changed, mask=mask)

    assert torch.equal(original[0], mutated[0])
    for original_tensor, mutated_tensor in zip(
        (*original_state.memory_tensors(), *original_state.momentum_tensors()),
        (*mutated_state.memory_tensors(), *mutated_state.momentum_tensors()),
        strict=True,
    ):
        assert torch.equal(original_tensor[0], mutated_tensor[0])


def test_series_change_resets_only_the_changed_batch_row() -> None:
    encoder = build_encoder(segment_size=1)
    first = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[0.0, 3.0], [1.0, 6.0]],
        ]
    )
    second = torch.tensor(
        [
            [[2.0, 4.0], [3.0, 8.0]],
            [[2.0, 12.0], [3.0, 24.0]],
        ]
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    first_ids = torch.tensor([10, 20])
    second_ids = torch.tensor([11, 20])

    with torch.no_grad():
        _, first_state, _ = encoder.forward_with_state(
            first,
            mask=mask,
            series_ids=first_ids,
        )
        continued, continued_state, _ = encoder.forward_with_state(
            second,
            mask=mask,
            state=first_state,
            series_ids=second_ids,
        )
        fresh, fresh_state, _ = encoder.forward_with_state(
            second[:1],
            mask=mask[:1],
            series_ids=second_ids[:1],
        )

    assert torch.allclose(continued[0], fresh[0], atol=1e-6)
    for continued_tensor, fresh_tensor in zip(
        (*continued_state.memory_tensors(), *continued_state.momentum_tensors()),
        (*fresh_state.memory_tensors(), *fresh_state.momentum_tensors()),
        strict=True,
    ):
        assert torch.allclose(continued_tensor[0], fresh_tensor[0], atol=1e-6)
    assert continued_state.positions.tolist() == [2, 4]
    assert continued_state.series_ids is not None
    assert continued_state.series_ids.tolist() == [11, 20]


def test_reusing_state_without_series_ids_is_rejected() -> None:
    encoder = build_encoder(segment_size=1)
    inputs = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)
    _, state, _ = encoder.forward_with_state(
        inputs,
        mask=mask,
        series_ids=torch.tensor([7]),
    )

    with pytest.raises(ValueError, match="series_ids are required"):
        encoder.forward_with_state(inputs, mask=mask, state=state)


def test_token_and_chunk_write_schedules_are_numerically_identical() -> None:
    torch.manual_seed(13)
    memory = TitansNeuralMemory(d_model=6)
    inputs = torch.randn(2, 7, 6)
    write_mask = torch.tensor(
        [
            [True, True, False, True, True, True, False],
            [True, False, True, True, False, True, True],
        ]
    )
    initial = memory.initial_state(
        2,
        device=inputs.device,
        dtype=inputs.dtype,
    )

    token_state, token_diag = memory.write_sequence(
        initial,
        inputs,
        write_mask,
        chunk_size=1,
    )
    chunk_state, chunk_diag = memory.write_sequence(
        initial,
        inputs,
        write_mask,
        chunk_size=4,
    )

    assert_state_allclose(token_state, chunk_state, atol=1e-6, rtol=1e-5)
    for name in token_diag:
        assert torch.allclose(
            token_diag[name],
            chunk_diag[name],
            atol=1e-6,
            rtol=1e-5,
        )


def test_momentum_forgetting_and_invalid_write_follow_exact_boundaries() -> None:
    torch.manual_seed(17)
    memory = TitansNeuralMemory(d_model=4)
    inputs = torch.randn(2, 4)
    state = memory.initial_state(
        2,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    seeded_momenta = tuple(torch.full_like(tensor, 0.03) for tensor in state.momentum_tensors())
    state = TitansMemoryState(
        *state.memory_tensors(),
        *seeded_momenta,
        positions=state.positions,
    )
    keys, values, theta, eta, alpha = memory._project_write(inputs)
    gradients, _ = memory.associative_gradients(state, keys, values)

    next_state, diagnostics = memory.write_token(
        state,
        inputs,
        torch.tensor([True, False]),
    )

    for parameter, momentum, gradient, next_parameter, next_momentum in zip(
        state.memory_tensors(),
        state.momentum_tensors(),
        gradients,
        next_state.memory_tensors(),
        next_state.momentum_tensors(),
        strict=True,
    ):
        view_shape = (2, *([1] * (parameter.ndim - 1)))
        expected_momentum = eta.squeeze(-1).view(view_shape) * momentum
        expected_momentum = expected_momentum - theta.squeeze(-1).view(
            view_shape
        ) * gradient
        expected_parameter = (
            1.0 - alpha.squeeze(-1).view(view_shape)
        ) * parameter + expected_momentum
        assert torch.allclose(next_momentum[0], expected_momentum[0])
        assert torch.allclose(next_parameter[0], expected_parameter[0])
        assert torch.equal(next_momentum[1], momentum[1])
        assert torch.equal(next_parameter[1], parameter[1])
    assert diagnostics["write_applied"].tolist() == [1.0, 0.0]


def test_persistent_memory_is_fixed_in_eval_but_trainable_in_outer_loop() -> None:
    encoder = build_encoder(segment_size=1)
    inputs = torch.randn(1, 3, 2)
    mask = torch.ones(1, 3, dtype=torch.bool)
    before = encoder.persistent_memory.detach().clone()

    with torch.no_grad():
        encoder(inputs, mask=mask)

    assert torch.equal(encoder.persistent_memory, before)
    assert encoder.persistent_memory.grad is None

    encoder.train()
    encoder(inputs, mask=mask).square().mean().backward()
    assert encoder.persistent_memory.grad is not None
    assert torch.isfinite(encoder.persistent_memory.grad).all()
    assert torch.count_nonzero(encoder.persistent_memory.grad) > 0


def test_extreme_finite_inputs_keep_forward_backward_and_state_finite() -> None:
    model = build_model()
    dts = torch.tensor(
        [
            [0.0, 1e-6, 1e3, 1e6],
            [0.0, 1e6, 1e-6, 1e3],
        ]
    )
    quantities = torch.tensor(
        [
            [0.0, 1.0, 1e6, 1e9],
            [1e9, 1e6, 1.0, 0.0],
        ]
    )
    mask = torch.ones(2, 4, dtype=torch.bool)

    outputs = target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    outputs["joint_loss"].mean().backward()
    _, state, diagnostics = model.encode_with_memory_state(
        dts,
        quantities,
        mask,
        series_ids=torch.tensor([1, 2]),
    )

    assert all(
        torch.isfinite(value).all()
        for value in outputs.values()
        if torch.is_tensor(value)
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert_state_finite(state)
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
