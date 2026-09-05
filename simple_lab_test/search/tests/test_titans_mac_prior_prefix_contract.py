"""Synthetic contracts for the opt-in prior-prefix final memory read.

These tests establish mechanism and replay properties, not forecasting gains.
The encoder probes bypass the initially zero quantity head where appropriate;
the separate quantity test exercises its real two-step training behavior.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from models.TPPs.CountAwareFactory import (
    build_count_aware_model,
    PRIOR_PREFIX_ROLE,
    validate_checkpoint_route,
)
from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP, LOG_MSE_VARIANT, LOGNORMAL_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED, TIME_HEAD_MODE_SCALED_EXACT,
)
from models.Titan.common.titans_mac import TitansMACEncoder, TitansMemoryState
from models.Titan.common.titans_mac_optimized import apply_titantpp_mac_semantic_optimization
from models.Titan.common.titans_mac_prior_prefix import (
    read_prior_prefix_with_fixed_shape_scan,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONES, TITAN_B012_BACKBONES, validate_model_role_contract,
)


BACKBONE = "titantpp_titans_mac_prior_prefix"
ROOT = Path(__file__).resolve().parents[3]
WRITER_PREFIXES = (
    "key_projection.", "value_projection.", "update_rate_projection.",
    "momentum_projection.", "forgetting_projection.",
)
if os.environ.get("REQUIRE_CUDA") == "1" and not torch.cuda.is_available():
    raise RuntimeError("REQUIRE_CUDA=1 requires a working CUDA runtime; refusing CPU-only validation")


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def encoder(policy="prior_prefix", *, cls=TitansMACEncoder):
    torch.manual_seed(314159)
    kwargs = {} if policy is None else {"output_read_policy": policy}
    result = cls(
        input_dim=2, d_model=8, n_layers=1, n_heads=2, d_ff=16,
        persistent_memory_size=2, segment_size=16, max_len=64,
        dropout=0.0, **kwargs,
    ).eval()
    result.neural_memory.gradient_max_norm = 1.0
    result.neural_memory.compile_cuda_scan = False
    return result


def inputs(batch=2, length=19):
    generator = torch.Generator().manual_seed(271828)
    return torch.randn(batch, length, 2, generator=generator)


def tensors(state):
    return (*state.memory_tensors(), *state.momentum_tensors(), state.positions)


def assert_state_equal(left, right):
    for actual, expected in zip(tensors(left), tensors(right), strict=True):
        assert torch.equal(actual, expected)
    if left.series_ids is None or right.series_ids is None:
        assert left.series_ids is right.series_ids
    else:
        assert torch.equal(left.series_ids, right.series_ids)


def assert_diagnostics_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def writer_norms(model):
    return {
        name: 0.0 if parameter.grad is None else float(parameter.grad.norm())
        for name, parameter in model.neural_memory.named_parameters()
        if name.startswith(WRITER_PREFIXES)
    }


def backward_probe(model, *, history_length):
    x = inputs(batch=1, length=32)
    positions = torch.arange(32).unsqueeze(0)
    mask = positions <= history_length
    write_mask = positions < history_length
    model.zero_grad(set_to_none=True)
    encoded = model(x, mask=mask, write_mask=write_mask)
    probe = torch.linspace(-1.1, 0.9, model.d_model)
    (encoded[:, history_length - 1] * probe).sum().backward()
    return writer_norms(model)


def candidate(**overrides):
    torch.manual_seed(123)
    kwargs = dict(hidden_dim=8, train_log_mean=1.5, max_seq_len=32)
    kwargs.update(overrides)
    model, metadata = build_count_aware_model(BACKBONE, **kwargs)
    assert isinstance(model, CountAwareTitanTPP)
    return model.eval(), metadata


def quantity_batch():
    return (
        torch.tensor([[0., 1., 2., 3., 4.], [0., 2., 3., 4., 0.]]),
        torch.tensor([[True, True, True, True, True],
                      [True, True, True, True, False]]),
        torch.tensor([[1., 2., 5., 13., 34.], [2., 3., 8., 21., 0.]]),
    )


def target_result(model, batch=None):
    dts, mask, quantities = quantity_batch() if batch is None else batch
    return target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)


def candidate_checkpoint():
    model, metadata = candidate()
    return {
        "backbone": BACKBONE, "variant": LOG_MSE_VARIANT,
        "encoder_config": metadata, "model_state_dict": model.state_dict(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
    }


# Frozen B1 forward from revision 567196e510e14211e0ee400ea312872bb3ec2703
# Full source SHA256: 9436fea650c66025fceb7aebb820941176f2c10c80e47d61c55555d8430e0ab1
# Kept in the test so source archives can replay without a Git object store.
def _historical_segment_start_forward(
    self,
    inputs: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    write_mask: torch.Tensor | None = None,
    state: TitansMemoryState | None = None,
    series_ids: torch.Tensor | None = None,
    segment_size: int | None = None,
    write_chunk_size: int | None = None,
) -> tuple[torch.Tensor, TitansMemoryState, dict[str, torch.Tensor]]:
    """Encode one batch and return explicit online state and diagnostics."""
    if inputs.ndim != 3 or inputs.size(-1) != self.input_dim:
        raise ValueError("inputs must have shape [batch, sequence, input_dim]")
    batch_size, seq_len, _ = inputs.shape
    mask = self._validate_mask(
        mask,
        batch_size=batch_size,
        seq_len=seq_len,
        device=inputs.device,
    )
    if write_mask is None:
        write_mask = mask
    else:
        write_mask = self._validate_mask(
            write_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=inputs.device,
        ) & mask
    state = self.neural_memory.prepare_state(
        state,
        batch_size=batch_size,
        device=inputs.device,
        dtype=inputs.dtype,
        series_ids=series_ids,
    )
    current_segment_size = (
        self.segment_size if segment_size is None else int(segment_size)
    )
    if current_segment_size < 1:
        raise ValueError("segment_size must be positive")

    projected = self.input_projection(inputs)
    projected = projected + self._position_values(state.positions, seq_len)
    projected = projected * mask.unsqueeze(-1).to(dtype=projected.dtype)
    outputs: list[torch.Tensor] = []
    diagnostics: dict[str, list[torch.Tensor]] = {
        "associative_loss": [],
        "update_rate": [],
        "momentum_rate": [],
        "forgetting_rate": [],
        "write_applied": [],
    }

    for start in range(0, seq_len, current_segment_size):
        end = min(start + current_segment_size, seq_len)
        current = projected[:, start:end]
        current_mask = mask[:, start:end]
        current_write_mask = write_mask[:, start:end]
        query = self.neural_memory.project_query(current)
        retrieved = self.neural_memory.read(state, query)
        persistent = self.persistent_memory.to(
            device=inputs.device,
            dtype=inputs.dtype,
        ).expand(batch_size, -1, -1)
        mac_tokens = torch.cat((persistent, retrieved, current), dim=1)
        prefix_valid = torch.ones(
            batch_size,
            self.persistent_memory_size,
            device=inputs.device,
            dtype=torch.bool,
        )
        valid_tokens = torch.cat(
            (prefix_valid, current_mask, current_mask),
            dim=1,
        )
        blocked_mask = self._mac_attention_mask(
            self.persistent_memory_size,
            end - start,
            device=inputs.device,
        )
        for layer in self.layers:
            mac_tokens = layer(
                mac_tokens,
                blocked_mask=blocked_mask,
                key_padding_mask=~valid_tokens,
                valid_mask=valid_tokens,
            )
        event_start = self.persistent_memory_size + (end - start)
        attention_output = mac_tokens[:, event_start:]

        # The prediction state reads the segment-start memory. Writes occur
        # only after these states are complete, so the target cannot leak.
        output_query = self.neural_memory.project_query(attention_output)
        output_memory = self.neural_memory.read(state, output_query)
        normalized_output = self.output_norm(attention_output)
        gated_memory = (
            torch.sigmoid(self.output_gate(normalized_output))
            * self.memory_output_projection(output_memory)
        )
        prediction_state = attention_output + gated_memory
        prediction_state = prediction_state * current_mask.unsqueeze(-1).to(
            dtype=prediction_state.dtype
        )
        outputs.append(prediction_state)

        state, write_diagnostics = self.neural_memory.write_sequence(
            state,
            attention_output,
            current_write_mask,
            chunk_size=write_chunk_size,
        )
        for name, value in write_diagnostics.items():
            diagnostics[name].append(value)

    encoded = torch.cat(outputs, dim=1) if outputs else projected
    next_positions = state.positions + mask.sum(dim=1)
    state = replace(state, positions=next_positions)
    combined_diagnostics = {
        name: (
            torch.cat(values, dim=1)
            if values
            else inputs.new_zeros(batch_size, 0)
        )
        for name, values in diagnostics.items()
    }
    return encoded, state, combined_diagnostics


def test_default_policy_replays_immutable_prechange_b1_exactly():
    old, current = encoder(None), encoder(None)
    old.forward_with_state = types.MethodType(_historical_segment_start_forward, old)
    assert current.output_read_policy == "segment_start"
    x = inputs()
    mask = torch.ones(2, 19, dtype=torch.bool)
    mask[1, 3] = False
    old_output, old_state, old_diag = old.forward_with_state(x, mask=mask)
    output, state, diag = current.forward_with_state(x, mask=mask)
    assert torch.equal(output, old_output)
    assert_state_equal(state, old_state)
    assert_diagnostics_equal(diag, old_diag)
    old_output.square().mean().backward()
    output.square().mean().backward()
    for (name, left), (_, right) in zip(
        old.named_parameters(), current.named_parameters(), strict=True,
    ):
        if left.grad is None or right.grad is None:
            assert left.grad is right.grad, name
        else:
            assert torch.equal(left.grad, right.grad), name


def test_policy_changes_neither_parameters_nor_write_trajectory():
    old, new = encoder("segment_start"), encoder()
    assert sum(p.numel() for p in old.parameters()) == sum(
        p.numel() for p in new.parameters()
    )
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    x = inputs(length=35)
    mask = torch.ones(2, 35, dtype=torch.bool)
    mask[0, [3, 18, 34]] = False
    mask[1, 29:] = False
    write_mask = mask.clone()
    write_mask[:, [0, 4, 16, 28]] = False
    with torch.no_grad():
        old_output, old_state, old_diag = old.forward_with_state(
            x, mask=mask, write_mask=write_mask,
        )
        output, state, diag = new.forward_with_state(
            x, mask=mask, write_mask=write_mask,
        )
    assert_state_equal(state, old_state)
    assert_diagnostics_equal(diag, old_diag)
    # Each segment's first event still reads its original segment-start state.
    assert torch.equal(output[:, [0, 16, 32]], old_output[:, [0, 16, 32]])
    assert not torch.equal(output[:, 2], old_output[:, 2])


@pytest.mark.parametrize("history_length,padded_length", [(1, 2), (1, 32), (2, 3), (2, 32)])
def test_h1_identity_and_h2_observed_write_visibility(history_length, padded_length):
    old, new = encoder("segment_start"), encoder()
    x = inputs(batch=1, length=padded_length)
    positions = torch.arange(padded_length).unsqueeze(0)
    mask = positions <= history_length
    writes = positions < history_length
    with torch.no_grad():
        previous = old(x, mask=mask, write_mask=writes)
        revised = new(x, mask=mask, write_mask=writes)
    assert torch.equal(previous[:, 0], revised[:, 0])
    if history_length == 2:
        assert (revised[:, history_length - 1] - previous[:, history_length - 1]).abs().max() > 1e-8


@pytest.mark.parametrize("write_position", [0, 1, 14, 15, 16])
def test_current_event_write_affects_only_later_outputs(write_position):
    model = encoder()
    x = inputs(batch=1, length=19)
    writes = torch.ones(1, 19, dtype=torch.bool)
    removed = writes.clone()
    removed[:, write_position] = False
    with torch.no_grad():
        full = model(x, write_mask=writes)
        changed = model(x, write_mask=removed)
    assert torch.equal(full[:, :write_position + 1], changed[:, :write_position + 1])
    assert (full[:, write_position + 1] - changed[:, write_position + 1]).abs().max() > 1e-8


def test_h16_connects_writer_and_h3_connects_momentum_without_detach():
    old = backward_probe(encoder("segment_start"), history_length=16)
    new = backward_probe(encoder(), history_length=16)
    assert len(old) == len(new) == 8
    assert all(value == 0.0 for value in old.values())
    assert all(value > 0.0 for value in new.values())
    h2 = backward_probe(encoder(), history_length=2)
    h3 = backward_probe(encoder(), history_length=3)
    for name in ("momentum_projection.weight", "momentum_projection.bias"):
        assert h2[name] == 0.0
        assert h3[name] > 0.0


def test_all_writes_disabled_is_exact_b1_and_changes_no_memory():
    old, new = encoder("segment_start"), encoder()
    x = inputs()
    write_mask = torch.zeros(2, 19, dtype=torch.bool)
    with torch.no_grad():
        output, state, diag = new.forward_with_state(x, write_mask=write_mask)
        baseline = old(x, write_mask=write_mask)
        initial = new.initial_state(2, device=x.device, dtype=x.dtype)
    assert torch.equal(output, baseline)
    for left, right in zip(tensors(state)[:-1], tensors(initial)[:-1], strict=True):
        assert torch.equal(left, right)
    assert all(torch.count_nonzero(value) == 0 for value in diag.values())


def test_future_events_padding_and_holes_do_not_leak_into_valid_prefix():
    model = encoder()
    x = inputs(length=35)
    mask = torch.ones(2, 35, dtype=torch.bool)
    mask[0, [1, 7, 17]] = False
    mask[0, 30:] = False
    mask[1] = False  # A fully padded row must remain finite and unwritten.
    changed_padding = x.clone()
    changed_padding[~mask] = 1e6
    changed_future = x.clone()
    changed_future[:, 15:] = -1e5
    with torch.no_grad():
        out, state, diag = model.forward_with_state(x, mask=mask)
        padded, padded_state, padded_diag = model.forward_with_state(
            changed_padding, mask=mask,
        )
        future = model(changed_future, mask=mask)
    assert torch.equal(out, padded)
    assert_state_equal(state, padded_state)
    assert_diagnostics_equal(diag, padded_diag)
    assert torch.equal(out[:, :15], future[:, :15])
    assert torch.count_nonzero(out[~mask]) == 0
    assert torch.equal(diag["write_applied"], mask.float())
    assert all(torch.isfinite(value).all() for value in (*tensors(state), out))


def test_empty_sequence_preserves_explicit_state_and_returns_empty_diagnostics():
    model = encoder()
    x = inputs(batch=2, length=0)
    ids = torch.tensor([11, 22])
    initial = model.initial_state(2, device=x.device, dtype=x.dtype, series_ids=ids)
    output, state, diag = model.forward_with_state(x, state=initial, series_ids=ids)
    assert output.shape == (2, 0, 8)
    assert_state_equal(state, initial)
    assert all(value.shape == (2, 0) for value in diag.values())


def test_series_reset_and_batch_permutation_keep_memory_row_independent():
    model = encoder()
    x = inputs(length=19)
    ids = torch.tensor([11, 22])
    permutation = torch.tensor([1, 0])
    with torch.no_grad():
        output, state, diag = model.forward_with_state(x, series_ids=ids)
        permuted, perm_state, perm_diag = model.forward_with_state(
            x[permutation], series_ids=ids[permutation],
        )
        continued, continued_state, _ = model.forward_with_state(
            x[:, :3], state=state, series_ids=torch.tensor([33, 22]),
        )
        fresh, fresh_state, _ = model.forward_with_state(
            x[:, :3], series_ids=torch.tensor([33, 22]),
        )
    assert torch.equal(permuted, output[permutation])
    for left, right in zip(tensors(perm_state), tensors(state), strict=True):
        assert torch.equal(left, right[permutation])
    for name in diag:
        assert torch.equal(perm_diag[name], diag[name][permutation])
    assert torch.equal(continued[0], fresh[0])
    for left, right in zip(tensors(continued_state), tensors(fresh_state), strict=True):
        assert torch.equal(left[0], right[0])
    assert continued_state.positions.tolist() == [3, 22]
    assert not torch.equal(continued[1], fresh[1])
    with pytest.raises(ValueError, match="series_ids are required"):
        model.forward_with_state(x[:, :1], state=state)


def test_write_chunk_scheduling_preserves_outputs_states_and_gradients():
    baseline = encoder()
    x = inputs(length=19)
    mask = torch.ones(2, 19, dtype=torch.bool)
    mask[0, [2, 8]] = False
    reference = baseline.forward_with_state(x, mask=mask, write_chunk_size=1)
    reference[0].square().mean().backward()
    for chunk_size in (7, 16):
        model = copy.deepcopy(baseline)
        model.zero_grad(set_to_none=True)
        output, state, diag = model.forward_with_state(
            x, mask=mask, write_chunk_size=chunk_size,
        )
        output.square().mean().backward()
        assert torch.equal(output, reference[0])
        assert_state_equal(state, reference[1])
        assert_diagnostics_equal(diag, reference[2])
        for left, right in zip(model.parameters(), baseline.parameters(), strict=True):
            if left.grad is None or right.grad is None:
                assert left.grad is right.grad
            else:
                assert torch.equal(left.grad, right.grad)


def test_eval_no_grad_still_writes_without_mutating_registered_parameters():
    model = encoder()
    before = copy.deepcopy(model.state_dict())
    x = inputs(length=19)
    with torch.no_grad():
        _, state, diag = model.forward_with_state(x)
        initial = model.initial_state(2, device=x.device, dtype=x.dtype)
    assert diag["write_applied"].sum() == 38
    assert any(not torch.equal(a, b) for a, b in zip(
        state.memory_tensors(), initial.memory_tensors(), strict=True,
    ))
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert all(parameter.grad is None for parameter in model.parameters())


def test_actual_quantity_head_blocks_initial_gradient_then_trains_writer():
    model, _ = candidate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert torch.count_nonzero(model.quantity_head.weight) == 0
    target_result(model)["quantity_train_loss"].mean().backward()
    first = writer_norms(model.titans_mac_encoder)
    assert all(value == 0.0 for value in first.values())
    assert model.quantity_head.weight.grad.norm() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    target_result(model)["quantity_train_loss"].mean().backward()
    second = writer_norms(model.titans_mac_encoder)
    assert all(value > 0.0 for value in second.values())


def test_real_target_path_masks_future_quantity_and_excludes_target_write(monkeypatch):
    model, _ = candidate()
    with torch.no_grad():
        model.quantity_head.weight.copy_(torch.linspace(-0.1, 0.1, 8).unsqueeze(0))
    calls = []
    original = model.encode_task_states

    def capture(dts, quantities, mask, *, memory_write_mask=None):
        result = original(dts, quantities, mask, memory_write_mask=memory_write_mask)
        calls.append((quantities.detach().clone(), memory_write_mask.detach().clone(),
                      result[0].detach().clone()))
        return result

    monkeypatch.setattr(model, "encode_task_states", capture)
    dts, mask, quantities = quantity_batch()
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    changed_dts[0, 4], changed_dts[1, 3] = 1e5, 2e5
    changed_quantities[0, 4], changed_quantities[1, 3] = 1e6, 2e6
    with torch.no_grad():
        original_result = target_result(model, (dts, mask, quantities))
        changed_result = target_result(model, (changed_dts, mask, changed_quantities))
    assert torch.equal(original_result["pred_qty"], changed_result["pred_qty"])
    assert not torch.equal(original_result["quantity_train_loss"],
                           changed_result["quantity_train_loss"])
    expected = torch.tensor([[True, True, True, True, False],
                             [True, True, True, False, False]])
    for history_quantities, write_mask, _ in calls:
        assert torch.equal(write_mask, expected)
        assert history_quantities[0, 4] == history_quantities[1, 3] == 0
    assert torch.equal(calls[0][2][0, :4], calls[1][2][0, :4])
    assert torch.equal(calls[0][2][1, :3], calls[1][2][1, :3])


def test_factory_freezes_candidate_policy_and_same_checkpoint_tensor_schema():
    model, metadata = candidate()
    torch.manual_seed(123)
    baseline, _ = build_count_aware_model(
        "titantpp_titans_mac", hidden_dim=8, train_log_mean=1.5,
        max_seq_len=32, titans_memory_gradient_clip=1.0,
    )
    assert model.memory_mode == "titans_mac_prior_prefix"
    assert model.titans_mac_encoder.output_read_policy == "prior_prefix"
    assert model.titans_mac_encoder.segment_size == 16
    assert model.titans_mac_encoder.neural_memory.gradient_max_norm == 1.0
    assert metadata["titans_output_read_policy"] == "prior_prefix"
    assert metadata["titans_prefix_read_contract_id"] == "titans_mac_prior_prefix_v1"
    assert model.state_dict().keys() == baseline.state_dict().keys()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name


def test_optimized_adapter_preserves_candidate_policy_rng_and_cpu_training():
    reference, _ = candidate()
    optimized = copy.deepcopy(reference)
    before = torch.random.get_rng_state().clone()
    apply_titantpp_mac_semantic_optimization(optimized)
    assert torch.equal(before, torch.random.get_rng_state())
    assert optimized.titans_mac_encoder.output_read_policy == "prior_prefix"
    assert optimized.state_dict().keys() == reference.state_dict().keys()
    original, revised = target_result(reference), target_result(optimized)
    for name in original:
        assert torch.equal(original[name], revised[name]), name
    original["joint_loss"].mean().backward()
    revised["joint_loss"].mean().backward()
    for left, right in zip(reference.parameters(), optimized.parameters(), strict=True):
        if left.grad is None or right.grad is None:
            assert left.grad is right.grad
        else:
            assert torch.equal(left.grad, right.grad)


@pytest.mark.parametrize("kwargs", [
    {"lambda_tail": 0.1}, {"quantity_variant": LOGNORMAL_VARIANT},
    {"time_head_mode": TIME_HEAD_MODE_SCALED_EXACT},
    {"titans_memory_gradient_clip": 0.5},
])
def test_factory_rejects_candidate_objective_and_stability_drift(kwargs):
    with pytest.raises(ValueError):
        candidate(**kwargs)


@pytest.mark.parametrize("expected_backbone", ["titantpp_titans_mac", "titantpp", "thp"])
@pytest.mark.parametrize("marker", ["backbone", "memory_mode", "titans_output_read_policy",
                                   "titans_prefix_read_contract_id", "backbone_contract_id"])
def test_each_candidate_identity_marker_blocks_legacy_checkpoint_route(expected_backbone, marker):
    original = candidate_checkpoint()
    # A partially stripped or incorrectly relabeled candidate must still be
    # rejected when any independent candidate marker survives.
    payload = {"backbone": expected_backbone, "encoder_config": {}}
    if marker == "backbone":
        payload[marker] = original[marker]
    else:
        payload["encoder_config"][marker] = original["encoder_config"][marker]
    with pytest.raises(ValueError, match="cannot be loaded as another backbone"):
        validate_checkpoint_route(payload, expected_backbone)


@pytest.mark.parametrize("time_head", [None, [], "legacy_clamped_rmtpp"])
def test_malformed_checkpoint_time_head_fails_with_contract_error(time_head):
    payload = candidate_checkpoint()
    payload["encoder_config"]["time_head"] = time_head
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload, BACKBONE)


def test_nonstring_checkpoint_tensor_key_fails_with_contract_error():
    payload = candidate_checkpoint()
    payload["model_state_dict"] = {7: torch.zeros(1)}
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload, BACKBONE)


def test_legacy_b1_checkpoint_cannot_be_reinterpreted_as_prior_prefix():
    model, metadata = build_count_aware_model(
        "titantpp_titans_mac", hidden_dim=8, train_log_mean=1.5, max_seq_len=32,
        titans_memory_gradient_clip=1.0,
    )
    payload = {**candidate_checkpoint(), "backbone": "titantpp_titans_mac",
               "encoder_config": metadata, "model_state_dict": model.state_dict()}
    validate_checkpoint_route(payload, "titantpp_titans_mac")
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload, BACKBONE)


def test_legacy_factory_metadata_and_paper_roles_remain_separate():
    _, metadata = build_count_aware_model(
        "titantpp_titans_mac", hidden_dim=8, train_log_mean=1.5, max_seq_len=32,
    )
    assert metadata["backbone_contract_id"] == "B1"
    assert metadata["candidate_name"] == "count_titan_faithful_titans_mac"
    assert metadata["titans_scan_backend"] == "compiled_sequence_cuda"
    assert metadata["titans_memory_gradient_clip"] is None
    assert metadata["titans_event_order"] == "segment_read_prediction_then_observed_write"
    assert "titans_prefix_read_contract_id" not in metadata
    assert "titans_output_read_policy" not in metadata
    assert BACKBONE not in BACKBONES
    assert TITAN_B012_BACKBONES == ("titantpp", "titantpp_titans_mac", "titantpp_tpp_gated_memory")


@pytest.mark.parametrize("overrides", [
    {"model_role": "experimental"},
    {"model_role": "t0_common_control"},
    {"backbones": ("titantpp_titans_mac",)},
    {"backbones": (BACKBONE, "titantpp_titans_mac")},
    {"quantity_variants": (LOGNORMAL_VARIANT,)},
])
def test_candidate_requires_dedicated_single_backbone_role(overrides):
    kwargs = dict(model_role=PRIOR_PREFIX_ROLE, backbones=(BACKBONE,),
                  quantity_variants=(LOG_MSE_VARIANT,),
                  time_head_mode=TIME_HEAD_MODE_LEGACY_CLAMPED, lambda_tail=0.0)
    validate_model_role_contract(**kwargs)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        validate_model_role_contract(**kwargs)


def test_contract_binds_frozen_references_and_keeps_adoption_claims_separate():
    contract = json.loads((ROOT / "paper/contracts/titans_mac_prior_prefix_v1.json").read_text())
    reference = contract["frozen_references"]
    assert hashlib.sha256((ROOT / reference["path"]).read_bytes()).hexdigest() == reference["sha256"]
    assert contract["baseline"] == "titantpp_titans_mac"
    assert set(reference["models"]) == {"titantpp", "rmtpp", "thp"}
    assert contract["architecture"]["additional_parameters"] == 0
    assert contract["architecture"]["segment_size"] == 16
    assert contract["training"]["time_intercept_limit"] == 300.0
    assert contract["phases"]["e1"]["performance_adoption"] is False
    assert contract["training"]["held_out_test_evaluated"] is False
    gates = contract["performance_gates"]
    assert gates["B1_incremental"]["instacart_body_mae_improvement_min"] == 0.01
    assert gates["T0_legacy_gate"]["all_datasets_body_mae_improvement_min"] == 0.05
    assert gates["T0_common_RMSE"]["all_datasets_strict_rmse_decrease"] is True
    assert gates["external_TPP_claim"]["required_for_seed_expansion"] is False


def launch_args(monkeypatch, *, dataset="insta_market_basket", epochs=300):
    from paper.scripts import titans_mac_prior_prefix_contract as contract_module
    contract = contract_module.load_contract()
    data = contract["datasets"][dataset]
    frozen = contract["training"]
    args = types.SimpleNamespace(
        backbones=BACKBONE, model_role=PRIOR_PREFIX_ROLE, hidden_dim=64,
        batch_size=128, lr=.001, lambda_log_qty=1., lambda_tail=0., grad_clip=1.,
        titans_memory_gradient_clip=1., titans_mac_execution_backend="optimized",
        time_head_mode=frozen["time_head"], time_scale=frozen["time_scale"],
        time_w_max=frozen["time_w_max"], time_intercept_limit=frozen["time_intercept_limit"],
        time_wd_safety_limit=frozen["time_wd_safety_limit"], time_head_lr_multiplier=1.,
        max_series=None, max_train_batches=None, max_val_batches=None,
        force_rerun=False, quantity_variants="log_mse", seeds="42", epochs=epochs,
        min_epochs=1 if epochs == 1 else 40,
        early_stopping_patience=1 if epochs == 1 else 40,
        dataset_contract=dataset, lookback_weeks=data["lookback"],
        max_seq_len=data["max_seq_len"], data=ROOT / data["data_path"],
        split_manifest=ROOT / data["split_manifest_path"],
    )
    # Real train/validation/test files are never opened by these launch tests.
    digests = {args.data: data["data_sha256"], args.split_manifest: data["split_manifest_sha256"]}
    monkeypatch.setattr(contract_module, "digest", lambda path: digests.get(path, "wrong-source"))
    return args, contract_module


@pytest.mark.parametrize("dataset", ["yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket"])
@pytest.mark.parametrize("epochs", [1, 300])
def test_frozen_full_data_launch_contract_accepts_each_phase_and_context(monkeypatch, dataset, epochs):
    args, module = launch_args(monkeypatch, dataset=dataset, epochs=epochs)
    module.validate_candidate_launch(args)


@pytest.mark.parametrize("field,value", [
    ("time_intercept_limit", 30.), ("titans_memory_gradient_clip", None),
    ("titans_mac_execution_backend", "reference"), ("batch_size", 64),
    ("max_train_batches", 1), ("max_val_batches", 1), ("max_series", 5),
    ("epochs", 2), ("seeds", "42,52,62"), ("seeds", "7"),
    ("lookback_weeks", 1), ("max_seq_len", 32), ("force_rerun", True),
    ("time_head_lr_multiplier", .1), ("quantity_variants", "tail_shared"),
])
def test_launch_rejects_budget_context_head_or_phase_drift(monkeypatch, field, value):
    args, module = launch_args(monkeypatch)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        module.validate_candidate_launch(args)


@pytest.mark.parametrize("seed", ["52", "62"])
def test_e1_is_restricted_to_frozen_seed42(monkeypatch, seed):
    args, module = launch_args(monkeypatch, epochs=1)
    args.seeds = seed
    with pytest.raises(ValueError):
        module.validate_candidate_launch(args)


def test_launch_rejects_source_checksum_drift_without_reading_real_data(monkeypatch):
    args, module = launch_args(monkeypatch)
    monkeypatch.setattr(module, "digest", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        module.validate_candidate_launch(args)


def synthetic_summary():
    return {
        "quantity_rows": [
            {"stratum": "le_p50", "count": 50, "qty_mae": 1.},
            {"stratum": "p50_p90", "count": 40, "qty_mae": 2.},
            {"stratum": "p90_p95", "count": 5, "qty_mae": 3.},
            {"stratum": "p95_p99", "count": 4, "qty_mae": 4.},
            {"stratum": "gt_p99", "count": 1, "qty_mae": 5.},
        ],
        "best_val_qty_mae": 1.66, "best_val_qty_rmse": 2.,
        "best_val_time_nll": -1., "best_val_log_qty_mse": .1,
    }


def test_body_metric_uses_target_counts_and_rejects_duplicate_strata():
    from paper.scripts.titans_mac_prior_prefix_contract import summary_metrics
    summary = synthetic_summary()
    metrics = summary_metrics(summary)
    assert metrics["body_mae"] == pytest.approx(145. / 95.)
    assert metrics["validation_targets"] == 100
    assert metrics["body_targets"] == 95
    summary["quantity_rows"][2]["stratum"] = "le_p50"
    with pytest.raises(ValueError):
        summary_metrics(summary)


def test_incremental_gate_does_not_imply_t0_or_external_superiority():
    from paper.scripts.titans_mac_prior_prefix_contract import load_contract, metric_gates
    candidate_metrics = dict(body_mae=9.9, qty_mae=10., qty_rmse=9., tail_mae=10., time_loss=-1.)
    baseline = {**candidate_metrics, "body_mae": 10., "qty_rmse": 10.}
    result = metric_gates(candidate_metrics, baseline, baseline,
                          {"rmtpp": baseline, "thp": baseline},
                          load_contract(), "insta_market_basket")
    assert all(result["checks"]["B1_incremental"].values())
    assert result["checks"]["T0_legacy_gate"]["body_improves_5pct"] is False
    assert result["screening_pass"] is False
    assert result["external_TPP_claim_pass"] is True
    candidate_metrics["qty_rmse"] = 10.
    equality = metric_gates(candidate_metrics, baseline, baseline,
                            {"rmtpp": baseline, "thp": baseline},
                            load_contract(), "insta_market_basket")
    assert equality["checks"]["B1_incremental"]["rmse_not_worse"] is True
    assert equality["checks"]["T0_common_RMSE"]["rmse_strictly_decreases"] is False
    assert equality["external_TPP_claim_pass"] is False


def test_external_claim_requires_all_frozen_comparators():
    from paper.scripts.titans_mac_prior_prefix_contract import load_contract, metric_gates
    metrics = dict(body_mae=10., qty_mae=10., qty_rmse=10., tail_mae=10., time_loss=-1.)
    with pytest.raises(ValueError):
        metric_gates(metrics, metrics, metrics, {}, load_contract(), "insta_market_basket")


def test_checkpoint_and_adam_restore_match_next_dropout_training_step(tmp_path):
    model, metadata = candidate()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step(network, opt):
        opt.zero_grad(set_to_none=True)
        loss = target_result(network)["joint_loss"].mean()
        loss.backward()
        opt.step()
        return loss.detach()

    step(model, optimizer)
    checkpoint = {
        "backbone": BACKBONE, "variant": LOG_MSE_VARIANT,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "encoder_config": metadata, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    path = tmp_path / "candidate.pt"
    torch.save(checkpoint, path)
    expected_loss = step(model, optimizer)
    restored, _ = candidate()
    restored.train()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_route(payload, BACKBONE)
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    restored_optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.random.set_rng_state(payload["torch_rng_state"])
    actual_loss = step(restored, restored_optimizer)
    assert torch.equal(actual_loss, expected_loss)
    for name, expected in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected), name
    for key, expected in optimizer.state_dict()["state"].items():
        for name, value in expected.items():
            assert torch.equal(restored_optimizer.state_dict()["state"][key][name], value)
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload, "titantpp_titans_mac")
    for key in ("titans_output_read_policy", "titans_prefix_read_contract_id"):
        incomplete = copy.deepcopy(payload)
        incomplete["encoder_config"].pop(key)
        with pytest.raises(ValueError):
            validate_checkpoint_route(incomplete, BACKBONE)


def test_stable_clip_one_extreme_features_have_finite_output_state_and_gradient():
    model = encoder()
    x = inputs(length=35) * 1e6
    output, state, diagnostics = model.forward_with_state(x)
    output.square().mean().backward()
    assert all(torch.isfinite(value).all() for value in (
        output, *tensors(state), *diagnostics.values(),
    ))
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_invalid_policy_and_write_chunk_are_rejected():
    with pytest.raises(ValueError, match="output_read_policy"):
        encoder("inclusive_current_write")
    with pytest.raises(ValueError, match="chunk_size"):
        encoder().forward_with_state(inputs(), write_chunk_size=0)


def scan_probe(model, *, fixed, chunk_size, compiled=False):
    parameter = next(model.parameters())
    generator = torch.Generator().manual_seed(17)
    y = torch.randn(3, 17, 8, generator=generator).to(parameter).requires_grad_(True)
    mask = torch.ones(3, 17, device=y.device, dtype=torch.bool)
    mask[0, [2, 7, 16]] = False
    mask[1, :9] = False
    mask[2] = False
    state = model.initial_state(
        3, device=y.device, dtype=y.dtype,
        series_ids=torch.tensor([11, 22, 33], device=y.device),
    )
    state = replace(state, **{
        name: torch.full_like(getattr(state, name), 0.03)
        for name in ("momentum_weight_1", "momentum_bias_1",
                     "momentum_weight_2", "momentum_bias_2")
    })
    queries = model.neural_memory.project_query(y)
    initial_reads = model.neural_memory.read(state, queries)
    if fixed:
        result = read_prior_prefix_with_fixed_shape_scan(
            model.neural_memory, state, y, queries, initial_reads, mask,
            chunk_size=chunk_size, compiled=compiled,
        )
    else:
        result = model._read_prior_prefix_and_write(
            state, y, queries, initial_reads, mask, chunk_size=chunk_size,
        )
    assert torch.equal(result[0][1, :10], initial_reads[1, :10])
    assert torch.equal(result[0][2], initial_reads[2])
    loss = result[0].square().mean() + 0.001 * sum(
        value.square().sum() for value in result[1].memory_tensors()
    )
    loss.backward()
    gradients = {
        name: parameter.grad for name, parameter in model.neural_memory.named_parameters()
    }
    return result, y.grad, gradients


@pytest.mark.parametrize("chunk_size", [7, 16])
def test_fixed_shape_scan_padding_and_cross_chunk_gradients_match_eager(chunk_size):
    reference = encoder().double()
    revised = copy.deepcopy(reference)
    (reads, state, diag), input_grad, gradients = scan_probe(
        reference, fixed=False, chunk_size=chunk_size,
    )
    (fixed_reads, fixed_state, fixed_diag), fixed_grad, fixed_gradients = scan_probe(
        revised, fixed=True, chunk_size=chunk_size,
    )
    for left, right in zip((reads, *tensors(state), input_grad),
                           (fixed_reads, *tensors(fixed_state), fixed_grad), strict=True):
        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)
    for name in diag:
        torch.testing.assert_close(diag[name], fixed_diag[name], atol=1e-11, rtol=1e-10)
    for name in gradients:
        assert gradients[name] is not None, name
        torch.testing.assert_close(gradients[name], fixed_gradients[name], atol=1e-11, rtol=1e-10)


def test_fixed_scan_rejects_cpu_compilation_and_handles_empty_input():
    model = encoder()
    y = torch.empty(2, 0, 8)
    mask = torch.empty(2, 0, dtype=torch.bool)
    state = model.initial_state(2, device=y.device, dtype=y.dtype)
    reads, next_state, diagnostics = read_prior_prefix_with_fixed_shape_scan(
        model.neural_memory, state, y, y, y, mask, compiled=False,
    )
    assert reads.shape == (2, 0, 8)
    assert_state_equal(state, next_state)
    assert all(value.shape == (2, 0) for value in diagnostics.values())
    with pytest.raises(ValueError, match="requires CUDA"):
        read_prior_prefix_with_fixed_shape_scan(
            model.neural_memory, state, y, y, y, mask, compiled=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity is opt-in on a GPU host")
def test_cuda_eager_prior_prefix_matches_cpu_forward_state_and_gradients():
    cpu = encoder()
    cuda = copy.deepcopy(cpu).cuda()
    x = inputs(length=19)
    mask = torch.ones(2, 19, dtype=torch.bool)
    mask[0, [3, 18]] = False
    cpu_output, cpu_state, cpu_diag = cpu.forward_with_state(x, mask=mask)
    gpu_output, gpu_state, gpu_diag = cuda.forward_with_state(x.cuda(), mask=mask.cuda())
    cpu_output.square().mean().backward()
    gpu_output.square().mean().backward()
    for left, right in zip((cpu_output, *tensors(cpu_state)),
                           (gpu_output, *tensors(gpu_state)), strict=True):
        torch.testing.assert_close(left, right.cpu(), atol=2e-5, rtol=2e-4)
    for name in cpu_diag:
        torch.testing.assert_close(cpu_diag[name], gpu_diag[name].cpu(), atol=2e-5, rtol=2e-4)
    for left, right in zip(cpu.parameters(), cuda.parameters(), strict=True):
        if left.grad is None or right.grad is None:
            assert left.grad is right.grad
        else:
            torch.testing.assert_close(left.grad, right.grad.cpu(), atol=2e-5, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Compiled parity requires a GPU host")
def test_cuda_compiled_fixed_scan_matches_eager_forward_and_gradients():
    eager = encoder().cuda()
    compiled = copy.deepcopy(eager)
    (reads, state, diag), input_grad, gradients = scan_probe(
        eager, fixed=False, chunk_size=16,
    )
    (fixed_reads, fixed_state, fixed_diag), fixed_grad, fixed_gradients = scan_probe(
        compiled, fixed=True, chunk_size=16, compiled=True,
    )
    for left, right in zip((reads, *tensors(state), input_grad),
                           (fixed_reads, *tensors(fixed_state), fixed_grad), strict=True):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)
    for name in diag:
        torch.testing.assert_close(diag[name], fixed_diag[name], atol=2e-5, rtol=2e-4)
    for name in gradients:
        torch.testing.assert_close(gradients[name], fixed_gradients[name], atol=2e-5, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Compiled model integration requires a GPU host")
def test_cuda_optimized_actual_target_path_matches_eager_candidate():
    import io

    reference, _ = candidate()
    with torch.no_grad():
        reference.quantity_head.weight.copy_(torch.linspace(-0.1, 0.1, 8).unsqueeze(0))
    reference.cuda()
    reference.titans_mac_encoder.neural_memory.compile_cuda_scan = False
    optimized = copy.deepcopy(reference)
    apply_titantpp_mac_semantic_optimization(optimized)
    assert optimized.titans_mac_encoder.neural_memory.compile_cuda_scan is True
    batch = tuple(value.cuda() for value in quantity_batch())
    original, revised = target_result(reference, batch), target_result(optimized, batch)
    for name in original:
        torch.testing.assert_close(original[name], revised[name], atol=2e-5, rtol=2e-4)
    original["joint_loss"].mean().backward()
    revised["joint_loss"].mean().backward()
    assert all(value > 0 for value in writer_norms(reference.titans_mac_encoder).values())
    assert all(value > 0 for value in writer_norms(optimized.titans_mac_encoder).values())
    for left, right in zip(reference.parameters(), optimized.parameters(), strict=True):
        if left.grad is None or right.grad is None:
            assert left.grad is right.grad
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=3e-3)

    optimizer_kwargs = dict(lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), **optimizer_kwargs)
    optimized_optimizer = torch.optim.AdamW(optimized.parameters(), **optimizer_kwargs)
    # Consume the gradients compared above using the frozen outer clipping and
    # optimizer contract; small gradient differences can be amplified by AdamW.
    for model, optimizer in ((reference, reference_optimizer), (optimized, optimized_optimizer)):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    for left, right in zip(reference.parameters(), optimized.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=3e-4)

    optimized.train()

    def optimized_step():
        optimized_optimizer.zero_grad(set_to_none=True)
        outputs = target_result(optimized, batch)
        outputs["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(optimized.parameters(), 1.0, error_if_nonfinite=True)
        optimized_optimizer.step()
        return {name: value.detach().clone() for name, value in outputs.items()}

    # Compile the training path before checkpointing so replay compares the same
    # established CUDA backend. Each target_outputs call resets window memory.
    optimized_step()
    checkpoint = io.BytesIO()
    torch.save({
        "model_state_dict": optimized.state_dict(),
        "optimizer_state_dict": optimized_optimizer.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }, checkpoint)
    expected_outputs = optimized_step()
    expected_model = copy.deepcopy(optimized.state_dict())
    expected_optimizer = copy.deepcopy(optimized_optimizer.state_dict())
    expected_cpu_rng = torch.random.get_rng_state().clone()
    expected_cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]

    checkpoint.seek(0)
    payload = torch.load(checkpoint, weights_only=True)
    optimized.load_state_dict(payload["model_state_dict"], strict=True)
    optimized_optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.random.set_rng_state(payload["torch_rng_state"])
    torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    actual_outputs = optimized_step()
    for name, expected in expected_outputs.items():
        assert torch.equal(actual_outputs[name], expected), name
    for name, expected in expected_model.items():
        assert torch.equal(optimized.state_dict()[name], expected), name
    actual_optimizer = optimized_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for key, expected_state in expected_optimizer["state"].items():
        for name, expected in expected_state.items():
            actual = actual_optimizer["state"][key][name]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual, expected), (key, name)
            else:
                assert actual == expected, (key, name)
    assert torch.equal(torch.random.get_rng_state(), expected_cpu_rng)
    for actual, expected in zip(torch.cuda.get_rng_state_all(), expected_cuda_rng, strict=True):
        assert torch.equal(actual, expected)
