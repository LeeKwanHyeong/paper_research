from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_NONE,
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_MEMORY_MODE_STATIC_SOFT_GATED,
    TITAN_MEMORY_MODE_SURPRISE_GATED,
    TITAN_MEMORY_MODE_TITANS_MAC,
)
from models.Titan.common.memory import (
    GatedSoftMemory,
    HardLocalMemoryMatcher,
    LMM,
    SurpriseGatedMemory,
)
from models.Titan.common.titans_mac import TitansMACEncoder
from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONES,
    BACKBONE_LABELS,
    SUPPORTED_BACKBONES,
    TITAN_MEMORY_BACKBONES,
)


def build_titan(backbone: str) -> tuple[CountAwareTitanTPP, dict[str, object]]:
    torch.manual_seed(41)
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
    )
    assert isinstance(model, CountAwareTitanTPP)
    return model, metadata


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 2.0, 4.0, 0.0],
    ])
    mask = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
    ])
    quantities = torch.tensor([
        [2.0, 5.0, 9.0, 20.0],
        [1.0, 4.0, 12.0, 0.0],
    ])
    return dts, mask, quantities


def legacy_surprise_process(
    memory: SurpriseGatedMemory,
    encoded: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reference the original event-wise implementation for regression tests."""
    batch_size, seq_len, _ = encoded.shape
    state = encoded.new_zeros(batch_size, memory.d_model, memory.memory_rank)
    momentum = torch.zeros_like(state)
    outputs: list[torch.Tensor] = []
    surprise_values: list[torch.Tensor] = []
    update_rates: list[torch.Tensor] = []
    retentions: list[torch.Tensor] = []
    retrieval_gates: list[torch.Tensor] = []
    momentum_rate = torch.sigmoid(memory.momentum_logit)
    rank_scale = math.sqrt(memory.memory_rank)

    for chunk_start in range(0, seq_len, memory.chunk_size):
        if chunk_start:
            state = state.detach()
            momentum = momentum.detach()
        chunk_end = min(chunk_start + memory.chunk_size, seq_len)
        for position in range(chunk_start, chunk_end):
            valid = mask[:, position]
            valid_vector = valid.to(dtype=encoded.dtype).unsqueeze(-1)
            valid_state = valid_vector.unsqueeze(-1)
            current = encoded[:, position]
            normalized = memory.input_norm(current)
            query = F.normalize(
                memory.query_proj(normalized),
                dim=-1,
                eps=1e-6,
            )
            key = F.normalize(
                memory.key_proj(normalized),
                dim=-1,
                eps=1e-6,
            )
            value = torch.tanh(memory.value_proj(normalized))
            retrieved = torch.bmm(state, query.unsqueeze(-1)).squeeze(-1)
            retrieved = memory.output_proj(memory.retrieval_norm(retrieved))
            retrieval_gate = torch.sigmoid(memory.gate_proj(normalized))
            residual = (
                torch.tanh(memory.residual_scale)
                * retrieval_gate
                * memory.dropout(retrieved)
            )
            outputs.append((current + residual) * valid_vector)

            predicted_value = torch.bmm(
                state,
                key.unsqueeze(-1),
            ).squeeze(-1)
            error = value - predicted_value
            surprise = torch.linalg.vector_norm(error, dim=-1)
            update_rate = torch.sigmoid(memory.update_rate_proj(normalized))
            retention = torch.sigmoid(memory.retention_proj(normalized))
            gradient_step = torch.bmm(
                error.unsqueeze(-1),
                key.unsqueeze(1),
            ) / rank_scale
            next_momentum = (
                momentum_rate * momentum
                + update_rate.unsqueeze(-1) * gradient_step
            )
            next_state = (
                retention.unsqueeze(-1) * state + next_momentum
            ).clamp(min=-memory.memory_clip, max=memory.memory_clip)
            state = valid_state * next_state + (1.0 - valid_state) * state
            momentum = (
                valid_state * next_momentum + (1.0 - valid_state) * momentum
            )

            surprise_values.append(surprise * valid_vector.squeeze(-1))
            update_rates.append(
                update_rate.squeeze(-1) * valid_vector.squeeze(-1)
            )
            retentions.append(
                retention.squeeze(-1) * valid_vector.squeeze(-1)
            )
            retrieval_gates.append(
                retrieval_gate.mean(dim=-1) * valid_vector.squeeze(-1)
            )

    return torch.stack(outputs, dim=1), {
        "surprise": torch.stack(surprise_values, dim=1),
        "update_rate": torch.stack(update_rates, dim=1),
        "retention": torch.stack(retentions, dim=1),
        "retrieval_gate": torch.stack(retrieval_gates, dim=1),
    }


def test_memory_variants_are_opt_in_supported_backbones() -> None:
    assert TITAN_MEMORY_BACKBONES == (
        "titantpp_no_memory",
        "titantpp_gated_soft_memory",
        "titantpp_surprise_memory",
        "titantpp_persistent_only",
        "titantpp_persistent_surprise_memory",
        "titantpp_dual_memory_shared",
        "titantpp_dual_memory_adapter_only",
        "titantpp_titans_mac",
        "titantpp_tpp_gated_memory",
        "titantpp_weighted_static_memory",
    )
    assert all(name not in BACKBONES for name in TITAN_MEMORY_BACKBONES)
    assert SUPPORTED_BACKBONES == (*BACKBONES, *TITAN_MEMORY_BACKBONES)
    assert BACKBONE_LABELS["titantpp_no_memory"] == "TitanTPP No Memory"
    assert BACKBONE_LABELS["titantpp_gated_soft_memory"] == (
        "TitanTPP Gated Soft Memory"
    )
    assert BACKBONE_LABELS["titantpp_surprise_memory"] == (
        "TitanTPP Surprise Memory"
    )
    assert BACKBONE_LABELS["titantpp_titans_mac"] == (
        "TitanTPP Faithful Titans-MAC"
    )
    assert BACKBONE_LABELS["titantpp_tpp_gated_memory"] == (
        "TitanTPP TPP-specific Gated Memory"
    )


def test_historical_lmm_alias_resolves_to_hard_local_matcher() -> None:
    assert LMM is HardLocalMemoryMatcher


def test_faithful_titans_mac_factory_contract_and_finite_backward() -> None:
    candidate, metadata = build_titan("titantpp_titans_mac")
    dts, mask, quantities = sample_batch()

    assert candidate.memory_mode == TITAN_MEMORY_MODE_TITANS_MAC
    assert candidate.encoder is None
    assert isinstance(candidate.titans_mac_encoder, TitansMACEncoder)
    assert candidate.lmm is None
    assert candidate.soft_memory is None
    assert candidate.surprise_memory is None
    assert metadata["candidate_name"] == "count_titan_faithful_titans_mac"
    assert metadata["backbone_contract_id"] == "B1"
    assert metadata["titans_neural_memory_depth"] == 2
    assert metadata["titans_mac_segment_size"] == 16
    assert metadata["titans_scan_backend"] == "compiled_sequence_cuda"
    assert metadata["persistent_memory_update_scope"] == "outer_loop_only"

    outputs = target_outputs(
        candidate,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    outputs["joint_loss"].mean().backward()

    assert all(
        torch.isfinite(value).all()
        for value in outputs.values()
        if torch.is_tensor(value)
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in candidate.parameters()
    )


def test_no_memory_backbone_removes_both_static_memory_paths() -> None:
    control, control_metadata = build_titan("titantpp")
    candidate, candidate_metadata = build_titan("titantpp_no_memory")

    assert control.memory_mode == TITAN_MEMORY_MODE_STATIC_HARD
    assert candidate.memory_mode == TITAN_MEMORY_MODE_NONE
    assert control.lmm is not None
    assert candidate.lmm is None
    assert all(
        layer.attn.persistent_mem is None
        for layer in candidate.encoder.layers
    )
    assert control_metadata["candidate_name"] == "count_titan_small_lmm"
    assert candidate_metadata["candidate_name"] == "count_titan_no_memory"
    assert candidate_metadata["persistent_mem_size"] == 0
    assert candidate_metadata["lmm_mem_size"] == 0
    assert sum(parameter.numel() for parameter in candidate.parameters()) < sum(
        parameter.numel() for parameter in control.parameters()
    )


def test_no_memory_forward_backward_is_finite_and_causal() -> None:
    model, _ = build_titan("titantpp_no_memory")
    model.eval()
    dts, mask, quantities = sample_batch()

    outputs = target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    outputs["joint_loss"].mean().backward()

    assert all(
        torch.isfinite(value).all()
        for value in outputs.values()
        if torch.is_tensor(value)
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[0, -1] = 1_000_000.0
    changed_quantities[0, -1] = 1_000_000.0
    with torch.no_grad():
        original = model.encode(dts, quantities, mask)
        changed = model.encode(changed_dts, changed_quantities, mask)

    assert torch.allclose(original[0, :-1], changed[0, :-1], atol=1e-6)
    assert torch.count_nonzero(original[1, -1]) == 0
    assert torch.count_nonzero(changed[1, -1]) == 0


def test_gated_soft_memory_starts_as_exact_no_memory_residual() -> None:
    control, _ = build_titan("titantpp_no_memory")
    candidate, metadata = build_titan("titantpp_gated_soft_memory")
    control.eval()
    candidate.eval()
    dts, mask, quantities = sample_batch()

    assert candidate.memory_mode == TITAN_MEMORY_MODE_STATIC_SOFT_GATED
    assert candidate.lmm is None
    assert isinstance(candidate.soft_memory, GatedSoftMemory)
    assert metadata["candidate_name"] == "count_titan_gated_soft_memory"
    assert metadata["memory_residual_gate_init"] == 0.0
    assert all(
        layer.attn.persistent_mem is None
        for layer in candidate.encoder.layers
    )

    with torch.no_grad():
        control_hidden = control.encode(dts, quantities, mask)
        candidate_hidden = candidate.encode(dts, quantities, mask)

    assert torch.equal(control_hidden, candidate_hidden)


def test_gated_soft_memory_uses_dense_normalized_retrieval() -> None:
    candidate, _ = build_titan("titantpp_gated_soft_memory")
    assert candidate.soft_memory is not None
    encoded = torch.randn(2, 5, candidate.hidden_dim)

    _, weights = candidate.soft_memory.retrieve(encoded)

    assert weights.shape == (2, 5, candidate.soft_memory.mem_size)
    assert torch.all(weights > 0.0)
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(2, 5),
        atol=1e-6,
    )


def test_open_gated_soft_memory_routes_finite_gradients_to_memory() -> None:
    candidate, _ = build_titan("titantpp_gated_soft_memory")
    assert candidate.soft_memory is not None
    candidate.soft_memory.residual_scale.data.fill_(0.5)
    dts, mask, quantities = sample_batch()

    loss = target_outputs(
        candidate,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )["joint_loss"].mean()
    loss.backward()

    memory_parameters = (
        candidate.soft_memory.memory_keys,
        candidate.soft_memory.memory_values,
        candidate.soft_memory.query_proj.weight,
        candidate.soft_memory.output_proj.weight,
    )
    assert all(parameter.grad is not None for parameter in memory_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in memory_parameters
        if parameter.grad is not None
    )
    assert all(
        torch.count_nonzero(parameter.grad) > 0
        for parameter in memory_parameters
        if parameter.grad is not None
    )


def test_surprise_memory_starts_as_exact_no_memory_residual() -> None:
    control, _ = build_titan("titantpp_no_memory")
    candidate, metadata = build_titan("titantpp_surprise_memory")
    control.eval()
    candidate.eval()
    dts, mask, quantities = sample_batch()

    assert candidate.memory_mode == TITAN_MEMORY_MODE_SURPRISE_GATED
    assert isinstance(candidate.surprise_memory, SurpriseGatedMemory)
    assert candidate.lmm is None
    assert candidate.soft_memory is None
    assert metadata["candidate_name"] == "count_titan_surprise_memory"
    assert metadata["surprise_memory_rank"] == 16
    assert metadata["surprise_chunk_size"] == 32
    assert metadata["surprise_scan_backend"] == "compiled_sequence_cuda"
    assert metadata["surprise_state_scope"] == "independent_input_sequence"

    with torch.no_grad():
        control_hidden = control.encode(dts, quantities, mask)
        candidate_hidden = candidate.encode(dts, quantities, mask)

    assert torch.equal(control_hidden, candidate_hidden)


def test_surprise_memory_is_prefix_causal_and_resets_between_calls() -> None:
    candidate, _ = build_titan("titantpp_surprise_memory")
    candidate.eval()
    assert candidate.surprise_memory is not None
    candidate.surprise_memory.residual_scale.data.fill_(0.5)
    dts, mask, quantities = sample_batch()
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[0, -1] = 1_000_000.0
    changed_quantities[0, -1] = 1_000_000.0

    with torch.no_grad():
        original = candidate.encode(dts, quantities, mask)
        repeated = candidate.encode(dts, quantities, mask)
        changed = candidate.encode(changed_dts, changed_quantities, mask)

    assert torch.equal(original, repeated)
    assert torch.allclose(original[0, :-1], changed[0, :-1], atol=1e-6)
    assert torch.count_nonzero(original[1, -1]) == 0
    assert torch.count_nonzero(changed[1, -1]) == 0


def test_surprise_memory_padding_does_not_update_fast_weight_state() -> None:
    memory = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=2,
        dropout=0.0,
    ).eval()
    memory.residual_scale.data.fill_(0.5)
    encoded = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, True, True]])
    changed = encoded.clone()
    changed[:, 1] = 1_000_000.0

    with torch.no_grad():
        original, original_diagnostics = memory.process(encoded, mask=mask)
        mutated, mutated_diagnostics = memory.process(changed, mask=mask)

    assert torch.equal(original, mutated)
    for key in original_diagnostics:
        assert torch.equal(original_diagnostics[key], mutated_diagnostics[key])
        assert original_diagnostics[key][0, 1] == 0.0


def test_surprise_chunking_changes_gradient_horizon_not_forward_values() -> None:
    short = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=2,
        dropout=0.0,
    ).eval()
    long = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=8,
        dropout=0.0,
    ).eval()
    long.load_state_dict(short.state_dict())
    short.residual_scale.data.fill_(0.5)
    long.residual_scale.data.fill_(0.5)
    encoded = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)

    with torch.no_grad():
        short_output, short_diagnostics = short.process(encoded, mask=mask)
        long_output, long_diagnostics = long.process(encoded, mask=mask)

    assert torch.equal(short_output, long_output)
    for key in short_diagnostics:
        assert torch.equal(short_diagnostics[key], long_diagnostics[key])


def test_open_surprise_memory_routes_finite_gradients_through_updates() -> None:
    candidate, _ = build_titan("titantpp_surprise_memory")
    assert candidate.surprise_memory is not None
    candidate.surprise_memory.residual_scale.data.fill_(0.5)
    dts, mask, quantities = sample_batch()

    loss = target_outputs(
        candidate,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )["joint_loss"].mean()
    loss.backward()

    memory_parameters = (
        candidate.surprise_memory.query_proj.weight,
        candidate.surprise_memory.key_proj.weight,
        candidate.surprise_memory.value_proj.weight,
        candidate.surprise_memory.output_proj.weight,
        candidate.surprise_memory.update_rate_proj.bias,
        candidate.surprise_memory.momentum_logit,
    )
    assert all(parameter.grad is not None for parameter in memory_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in memory_parameters
        if parameter.grad is not None
    )
    assert all(
        torch.count_nonzero(parameter.grad) > 0
        for parameter in memory_parameters
        if parameter.grad is not None
    )


def test_vectorized_surprise_process_matches_event_wise_reference() -> None:
    memory = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=3,
        dropout=0.0,
    ).eval()
    memory.residual_scale.data.fill_(0.5)
    encoded = torch.randn(3, 7, 8)
    mask = torch.tensor([
        [True, True, True, True, True, True, True],
        [True, False, True, True, True, True, True],
        [False, False, True, True, True, True, True],
    ])

    with torch.no_grad():
        reference_output, reference_diagnostics = legacy_surprise_process(
            memory,
            encoded,
            mask,
        )
        output, diagnostics = memory.process(encoded, mask=mask)

    assert torch.allclose(output, reference_output, atol=1e-6, rtol=1e-5)
    for key in reference_diagnostics:
        assert torch.allclose(
            diagnostics[key],
            reference_diagnostics[key],
            atol=1e-6,
            rtol=1e-5,
        )


def test_vectorized_surprise_process_matches_reference_gradients() -> None:
    reference = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=3,
        dropout=0.0,
    ).eval()
    optimized = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=3,
        dropout=0.0,
    ).eval()
    optimized.load_state_dict(reference.state_dict())
    reference.residual_scale.data.fill_(0.5)
    optimized.residual_scale.data.fill_(0.5)
    reference_input = torch.randn(2, 7, 8, requires_grad=True)
    optimized_input = reference_input.detach().clone().requires_grad_(True)
    mask = torch.ones(2, 7, dtype=torch.bool)

    reference_output, reference_diagnostics = legacy_surprise_process(
        reference,
        reference_input,
        mask,
    )
    output, diagnostics = optimized.process(optimized_input, mask=mask)
    reference_loss = reference_output.square().mean()
    optimized_loss = output.square().mean()
    for key in reference_diagnostics:
        reference_loss = reference_loss + 0.01 * reference_diagnostics[key].mean()
        optimized_loss = optimized_loss + 0.01 * diagnostics[key].mean()
    reference_loss.backward()
    optimized_loss.backward()

    assert torch.allclose(
        optimized_input.grad,
        reference_input.grad,
        atol=2e-6,
        rtol=2e-5,
    )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in optimized.named_parameters():
        reference_gradient = reference_parameters[name].grad
        assert parameter.grad is not None
        assert reference_gradient is not None
        assert torch.allclose(
            parameter.grad,
            reference_gradient,
            atol=2e-6,
            rtol=2e-5,
        ), name


def test_surprise_forward_skips_diagnostic_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=3,
        dropout=0.0,
    ).eval()
    encoded = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)

    def fail_surprise_norm(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("diagnostic norm should not run in forward")

    monkeypatch.setattr(memory, "_diagnostic_surprise", fail_surprise_norm)

    output = memory(encoded, mask=mask)
    assert output.shape == encoded.shape
    with pytest.raises(AssertionError, match="diagnostic norm"):
        memory.process(encoded, mask=mask)


def test_surprise_event_local_projections_run_once_per_sequence() -> None:
    memory = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=3,
        dropout=0.0,
    ).eval()
    encoded = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    counts = {
        name: 0
        for name in (
            "input_norm",
            "query_proj",
            "key_proj",
            "value_proj",
            "gate_proj",
            "update_rate_proj",
            "retention_proj",
            "retrieval_norm",
            "output_proj",
        )
    }
    handles = []
    for name in counts:
        module = getattr(memory, name)

        def count_call(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
            *,
            module_name: str = name,
        ) -> None:
            counts[module_name] += 1

        handles.append(module.register_forward_hook(count_call))

    try:
        memory(encoded, mask=mask)
    finally:
        for handle in handles:
            handle.remove()

    assert counts == {name: 1 for name in counts}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_surprise_scan_matches_cuda_eager_gradients() -> None:
    device = torch.device("cuda")
    eager = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=3,
        dropout=0.0,
        compile_cuda_scan=False,
    ).to(device).eval()
    compiled = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=3,
        dropout=0.0,
        compile_cuda_scan=True,
    ).to(device).eval()
    compiled.load_state_dict(eager.state_dict())
    eager.residual_scale.data.fill_(0.5)
    compiled.residual_scale.data.fill_(0.5)
    eager_input = torch.randn(2, 7, 8, device=device, requires_grad=True)
    compiled_input = eager_input.detach().clone().requires_grad_(True)
    mask = torch.tensor([
        [True, True, True, True, True, True, True],
        [False, True, True, True, True, True, True],
    ], device=device)

    eager_output = eager(eager_input, mask=mask)
    compiled_output = compiled(compiled_input, mask=mask)
    eager_output.square().mean().backward()
    compiled_output.square().mean().backward()

    assert torch.allclose(compiled_output, eager_output, atol=2e-6, rtol=2e-5)
    assert torch.allclose(
        compiled_input.grad,
        eager_input.grad,
        atol=2e-6,
        rtol=2e-5,
    )
    eager_parameters = dict(eager.named_parameters())
    for name, parameter in compiled.named_parameters():
        eager_gradient = eager_parameters[name].grad
        assert parameter.grad is not None
        assert eager_gradient is not None
        assert torch.allclose(
            parameter.grad,
            eager_gradient,
            atol=2e-6,
            rtol=2e-5,
        ), name
