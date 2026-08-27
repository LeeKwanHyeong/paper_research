from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_TPP_GATED,
)
from models.Titan.common.tpp_gated_memory import (
    TPPGatedMemoryState,
    TPPSpecificGatedMemory,
    _scan_tpp_gated_sequence,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONE_LABELS,
    SUPPORTED_BACKBONES,
    TITAN_MEMORY_BACKBONES,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "paper/contracts/count_aware_titans_backbone_reproduction_v1.json"
)


def build_memory(
    *,
    d_model: int = 4,
    memory_size: int = 8,
    topk: int = 2,
) -> TPPSpecificGatedMemory:
    torch.manual_seed(101)
    return TPPSpecificGatedMemory(
        d_model=d_model,
        memory_size=memory_size,
        topk=topk,
        temperature=1.0,
        dropout=0.0,
    ).eval()


def build_model() -> tuple[CountAwareTitanTPP, dict[str, object]]:
    torch.manual_seed(103)
    model, metadata = build_count_aware_model(
        "titantpp_tpp_gated_memory",
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=16,
    )
    assert isinstance(model, CountAwareTitanTPP)
    return model, metadata


def assert_state_allclose(
    left: TPPGatedMemoryState,
    right: TPPGatedMemoryState,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> None:
    assert torch.allclose(left.keys, right.keys, atol=atol, rtol=rtol)
    assert torch.allclose(left.values, right.values, atol=atol, rtol=rtol)
    assert torch.equal(left.valid_slots, right.valid_slots)
    assert torch.equal(left.write_counts, right.write_counts)
    assert torch.equal(left.positions, right.positions)
    if left.series_ids is None or right.series_ids is None:
        assert left.series_ids is right.series_ids
    else:
        assert torch.equal(left.series_ids, right.series_ids)


def open_memory_gate(memory: TPPSpecificGatedMemory) -> None:
    nn.init.zeros_(memory.null_logit_projection.weight)
    nn.init.constant_(memory.null_logit_projection.bias, -10.0)
    nn.init.zeros_(memory.confidence_projection.weight)
    nn.init.constant_(memory.confidence_projection.bias, 10.0)


def test_b2_contract_factory_runner_and_persistent_memory_match() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())
    candidate, metadata = build_model()
    b0, b0_metadata = build_count_aware_model(
        "titantpp",
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=16,
    )
    b1, b1_metadata = build_count_aware_model(
        "titantpp_titans_mac",
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=16,
    )

    assert contract["variants"]["B2"]["implementation"] == (
        "TPPSpecificGatedMemory"
    )
    assert contract["b2_status"] == "implemented_pending_matched_validation"
    assert contract["b2_mechanism"]["relationship_to_titans"] == (
        "project_specific_proposal_not_faithful_titans_ltm"
    )
    assert contract["b2_mechanism"]["cuda_scan_backend"] == (
        "compiled_fullgraph_exact_read_before_write_recurrence"
    )
    assert "titantpp_tpp_gated_memory" in TITAN_MEMORY_BACKBONES
    assert "titantpp_tpp_gated_memory" in SUPPORTED_BACKBONES
    assert BACKBONE_LABELS["titantpp_tpp_gated_memory"] == (
        "TitanTPP TPP-specific Gated Memory"
    )
    assert candidate.memory_mode == TITAN_MEMORY_MODE_TPP_GATED
    assert isinstance(candidate.tpp_gated_memory, TPPSpecificGatedMemory)
    assert candidate.titans_mac_encoder is None
    assert candidate.lmm is None
    assert candidate.soft_memory is None
    assert candidate.surprise_memory is None
    assert metadata["candidate_name"] == "count_titan_tpp_specific_gated_memory"
    assert metadata["backbone_contract_id"] == "B2"
    assert metadata["tpp_gated_memory_size"] == 64
    assert metadata["tpp_gated_topk"] == 4
    assert metadata["tpp_gated_retrieval"] == (
        "similarity_weighted_sparse_topk_with_null"
    )
    assert metadata["tpp_gated_write_policy"] == (
        "circular_observed_event_after_prediction"
    )
    assert metadata["tpp_gated_scan_backend"] == "compiled_sequence_cuda"
    assert metadata["persistent_mem_size"] == 16
    assert b0_metadata["persistent_mem_size"] == 16
    assert b1_metadata["persistent_mem_size"] == 16
    assert candidate.encoder is not None
    assert all(layer.attn.persistent_mem_size == 16 for layer in candidate.encoder.layers)
    assert isinstance(b0, CountAwareTitanTPP)
    assert b0.encoder is not None
    assert all(layer.attn.persistent_mem_size == 16 for layer in b0.encoder.layers)
    assert isinstance(b1, CountAwareTitanTPP)
    assert b1.titans_mac_encoder is not None
    assert b1.titans_mac_encoder.persistent_memory_size == 16


def test_sparse_retrieval_uses_similarity_weights_and_only_topk_slots() -> None:
    memory = build_memory(d_model=2, memory_size=4, topk=2)
    memory.input_norm = nn.Identity()
    memory.output_norm = nn.Identity()
    with torch.no_grad():
        memory.query_projection.weight.copy_(torch.eye(2))
        memory.output_projection.weight.copy_(torch.eye(2))
        memory.null_logit_projection.weight.zero_()
        memory.null_logit_projection.bias.fill_(-4.0)
        memory.confidence_projection.weight.zero_()
        memory.confidence_projection.bias.fill_(10.0)
    state = TPPGatedMemoryState(
        keys=torch.tensor(
            [[[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]]]
        ),
        values=torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0], [5.0, 5.0], [-5.0, -5.0]]]
        ),
        valid_slots=torch.ones(1, 4, dtype=torch.bool),
        write_counts=torch.tensor([4]),
        positions=torch.tensor([4]),
    )

    residual, diagnostics = memory.retrieve_token(state, torch.tensor([[1.0, 0.0]]))

    assert diagnostics["selected_indices"].tolist() == [[0, 1]]
    selected_weights = diagnostics["selected_weights"][0]
    assert selected_weights[0] > selected_weights[1] > 0.0
    assert torch.allclose(
        selected_weights.sum() + diagnostics["null_probability"][0],
        torch.tensor(1.0),
        atol=1e-6,
    )
    assert torch.isfinite(residual).all()


def test_null_memory_preserves_local_state_when_no_slot_is_available() -> None:
    memory = build_memory()
    encoded = torch.randn(2, 3, memory.d_model)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    write_mask = torch.zeros_like(mask)

    output, state, diagnostics = memory.forward_with_state(
        encoded,
        mask=mask,
        write_mask=write_mask,
    )

    expected = encoded * mask.unsqueeze(-1)
    assert torch.equal(output, expected)
    assert torch.equal(
        diagnostics["null_probability"],
        mask.to(dtype=encoded.dtype),
    )
    assert torch.count_nonzero(diagnostics["retrieval_confidence"]) == 0
    assert torch.count_nonzero(diagnostics["effective_gate"]) == 0
    assert torch.count_nonzero(state.valid_slots) == 0


def test_prediction_is_built_before_current_event_write() -> None:
    memory = build_memory()
    open_memory_gate(memory)
    encoded = torch.randn(1, 4, memory.d_model)
    mask = torch.ones(1, 4, dtype=torch.bool)

    written, written_state, written_diagnostics = memory.forward_with_state(
        encoded,
        mask=mask,
        write_mask=mask,
    )
    unwritten, unwritten_state, _ = memory.forward_with_state(
        encoded,
        mask=mask,
        write_mask=torch.zeros_like(mask),
    )

    assert torch.equal(written[:, 0], unwritten[:, 0])
    assert not torch.allclose(written[:, 1:], unwritten[:, 1:])
    assert written_state.write_counts.tolist() == [4]
    assert unwritten_state.write_counts.tolist() == [0]
    assert torch.equal(written_diagnostics["write_applied"], mask.float())


def test_nonwritable_target_cannot_change_memory_state_or_prefix() -> None:
    model, _ = build_model()
    model.eval()
    assert model.tpp_gated_memory is not None
    open_memory_gate(model.tpp_gated_memory)
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
        original, original_state, original_diagnostics = (
            model.encode_with_memory_state(
                dts,
                quantities,
                mask,
                memory_write_mask=write_mask,
                series_ids=torch.tensor([7]),
            )
        )
        changed, changed_state, changed_diagnostics = (
            model.encode_with_memory_state(
                changed_dts,
                changed_quantities,
                mask,
                memory_write_mask=write_mask,
                series_ids=torch.tensor([7]),
            )
        )

    assert torch.allclose(original[:, :-1], changed[:, :-1], atol=1e-6)
    assert_state_allclose(original_state, changed_state)
    assert original_diagnostics["write_applied"][0, -1] == 0.0
    assert changed_diagnostics["write_applied"][0, -1] == 0.0


def test_padding_does_not_change_output_or_state() -> None:
    memory = build_memory()
    encoded = torch.randn(1, 4, memory.d_model)
    changed = encoded.clone()
    changed[:, 2:] = 1_000_000.0
    mask = torch.tensor([[True, True, False, False]])

    original, original_state, original_diagnostics = memory.forward_with_state(
        encoded,
        mask=mask,
    )
    mutated, mutated_state, mutated_diagnostics = memory.forward_with_state(
        changed,
        mask=mask,
    )

    assert torch.equal(original, mutated)
    assert_state_allclose(original_state, mutated_state)
    for name in original_diagnostics:
        assert torch.equal(original_diagnostics[name], mutated_diagnostics[name])
        assert torch.count_nonzero(original_diagnostics[name][:, 2:]) == 0


def test_series_change_resets_only_changed_row() -> None:
    memory = build_memory()
    first = torch.randn(2, 3, memory.d_model)
    second = torch.randn(2, 2, memory.d_model)
    first_mask = torch.ones(2, 3, dtype=torch.bool)
    second_mask = torch.ones(2, 2, dtype=torch.bool)

    _, first_state, _ = memory.forward_with_state(
        first,
        mask=first_mask,
        series_ids=torch.tensor([10, 20]),
    )
    continued, continued_state, _ = memory.forward_with_state(
        second,
        mask=second_mask,
        state=first_state,
        series_ids=torch.tensor([11, 20]),
    )
    fresh, fresh_state, _ = memory.forward_with_state(
        second[:1],
        mask=second_mask[:1],
        series_ids=torch.tensor([11]),
    )

    assert torch.allclose(continued[0], fresh[0], atol=1e-6)
    assert torch.allclose(continued_state.keys[0], fresh_state.keys[0], atol=1e-6)
    assert torch.allclose(
        continued_state.values[0],
        fresh_state.values[0],
        atol=1e-6,
    )
    assert torch.equal(continued_state.valid_slots[0], fresh_state.valid_slots[0])
    assert continued_state.write_counts.tolist() == [2, 5]
    assert continued_state.positions.tolist() == [2, 5]
    assert continued_state.series_ids is not None
    assert continued_state.series_ids.tolist() == [11, 20]


def test_write_chunk_schedules_are_numerically_identical() -> None:
    memory = build_memory()
    encoded = torch.randn(2, 7, memory.d_model)
    mask = torch.tensor(
        [
            [True, True, False, True, True, True, False],
            [True, False, True, True, False, True, True],
        ]
    )

    token_output, token_state, token_diagnostics = memory.forward_with_state(
        encoded,
        mask=mask,
        write_chunk_size=1,
    )
    chunk_output, chunk_state, chunk_diagnostics = memory.forward_with_state(
        encoded,
        mask=mask,
        write_chunk_size=4,
    )

    assert torch.allclose(token_output, chunk_output, atol=1e-6, rtol=1e-5)
    assert_state_allclose(token_state, chunk_state, atol=1e-6, rtol=1e-5)
    for name in token_diagnostics:
        assert torch.allclose(
            token_diagnostics[name],
            chunk_diagnostics[name],
            atol=1e-6,
            rtol=1e-5,
        )


def test_compilable_scan_matches_eager_read_before_write_reference() -> None:
    memory = build_memory()
    encoded = torch.randn(2, 7, memory.d_model)
    mask = torch.tensor(
        [
            [True, True, False, True, True, True, False],
            [True, False, True, True, False, True, True],
        ]
    )
    write_mask = mask.clone()
    write_mask[:, -2] = False
    series_ids = torch.tensor([1, 2])
    initial = memory.initial_state(
        2,
        device=encoded.device,
        dtype=encoded.dtype,
        series_ids=series_ids,
    )
    eager_output, eager_state, eager_diagnostics = memory.forward_with_state(
        encoded,
        mask=mask,
        write_mask=write_mask,
        state=initial,
        series_ids=series_ids,
        write_chunk_size=1,
    )
    normalized = memory.input_norm(encoded)
    queries = torch.nn.functional.normalize(
        memory.query_projection(normalized),
        dim=-1,
        eps=1e-6,
    )
    write_keys = torch.nn.functional.normalize(
        memory.key_projection(normalized),
        dim=-1,
        eps=1e-6,
    )
    scanned = _scan_tpp_gated_sequence(
        initial.keys,
        initial.values,
        initial.valid_slots,
        initial.write_counts,
        encoded,
        normalized,
        queries,
        write_keys,
        memory.value_projection(normalized),
        memory.null_logit_projection(normalized),
        memory.confidence_projection.weight,
        memory.confidence_projection.bias,
        memory.output_norm.weight,
        memory.output_norm.bias,
        memory.output_projection.weight,
        mask,
        write_mask,
        memory.topk,
        memory.temperature,
        memory.dropout.p,
        memory.training,
    )
    scanned_state = TPPGatedMemoryState(
        keys=scanned[1],
        values=scanned[2],
        valid_slots=scanned[3],
        write_counts=scanned[4],
        positions=initial.positions + mask.sum(dim=1),
        series_ids=initial.series_ids,
    )
    scanned_diagnostics = {
        "null_probability": scanned[5],
        "retrieval_confidence": scanned[6],
        "learned_confidence": scanned[7],
        "effective_gate": scanned[8],
        "selected_slot_count": scanned[9],
        "write_applied": scanned[10],
    }

    assert torch.allclose(scanned[0], eager_output, atol=1e-6, rtol=1e-5)
    assert_state_allclose(eager_state, scanned_state, atol=1e-6, rtol=1e-5)
    for name, expected in eager_diagnostics.items():
        assert torch.allclose(
            scanned_diagnostics[name],
            expected,
            atol=1e-6,
            rtol=1e-5,
        )


def test_extreme_inputs_keep_model_backward_state_and_diagnostics_finite() -> None:
    model, _ = build_model()
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
    encoded, state, diagnostics = model.encode_with_memory_state(
        dts,
        quantities,
        mask,
        series_ids=torch.tensor([1, 2]),
    )

    assert torch.isfinite(encoded).all()
    assert all(
        torch.isfinite(value).all()
        for value in outputs.values()
        if torch.is_tensor(value)
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert torch.isfinite(state.keys).all()
    assert torch.isfinite(state.values).all()
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
