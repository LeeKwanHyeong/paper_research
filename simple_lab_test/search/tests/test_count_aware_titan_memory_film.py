from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.TPPs.CountAwareTitanMemoryFiLM import (
    CountAwareTitanMemoryFiLM,
    MEMORY_FILM_BACKBONE,
    MEMORY_FILM_CONTRACT_ID,
    MEMORY_FILM_ROLE,
    memory_film_metadata,
    validate_memory_film_checkpoint,
)
from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_STATIC_HARD,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


ROOT = Path(__file__).resolve().parents[3]


def build_pair(seed: int = 29) -> tuple[CountAwareTitanTPP, CountAwareTitanMemoryFiLM]:
    kwargs = {
        "hidden_dim": 16,
        "train_log_mean": 1.5,
        "max_seq_len": 12,
    }
    torch.manual_seed(seed)
    baseline = CountAwareTitanTPP(
        **kwargs,
        memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
    )
    torch.manual_seed(seed)
    candidate = CountAwareTitanMemoryFiLM(**kwargs)
    return baseline, candidate


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor(
        [
            [0.0, 1.0, 2.0, 4.0, 8.0],
            [0.0, 0.5, 3.0, 0.0, 0.0],
        ]
    )
    quantities = torch.tensor(
        [
            [2.0, 5.0, 9.0, 12.0, 20.0],
            [1.0, 7.0, 13.0, 0.0, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    return dts, quantities, mask


def test_contract_declares_a_structural_candidate_and_local_boundary() -> None:
    contract = (
        ROOT / "paper/contracts/hard_lmm_memory_film_v1.md"
    ).read_text(encoding="utf-8")
    normalized_contract = " ".join(contract.split())

    assert MEMORY_FILM_CONTRACT_ID == "hard_lmm_memory_film_v1"
    assert MEMORY_FILM_BACKBONE == "titantpp_hard_memory_film"
    assert "This is a backbone candidate" in normalized_contract
    assert "same prototype tensor" in normalized_contract
    assert "Passing these gates establishes only implementation" in normalized_contract
    assert "outside this local contract task" in normalized_contract


def test_identity_initialization_preserves_b_state_and_predictions_exactly() -> None:
    baseline, candidate = build_pair()
    baseline.eval()
    candidate.eval()
    dts, quantities, mask = sample_batch()

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert set(candidate_state) - set(baseline_state) == {
        "film_scale_gain",
        "film_shift_gain",
    }
    for name, value in baseline_state.items():
        assert torch.equal(value, candidate_state[name]), name
    assert torch.count_nonzero(candidate.film_scale_gain).item() == 0
    assert torch.count_nonzero(candidate.film_shift_gain).item() == 0

    with torch.no_grad():
        baseline_time, baseline_quantity = baseline.encode_task_states(
            dts, quantities, mask
        )
        candidate_time, candidate_quantity = candidate.encode_task_states(
            dts, quantities, mask
        )
        baseline_outputs = target_outputs(
            baseline, dts, mask, quantities, lambda_log_qty=1.0
        )
        candidate_outputs = target_outputs(
            candidate, dts, mask, quantities, lambda_log_qty=1.0
        )

    assert torch.equal(candidate_time, baseline_time)
    assert torch.equal(candidate_quantity, baseline_quantity)
    assert baseline_outputs.keys() == candidate_outputs.keys()
    for name in baseline_outputs:
        assert torch.equal(candidate_outputs[name], baseline_outputs[name]), name


def test_candidate_reuses_one_bank_and_adds_only_two_feature_vectors() -> None:
    baseline, candidate = build_pair()

    assert candidate.memory_mode == TITAN_MEMORY_MODE_STATIC_HARD
    assert candidate.lmm is not None
    assert candidate.additional_parameter_count == 2 * candidate.hidden_dim
    assert [name for name, _ in candidate.named_parameters() if name == "lmm.mem"] == [
        "lmm.mem"
    ]
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    assert candidate_count - baseline_count == 2 * candidate.hidden_dim


def candidate_checkpoint_payload(
    candidate: CountAwareTitanMemoryFiLM,
) -> dict[str, object]:
    return {
        "backbone": MEMORY_FILM_BACKBONE,
        "variant": "count_only_log_regression",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "encoder_config": {
            **memory_film_metadata(candidate.hidden_dim),
            "time_head": candidate.time_head_contract(),
        },
        "model_state_dict": candidate.state_dict(),
    }


def test_metadata_and_checkpoint_identity_cannot_be_relabelled() -> None:
    _, candidate = build_pair()
    metadata = memory_film_metadata(candidate.hidden_dim)
    assert metadata["model_role"] == MEMORY_FILM_ROLE
    assert metadata["shared_intermediate_final_bank"] is True
    assert metadata["additional_parameter_count"] == 2 * candidate.hidden_dim

    payload = candidate_checkpoint_payload(candidate)
    assert validate_memory_film_checkpoint(payload, MEMORY_FILM_BACKBONE) is True
    with pytest.raises(ValueError, match="cannot be relabelled"):
        validate_memory_film_checkpoint(payload, "titantpp")

    missing_gain = candidate_checkpoint_payload(candidate)
    missing_gain["model_state_dict"] = dict(candidate.state_dict())
    missing_gain["model_state_dict"].pop("film_shift_gain")
    with pytest.raises(ValueError, match="state is missing or invalid"):
        validate_memory_film_checkpoint(missing_gain, MEMORY_FILM_BACKBONE)

    relabelled_b = candidate_checkpoint_payload(candidate)
    relabelled_b["backbone"] = "titantpp"
    with pytest.raises(ValueError, match="routing metadata mismatch"):
        validate_memory_film_checkpoint(relabelled_b, MEMORY_FILM_BACKBONE)


def test_b_checkpoint_loading_requires_all_inherited_tensors() -> None:
    baseline, candidate = build_pair()
    candidate.film_scale_gain.data.fill_(0.3)
    candidate.film_shift_gain.data.fill_(-0.2)

    candidate.load_hard_lmm_state_dict(baseline.state_dict())
    assert torch.count_nonzero(candidate.film_scale_gain).item() == 0
    assert torch.count_nonzero(candidate.film_shift_gain).item() == 0
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name

    incomplete = dict(baseline.state_dict())
    incomplete.pop("quantity_head.bias")
    with pytest.raises(RuntimeError, match="missing inherited parameters"):
        candidate.load_hard_lmm_state_dict(incomplete)


def test_open_modulation_changes_valid_states_and_both_gains_learn() -> None:
    _, candidate = build_pair()
    candidate.eval()
    candidate.film_scale_gain.data.fill_(0.25)
    candidate.film_shift_gain.data.fill_(-0.2)
    candidate.quantity_head.weight.data.fill_(0.1)
    dts, quantities, mask = sample_batch()

    features = candidate.continuous_features(dts, quantities, mask)
    assert candidate.encoder is not None
    first = candidate.encoder.input_proj(features)
    first = first + candidate.encoder._get_pos(
        first.size(1), first.device, first.dtype
    )
    first = first * mask.unsqueeze(-1)
    first = candidate.encoder.layers[0](first, mask=mask)
    modulated, diagnostics = candidate.apply_memory_film(first, mask)

    assert not torch.equal(modulated[mask], first[mask])
    assert torch.isfinite(modulated).all()
    assert torch.isfinite(diagnostics["scale_delta"]).all()
    assert torch.isfinite(diagnostics["shift"]).all()

    outputs = target_outputs(
        candidate, dts, mask, quantities, lambda_log_qty=1.0
    )
    outputs["quantity_train_loss"].mean().backward()
    for parameter in (candidate.film_scale_gain, candidate.film_shift_gain):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    assert candidate.lmm is not None and candidate.lmm.mem.grad is not None
    assert torch.isfinite(candidate.lmm.mem.grad).all()


def test_causal_prefix_does_not_depend_on_future_values() -> None:
    _, candidate = build_pair()
    candidate.eval()
    candidate.film_scale_gain.data.fill_(0.2)
    candidate.film_shift_gain.data.fill_(0.1)

    dts = torch.tensor([[0.0, 1.0, 2.0, 4.0, 8.0]])
    quantities = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[:, 3:] = torch.tensor([99.0, 999.0])
    changed_quantities[:, 3:] = torch.tensor([777.0, 7777.0])

    with torch.no_grad():
        reference = candidate.encode(dts, quantities, mask)
        changed = candidate.encode(changed_dts, changed_quantities, mask)

    torch.testing.assert_close(reference[:, :3], changed[:, :3], rtol=0.0, atol=1e-6)


def test_padding_is_zero_and_padding_values_are_ignored() -> None:
    _, candidate = build_pair()
    candidate.eval()
    candidate.film_scale_gain.data.fill_(0.2)
    candidate.film_shift_gain.data.fill_(-0.3)
    dts, quantities, mask = sample_batch()
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[1, 3:] = torch.tensor([1e6, 1e8])
    changed_quantities[1, 3:] = torch.tensor([1e5, 1e7])

    with torch.no_grad():
        reference = candidate.encode(dts, quantities, mask)
        changed = candidate.encode(changed_dts, changed_quantities, mask)

        features = candidate.continuous_features(dts, quantities, mask)
        assert candidate.encoder is not None
        first = candidate.encoder.input_proj(features)
        first = first + candidate.encoder._get_pos(
            first.size(1), first.device, first.dtype
        )
        first = first * mask.unsqueeze(-1)
        first = candidate.encoder.layers[0](first, mask=mask)
        _, diagnostics = candidate.apply_memory_film(first, mask)

    assert torch.equal(reference, changed)
    assert torch.count_nonzero(reference[~mask]).item() == 0
    for name in ("retrieval", "scale_delta", "shift"):
        assert torch.count_nonzero(diagnostics[name][~mask]).item() == 0
    assert bool((diagnostics["prototype_indices"][~mask] == -1).all())
    assert torch.count_nonzero(diagnostics["topk_similarity"][~mask]).item() == 0


def test_extreme_finite_inputs_have_finite_outputs_losses_and_gradients() -> None:
    _, candidate = build_pair()
    candidate.train()
    candidate.film_scale_gain.data.fill_(1.0)
    candidate.film_shift_gain.data.fill_(-1.0)
    dts = torch.tensor([[0.0, 1e-8, 1e4, 1e8]])
    quantities = torch.tensor([[0.0, 1.0, 1e6, 1e10]])
    mask = torch.ones_like(dts, dtype=torch.bool)

    outputs = target_outputs(
        candidate, dts, mask, quantities, lambda_log_qty=1.0
    )
    outputs["joint_loss"].mean().backward()

    assert all(
        bool(torch.isfinite(value).all())
        for value in outputs.values()
        if torch.is_tensor(value)
    )
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in candidate.parameters()
    )


def test_invalid_alternate_memory_modes_are_rejected() -> None:
    with pytest.raises(ValueError, match="fixes memory_mode"):
        CountAwareTitanMemoryFiLM(
            hidden_dim=16,
            train_log_mean=1.5,
            max_seq_len=8,
            memory_mode="none",
        )
