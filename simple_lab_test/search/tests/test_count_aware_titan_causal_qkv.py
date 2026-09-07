"""Contracts for the Hard-LMM causal event-Q/K/V backbone candidate."""

from __future__ import annotations

import copy
import io

import pytest
import torch

from models.TPPs.CountAwareTitanCausalQKV import (
    CAUSAL_QKV_BACKBONE,
    CAUSAL_QKV_CONTRACT_ID,
    CAUSAL_QKV_KERNEL_KEYS,
    CAUSAL_QKV_ROLE,
    CausalQKVMemoryAttention,
    CountAwareTitanCausalQKVTPP,
    causal_qkv_metadata,
    validate_causal_qkv_checkpoint,
)
from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_STATIC_HARD,
)
from models.Titan.common.memory import MemoryAttention
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def single_thread() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_kwargs(hidden_dim: int = 16) -> dict[str, object]:
    return {
        "hidden_dim": hidden_dim,
        "train_log_mean": 1.5,
        "max_seq_len": 10,
        "quantity_variant": "count_only_log_regression",
        "lambda_tail": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp",
    }


def build_pair(
    *,
    seed: int = 42,
    device: str = "cpu",
    hidden_dim: int = 16,
) -> tuple[CountAwareTitanTPP, CountAwareTitanCausalQKVTPP]:
    torch.manual_seed(seed)
    baseline = CountAwareTitanTPP(
        **model_kwargs(hidden_dim),
        memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
    )
    random_state = torch.get_rng_state()
    torch.manual_seed(seed)
    candidate = CountAwareTitanCausalQKVTPP(**model_kwargs(hidden_dim))
    assert torch.equal(torch.get_rng_state(), random_state)
    return baseline.to(device), candidate.to(device)


def sample_batch(
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor(
        [
            [0.0, 1.0, 2.0, 4.0, 8.0],
            [0.0, 0.5, 3.0, 6.0, 0.0],
        ],
        device=device,
    )
    quantities = torch.tensor(
        [
            [2.0, 5.0, 9.0, 12.0, 20.0],
            [1.0, 7.0, 13.0, 21.0, 0.0],
        ],
        device=device,
    )
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, True, False],
        ],
        device=device,
    )
    return dts, quantities, mask


def new_parameters(
    candidate: CountAwareTitanCausalQKVTPP,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    attention = candidate.causal_qkv_attention
    return (
        attention.causal_q_kernel,
        attention.causal_k_kernel,
        attention.causal_v_kernel,
    )


def set_random_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_initialization_preserves_rng_state_outputs_and_common_gradients(
    device: str,
) -> None:
    baseline, candidate = build_pair(device=device)
    baseline.train()
    candidate.train()
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert set(candidate_state) == set(baseline_state) | set(CAUSAL_QKV_KERNEL_KEYS)
    for name, tensor in baseline_state.items():
        assert torch.equal(candidate_state[name], tensor), name
    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in new_parameters(candidate))

    dts, quantities, mask = sample_batch(device)
    set_random_seed(719, device)
    baseline_outputs = target_outputs(
        baseline, dts, mask, quantities, lambda_log_qty=1.0
    )
    set_random_seed(719, device)
    candidate_outputs = target_outputs(
        candidate, dts, mask, quantities, lambda_log_qty=1.0
    )
    assert baseline_outputs.keys() == candidate_outputs.keys()
    for name in baseline_outputs:
        assert torch.equal(candidate_outputs[name], baseline_outputs[name]), name

    baseline_outputs["joint_loss"].sum().backward()
    candidate_outputs["joint_loss"].sum().backward()
    baseline_parameters = dict(baseline.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    for name, baseline_parameter in baseline_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if baseline_parameter.grad is None:
            assert candidate_parameter.grad is None, name
        else:
            assert torch.equal(candidate_parameter.grad, baseline_parameter.grad), name


def test_candidate_adds_exactly_three_depthwise_kernel3_parameters() -> None:
    baseline, candidate = build_pair(hidden_dim=64)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())

    assert candidate_count - baseline_count == 576
    assert candidate.additional_parameter_count == 576
    assert isinstance(candidate.encoder.layers[0].attn, CausalQKVMemoryAttention)
    assert type(candidate.encoder.layers[1].attn) is MemoryAttention
    assert [tuple(parameter.shape) for parameter in new_parameters(candidate)] == [
        (3, 64),
        (3, 64),
        (3, 64),
    ]


def test_depthwise_formula_is_channelwise_and_strictly_left_looking() -> None:
    values = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]]])
    kernel = torch.tensor([[1.0, 0.1], [10.0, 1.0], [100.0, 10.0]])

    actual = CausalQKVMemoryAttention.causal_depthwise_residual(values, kernel)
    expected = torch.tensor(
        [[[1.0, 1.0], [12.0, 12.0], [124.0, 124.0], [248.0, 248.0]]]
    )
    assert torch.equal(actual, expected)

    changed = values.clone()
    changed[:, 3] = 9999.0
    changed_output = CausalQKVMemoryAttention.causal_depthwise_residual(
        changed, kernel
    )
    assert torch.equal(actual[:, :3], changed_output[:, :3])


@pytest.mark.parametrize("device", DEVICES)
def test_all_kernel_rows_receive_finite_nonzero_gradient(device: str) -> None:
    _, candidate = build_pair(seed=57, device=device)
    candidate.train()
    dts, quantities, mask = sample_batch(device)

    outputs = target_outputs(
        candidate, dts, mask, quantities, lambda_log_qty=1.0
    )
    outputs["joint_loss"].sum().backward()

    for parameter in new_parameters(candidate):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        # Rows 1 and 2 are the causal lag-1 and lag-2 mechanisms.
        assert bool((parameter.grad.abs().sum(dim=1) > 0.0).all())


def test_learned_route_changes_both_encoder_block_states_and_predictions() -> None:
    _, candidate = build_pair(seed=73)
    candidate.eval()
    dts, quantities, mask = sample_batch()
    candidate.quantity_head.weight.data.fill_(0.1)

    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[name] = output.detach().clone()

        return hook

    handles = [
        candidate.encoder.layers[0].register_forward_hook(capture("h1")),
        candidate.encoder.layers[1].register_forward_hook(capture("h2")),
    ]
    try:
        with torch.no_grad():
            closed = target_outputs(
                candidate, dts, mask, quantities, lambda_log_qty=1.0
            )
            closed_h1, closed_h2 = captured["h1"], captured["h2"]
            for index, parameter in enumerate(new_parameters(candidate), start=1):
                parameter.copy_(
                    torch.linspace(
                        -0.03 * index,
                        0.03 * index,
                        parameter.numel(),
                    ).reshape_as(parameter)
                )
            opened = target_outputs(
                candidate, dts, mask, quantities, lambda_log_qty=1.0
            )
            opened_h1, opened_h2 = captured["h1"], captured["h2"]
    finally:
        for handle in handles:
            handle.remove()

    assert not torch.equal(opened_h1[mask], closed_h1[mask])
    assert not torch.equal(opened_h2[mask], closed_h2[mask])
    assert not torch.equal(opened["pred_qty"], closed["pred_qty"])
    assert torch.count_nonzero(opened_h1[~mask]).item() == 0
    assert torch.count_nonzero(opened_h2[~mask]).item() == 0


def test_open_route_preserves_full_prefix_causality_and_padding_contract() -> None:
    _, candidate = build_pair(seed=91)
    candidate.eval()
    for parameter in new_parameters(candidate):
        parameter.data.fill_(0.025)

    dts = torch.tensor([[0.0, 1.0, 2.0, 4.0, 8.0]])
    quantities = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[:, 3:] = torch.tensor([99.0, 999.0])
    changed_quantities[:, 3:] = torch.tensor([777.0, 7777.0])

    padded_mask = torch.tensor([[True, True, True, False, False]])
    padded_dts = dts.clone()
    padded_quantities = quantities.clone()
    altered_padding_dts = padded_dts.clone()
    altered_padding_quantities = padded_quantities.clone()
    altered_padding_dts[:, 3:] = torch.tensor([1e7, 1e10])
    altered_padding_quantities[:, 3:] = torch.tensor([1e8, 1e12])

    with torch.no_grad():
        reference = candidate.encode(dts, quantities, mask)
        future_changed = candidate.encode(
            changed_dts, changed_quantities, mask
        )
        padded = candidate.encode(padded_dts, padded_quantities, padded_mask)
        padding_changed = candidate.encode(
            altered_padding_dts,
            altered_padding_quantities,
            padded_mask,
        )

    torch.testing.assert_close(
        reference[:, :3], future_changed[:, :3], rtol=0.0, atol=1e-6
    )
    assert torch.equal(padded, padding_changed)
    assert torch.count_nonzero(padded[:, 3:]).item() == 0


def checkpoint_payload(
    candidate: CountAwareTitanCausalQKVTPP,
    *,
    include_state: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "backbone": CAUSAL_QKV_BACKBONE,
        "variant": "count_only_log_regression",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "encoder_config": {
            **causal_qkv_metadata(candidate.hidden_dim),
            "time_head": candidate.time_head_contract(),
        },
    }
    if include_state:
        payload["model_state_dict"] = candidate.state_dict()
    else:
        payload.update(
            {
                "best_epoch": 17,
                "checkpoint_state_sha256": "a" * 64,
                "checkpoint_file_sha256": "b" * 64,
            }
        )
    return payload


def test_metadata_state_and_summary_identity_cannot_be_relabelled() -> None:
    _, candidate = build_pair()
    metadata = causal_qkv_metadata(candidate.hidden_dim)
    assert CAUSAL_QKV_CONTRACT_ID == "hard_lmm_causal_qkv_v1"
    assert CAUSAL_QKV_ROLE == "hard_lmm_causal_qkv_candidate"
    assert metadata["causal_qkv_scope"] == "event_qkv_only"
    assert metadata["persistent_key_value_adaptation"] is False
    assert metadata["additional_parameter_count"] == 9 * candidate.hidden_dim

    payload = checkpoint_payload(candidate)
    assert validate_causal_qkv_checkpoint(payload, CAUSAL_QKV_BACKBONE) is True
    with pytest.raises(ValueError, match="cannot be relabelled"):
        validate_causal_qkv_checkpoint(payload, "titantpp")

    missing_lag = copy.deepcopy(payload)
    missing_lag["model_state_dict"].pop(CAUSAL_QKV_KERNEL_KEYS[-1])
    with pytest.raises(ValueError, match="kernels are missing or invalid"):
        validate_causal_qkv_checkpoint(missing_lag, CAUSAL_QKV_BACKBONE)

    summary = checkpoint_payload(candidate, include_state=False)
    assert validate_causal_qkv_checkpoint(summary, CAUSAL_QKV_BACKBONE) is True
    summary["checkpoint_state_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="canonical state digest"):
        validate_causal_qkv_checkpoint(summary, CAUSAL_QKV_BACKBONE)


def test_b_checkpoint_loader_accepts_only_complete_inherited_state() -> None:
    baseline, candidate = build_pair(seed=103)
    for parameter in new_parameters(candidate):
        parameter.data.fill_(0.5)

    candidate.load_hard_lmm_state_dict(baseline.state_dict())
    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in new_parameters(candidate))
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(candidate.state_dict()[name], tensor), name

    incomplete = dict(baseline.state_dict())
    incomplete.pop("quantity_head.bias")
    with pytest.raises(RuntimeError, match="missing inherited parameters"):
        candidate.load_hard_lmm_state_dict(incomplete)


def test_finite_extremes_and_model_optimizer_save_restore() -> None:
    _, candidate = build_pair(seed=117)
    candidate.train()
    dts = torch.tensor([[0.0, 1e-8, 1e4, 1e8]])
    quantities = torch.tensor([[0.0, 1.0, 1e6, 1e10]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)

    outputs = target_outputs(
        candidate, dts, mask, quantities, lambda_log_qty=1.0
    )
    loss = outputs["joint_loss"].mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert all(
        torch.isfinite(value).all()
        for value in outputs.values()
        if torch.is_tensor(value)
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in candidate.parameters()
    )

    buffer = io.BytesIO()
    torch.save(
        {
            "model_state_dict": candidate.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        buffer,
    )
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    _, restored = build_pair(seed=999)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored.load_state_dict(saved["model_state_dict"], strict=True)
    restored_optimizer.load_state_dict(saved["optimizer_state_dict"])

    for name, tensor in candidate.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor), name
    original_optimizer = optimizer.state_dict()
    reloaded_optimizer = restored_optimizer.state_dict()
    assert original_optimizer["param_groups"] == reloaded_optimizer["param_groups"]
    assert original_optimizer["state"].keys() == reloaded_optimizer["state"].keys()
    for parameter_id, state in original_optimizer["state"].items():
        for name, value in state.items():
            restored_value = reloaded_optimizer["state"][parameter_id][name]
            if torch.is_tensor(value):
                assert torch.equal(restored_value, value)
            else:
                assert restored_value == value
