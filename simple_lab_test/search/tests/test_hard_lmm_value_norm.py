"""Focused contracts for the value-norm-consistent Hard-LMM candidate."""

from __future__ import annotations

import copy
import io
import math

import pytest
import torch

from models.TPPs.CountAwareFactory import (
    build_count_aware_model,
    validate_checkpoint_route,
)
from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_STATIC_HARD,
)
from models.TPPs.CountAwareTitanValueNorm import (
    CountAwareTitanValueNormTPP,
    VALUE_NORM_ALPHA_KEY,
    VALUE_NORM_BACKBONE,
    VALUE_NORM_CONTRACT_ID,
    VALUE_NORM_EPSILON,
    VALUE_NORM_REQUIRED_STATE_KEYS,
    VALUE_NORM_ROLE,
    ValueNormHardLocalMemoryMatcher,
    validate_value_norm_checkpoint,
    value_norm_metadata,
)
from models.Titan.common.memory import HardLocalMemoryMatcher
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
    seed: int = 42,
    *,
    device: str = "cpu",
    hidden_dim: int = 16,
) -> tuple[CountAwareTitanTPP, CountAwareTitanValueNormTPP]:
    torch.manual_seed(seed)
    baseline = CountAwareTitanTPP(
        **model_kwargs(hidden_dim), memory_mode=TITAN_MEMORY_MODE_STATIC_HARD
    )
    expected_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    candidate = CountAwareTitanValueNormTPP(**model_kwargs(hidden_dim))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    return baseline.to(device), candidate.to(device)


def sample_batch(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(
            [[0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 3.0, 6.0, 0.0]],
            device=device,
        ),
        torch.tensor(
            [[2.0, 5.0, 9.0, 12.0, 20.0], [1.0, 7.0, 13.0, 21.0, 0.0]],
            device=device,
        ),
        torch.tensor(
            [[True] * 5, [True] * 4 + [False]], device=device
        ),
    )


def objective(
    model: CountAwareTitanTPP,
    dts: torch.Tensor,
    quantities: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return target_outputs(
        model, dts, mask, quantities, lambda_log_qty=1.0
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hidden_dim", [16, 64])
def test_same_seed_common_state_rng_and_zero_gate_output_identity(
    device: str, hidden_dim: int,
) -> None:
    baseline, candidate = build_pair(device=device, hidden_dim=hidden_dim)
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert set(candidate_state) == set(baseline_state) | {VALUE_NORM_ALPHA_KEY}
    assert candidate.value_norm_matcher.mem is candidate.lmm.mem
    assert candidate.value_norm_matcher.alpha.item() == 0.0
    for name, expected in baseline_state.items():
        assert torch.equal(candidate_state[name], expected), name

    baseline.train()
    candidate.train()
    outputs = []
    for model in (baseline, candidate):
        torch.manual_seed(917)
        outputs.append(objective(model, *sample_batch(device)))
    for name in outputs[0]:
        assert torch.equal(outputs[0][name], outputs[1][name]), name
    outputs[0]["joint_loss"].sum().backward()
    outputs[1]["joint_loss"].sum().backward()
    candidate_parameters = dict(candidate.named_parameters())
    for name, parameter in baseline.named_parameters():
        actual = candidate_parameters[name].grad
        if parameter.grad is None:
            assert actual is None, name
        else:
            assert torch.equal(parameter.grad, actual), name


def test_b_checkpoint_has_only_alpha_missing_and_explicit_load_is_strict() -> None:
    baseline, candidate = build_pair(seed=19)
    incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.missing_keys == [VALUE_NORM_ALPHA_KEY]
    assert incompatible.unexpected_keys == []

    candidate.value_norm_matcher.alpha_raw.data.fill_(0.8)
    before = copy.deepcopy(candidate.state_dict())
    baseline_state = baseline.state_dict()
    incomplete = dict(baseline_state)
    incomplete.pop("quantity_head.bias")
    wrong_dtype = dict(baseline_state)
    wrong_dtype["quantity_head.bias"] = wrong_dtype["quantity_head.bias"].double()
    wrong_shape = dict(baseline_state)
    wrong_shape["quantity_head.weight"] = wrong_shape[
        "quantity_head.weight"
    ][:, :-1]
    nonfinite = dict(baseline_state)
    nonfinite["lmm.mem"] = nonfinite["lmm.mem"].clone()
    nonfinite["lmm.mem"][0, 0, 0] = float("nan")
    unexpected = dict(baseline_state)
    unexpected["foreign.weight"] = torch.zeros(1)
    corruptions = (
        (incomplete, "missing inherited"),
        (wrong_dtype, "invalid tensor"),
        (wrong_shape, "invalid tensor"),
        (nonfinite, "invalid tensor"),
        (unexpected, "unexpected parameters"),
    )
    for corrupted, message in corruptions:
        with pytest.raises(RuntimeError, match=message):
            candidate.load_hard_lmm_state_dict(corrupted)
        for name, tensor in before.items():
            assert torch.equal(tensor, candidate.state_dict()[name]), name

    candidate.load_hard_lmm_state_dict(baseline.state_dict())
    assert candidate.value_norm_matcher.alpha.item() == 0.0
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name


def _crafted_matcher() -> tuple[ValueNormHardLocalMemoryMatcher, torch.Tensor]:
    torch.manual_seed(1)
    base = HardLocalMemoryMatcher(d_model=4, mem_size=8, topk=4)
    matcher = ValueNormHardLocalMemoryMatcher(base)
    assert matcher.mem is base.mem
    bank = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [0.25, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0, 0.0],
        ]
    ).unsqueeze(0)
    matcher.mem.data.copy_(bank)
    query = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    return matcher, query


def test_open_gate_changes_direction_without_changing_top4_or_similarity() -> None:
    matcher, query = _crafted_matcher()
    base_residual, base_trace = HardLocalMemoryMatcher.retrieve(matcher, query)
    component_base, normalized_residual, component_trace = (
        matcher.retrieve_components(query)
    )
    assert torch.equal(component_base, base_residual)
    assert torch.equal(component_trace["prototype_indices"], base_trace["prototype_indices"])
    assert torch.equal(component_trace["topk_similarity"], base_trace["topk_similarity"])
    selected = matcher.mem[0, component_trace["prototype_indices"][0, 0]]
    target_norm = torch.linalg.vector_norm(matcher.mem[0], dim=-1).mean()
    expected_selected = (
        selected
        / torch.linalg.vector_norm(selected, dim=-1, keepdim=True)
        * target_norm
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(expected_selected, dim=-1),
        target_norm.expand(expected_selected.size(0)),
    )
    torch.testing.assert_close(
        normalized_residual[0, 0], expected_selected.mean(dim=0)
    )

    correction = normalized_residual - base_residual
    determinant = (
        base_residual[..., 0] * correction[..., 1]
        - base_residual[..., 1] * correction[..., 0]
    )
    assert determinant.abs().item() > 1e-4

    matcher.alpha_raw.data.fill_(math.atanh(0.5))
    opened, opened_trace = matcher.retrieve(query)
    assert torch.equal(opened_trace["prototype_indices"], base_trace["prototype_indices"])
    assert torch.equal(opened_trace["topk_similarity"], base_trace["topk_similarity"])
    cosine = torch.nn.functional.cosine_similarity(opened, base_residual, dim=-1)
    assert cosine.item() < 0.9999


def test_detached_bank_reference_does_not_update_unselected_rows() -> None:
    matcher, query = _crafted_matcher()
    _, normalized_residual, trace = matcher.retrieve_components(query)
    selected_indices = set(trace["prototype_indices"].reshape(-1).tolist())
    unselected_indices = set(range(matcher.mem_size)) - selected_indices
    assert selected_indices and unselected_indices

    weights = normalized_residual.new_tensor([1.0, -2.0, 3.0, -4.0])
    (normalized_residual * weights).sum().backward()
    gradient = matcher.mem.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert any(torch.count_nonzero(gradient[0, index]) for index in selected_indices)
    assert all(
        torch.count_nonzero(gradient[0, index]) == 0
        for index in unselected_indices
    )


def test_zero_and_tiny_selected_rows_have_exact_zero_value_and_gradient() -> None:
    torch.manual_seed(2)
    base = HardLocalMemoryMatcher(d_model=4, mem_size=4, topk=4)
    matcher = ValueNormHardLocalMemoryMatcher(base)
    matcher.mem.data.copy_(
        torch.tensor(
            [[
                [0.0, 0.0, 0.0, 0.0],
                [VALUE_NORM_EPSILON / 10, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
            ]]
        )
    )
    query = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    _, _, trace = matcher.retrieve_components(query)
    selected = matcher.mem.expand(1, -1, -1).unsqueeze(1)
    selected = torch.gather(
        selected,
        2,
        trace["prototype_indices"].unsqueeze(-1).expand(-1, -1, -1, 4),
    )
    normalized = matcher._normalize_selected_values(selected, matcher.mem)
    selected_indices = trace["prototype_indices"][0, 0]
    invalid = torch.isin(selected_indices, torch.tensor([0, 1]))
    assert invalid.sum().item() == 2
    assert torch.count_nonzero(normalized[0, 0, invalid]) == 0

    weights = normalized.new_tensor([1.0, -2.0, 3.0, -4.0])
    (normalized * weights).sum().backward()
    assert matcher.mem.grad is not None
    assert torch.count_nonzero(matcher.mem.grad[0, :2]) == 0
    assert torch.count_nonzero(matcher.mem.grad[0, 2:]) > 0


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [
        (torch.float16, 5e3),
        (torch.bfloat16, 1e20),
        (torch.float32, 1e20),
    ],
)
def test_supported_norm_paths_remain_finite(
    dtype: torch.dtype, magnitude: float,
) -> None:
    matcher, query = _crafted_matcher()
    matcher = matcher.to(dtype=dtype)
    query = query.to(dtype=dtype)
    matcher.mem.data.mul_(magnitude)
    matcher.alpha_raw.data.fill_(math.atanh(0.75))
    residual, trace = matcher.retrieve(query)
    assert residual.dtype == dtype
    assert torch.isfinite(residual).all()
    assert torch.isfinite(trace["topk_similarity"]).all()
    residual.float().sum().backward()
    assert matcher.mem.grad is not None
    assert torch.isfinite(matcher.mem.grad).all()
    assert matcher.alpha_raw.grad is not None
    assert torch.isfinite(matcher.alpha_raw.grad)


def test_zero_gate_crafted_common_gradients_are_exact_and_alpha_can_open() -> None:
    candidate, query = _crafted_matcher()
    torch.manual_seed(3)
    baseline = HardLocalMemoryMatcher(d_model=4, mem_size=8, topk=4)
    baseline.mem.data.copy_(candidate.mem.detach())
    baseline_query = query.clone().requires_grad_()
    candidate_query = query.clone().requires_grad_()
    weights = query.new_tensor([1.0, -2.0, 3.0, -4.0])

    baseline_output = baseline(baseline_query)
    candidate_output = candidate(candidate_query)
    assert candidate.alpha.item() == 0.0
    assert torch.equal(candidate_output, baseline_output)
    (baseline_output * weights).sum().backward()
    (candidate_output * weights).sum().backward()

    assert torch.equal(candidate_query.grad, baseline_query.grad)
    assert torch.equal(candidate.mem.grad, baseline.mem.grad)
    alpha_gradient = candidate.alpha_raw.grad
    assert alpha_gradient is not None
    assert torch.isfinite(alpha_gradient)
    assert alpha_gradient.abs().item() > 0.0


def test_open_head_objective_gives_finite_nonzero_alpha_gradient() -> None:
    _, candidate = build_pair(seed=73)
    candidate.train()
    candidate.value_norm_matcher.alpha_raw.data.fill_(math.atanh(0.25))
    candidate.quantity_head.weight.data.fill_(0.1)
    outputs = objective(candidate, *sample_batch())
    outputs["joint_loss"].mean().backward()
    gradient = candidate.value_norm_matcher.alpha_raw.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert gradient.abs().item() > 0.0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in candidate.parameters()
    )


def test_open_candidate_is_causal_and_padding_and_target_invariant() -> None:
    _, candidate = build_pair(seed=91)
    candidate.eval()
    candidate.value_norm_matcher.alpha_raw.data.fill_(math.atanh(0.5))
    dts, quantities, mask = sample_batch()
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[:, 3:] = torch.tensor([99.0, 999.0])
    changed_quantities[:, 3:] = torch.tensor([777.0, 7777.0])
    with torch.no_grad():
        reference = candidate.encode(dts, quantities, mask)
        changed = candidate.encode(changed_dts, changed_quantities, mask)
        torch.testing.assert_close(reference[:, :3], changed[:, :3], rtol=0, atol=1e-6)

        padded_dts, padded_quantities = dts.clone(), quantities.clone()
        padded_dts[~mask] = 1e20
        padded_quantities[~mask] = 1e20
        assert torch.equal(
            reference, candidate.encode(padded_dts, padded_quantities, mask)
        )
        assert torch.count_nonzero(reference[~mask]) == 0

        expected = objective(candidate, dts, quantities, mask)
        target_dts, target_quantities = dts.clone(), quantities.clone()
        target_positions = mask.sum(dim=1) - 1
        target_dts[torch.arange(2), target_positions] = 333.0
        target_quantities[torch.arange(2), target_positions] = 77777.0
        actual = objective(candidate, target_dts, target_quantities, mask)
        assert torch.equal(expected["pred_qty"], actual["pred_qty"])
        assert not torch.equal(expected["true_qty"], actual["true_qty"])


@pytest.mark.parametrize("device", DEVICES)
def test_extreme_finite_values_and_checkpoint_optimizer_roundtrip(device: str) -> None:
    _, candidate = build_pair(seed=117, device=device)
    candidate.value_norm_matcher.alpha_raw.data.fill_(math.atanh(0.5))
    dts = torch.tensor([[0.0, 1e-8, 1e4, 1e20]], device=device)
    quantities = torch.tensor([[0.0, 1.0, 1e10, 1e20]], device=device)
    mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    candidate.train()
    torch.manual_seed(951)
    outputs = objective(candidate, dts, quantities, mask)
    outputs["joint_loss"].mean().backward()
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in candidate.parameters()
    )
    optimizer.step()

    buffer = io.BytesIO()
    torch.save(
        {"model": candidate.state_dict(), "optimizer": optimizer.state_dict()},
        buffer,
    )
    buffer.seek(0)
    saved = torch.load(buffer, map_location=device, weights_only=True)
    _, restored = build_pair(seed=999, device=device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer.load_state_dict(saved["optimizer"])
    for name, tensor in candidate.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name


def _payload(candidate: CountAwareTitanValueNormTPP) -> dict[str, object]:
    return {
        "backbone": VALUE_NORM_BACKBONE,
        "variant": "count_only_log_regression",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "encoder_config": {
            **value_norm_metadata(candidate.hidden_dim),
            "time_head": candidate.time_head_contract(),
        },
        "model_state_dict": candidate.state_dict(),
    }


def test_factory_metadata_and_checkpoint_routes_reject_relabels_and_foreign_keys() -> None:
    model, metadata = build_count_aware_model(
        VALUE_NORM_BACKBONE, **model_kwargs()
    )
    assert isinstance(model, CountAwareTitanValueNormTPP)
    assert model.contract_id == VALUE_NORM_CONTRACT_ID
    assert metadata["model_role"] == VALUE_NORM_ROLE
    assert metadata["routing_contract_id"] == VALUE_NORM_CONTRACT_ID
    assert metadata["value_norm_epsilon"] == 1e-8
    assert metadata["value_norm_reference_detached"] is True
    assert metadata["value_norm_norm_compute_dtype"] == (
        "fp32_for_fp16_bfloat16_fp32_and_fp64_for_fp64"
    )
    assert metadata["value_norm_norm_algorithm"] == "max_abs_scaled_l2"
    payload = _payload(model)
    assert set(payload["model_state_dict"]) == set(VALUE_NORM_REQUIRED_STATE_KEYS)
    assert validate_value_norm_checkpoint(payload, VALUE_NORM_BACKBONE)
    validate_checkpoint_route(payload, VALUE_NORM_BACKBONE)

    with pytest.raises(ValueError, match="cannot be relabelled"):
        validate_checkpoint_route(payload, "titantpp")
    relabelled = copy.deepcopy(payload)
    relabelled["backbone"] = "titantpp"
    with pytest.raises(ValueError, match="routing metadata mismatch"):
        validate_value_norm_checkpoint(relabelled, VALUE_NORM_BACKBONE)
    foreign = copy.deepcopy(payload)
    foreign["model_state_dict"]["interlayer_alpha_raw"] = torch.zeros(())
    with pytest.raises(ValueError, match="foreign candidate route"):
        validate_value_norm_checkpoint(foreign, VALUE_NORM_BACKBONE)
    baseline, _ = build_pair()
    missing = copy.deepcopy(payload)
    missing["model_state_dict"] = baseline.state_dict()
    with pytest.raises(ValueError, match="key population"):
        validate_value_norm_checkpoint(missing, VALUE_NORM_BACKBONE)
    b_model, b_metadata = build_count_aware_model("titantpp", **model_kwargs())
    b_payload = {
        "backbone": "titantpp",
        "variant": "count_only_log_regression",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "encoder_config": b_metadata,
        "model_state_dict": b_model.state_dict(),
    }
    with pytest.raises(ValueError, match="routing metadata mismatch"):
        validate_checkpoint_route(b_payload, VALUE_NORM_BACKBONE)
    integral_alpha = copy.deepcopy(payload)
    integral_alpha["model_state_dict"][VALUE_NORM_ALPHA_KEY] = torch.zeros(
        (), dtype=torch.int64
    )
    with pytest.raises(ValueError, match="missing or invalid"):
        validate_value_norm_checkpoint(integral_alpha, VALUE_NORM_BACKBONE)
    nonfinite_common = copy.deepcopy(payload)
    nonfinite_common["model_state_dict"]["quantity_head.bias"] = torch.tensor(
        [float("inf")]
    )
    with pytest.raises(ValueError, match="invalid tensor"):
        validate_value_norm_checkpoint(nonfinite_common, VALUE_NORM_BACKBONE)
    extra = copy.deepcopy(payload)
    extra["model_state_dict"]["unused.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="key population"):
        validate_value_norm_checkpoint(extra, VALUE_NORM_BACKBONE)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantity_variant": "count_only_lognormal"},
        {"lambda_tail": 0.1},
        {"time_head_mode": "scaled_exact_rmtpp"},
    ],
)
def test_factory_keeps_fixed_heads_loss_and_selector_contract(
    kwargs: dict[str, object],
) -> None:
    options = model_kwargs()
    options.update(kwargs)
    with pytest.raises(ValueError, match="Intermediate-memory candidates"):
        build_count_aware_model(VALUE_NORM_BACKBONE, **options)
