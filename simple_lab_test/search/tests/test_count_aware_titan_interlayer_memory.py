"""Local contracts for the inter-layer Hard-LMM backbone candidate."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from models.TPPs.CountAwareTitanInterLayerMemory import (
    CountAwareTitanInterLayerMemoryTPP,
)
from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


@pytest.fixture(autouse=True)
def single_thread() -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_kwargs() -> dict[str, object]:
    return {
        "hidden_dim": 16,
        "train_log_mean": 1.5,
        "max_seq_len": 8,
        "quantity_variant": "count_only_log_regression",
        "lambda_tail": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp",
    }


def build_pair(seed: int = 42) -> tuple[
    CountAwareTitanTPP,
    CountAwareTitanInterLayerMemoryTPP,
]:
    torch.manual_seed(seed)
    control = CountAwareTitanTPP(**model_kwargs())
    control_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    candidate = CountAwareTitanInterLayerMemoryTPP(**model_kwargs())
    assert torch.equal(torch.get_rng_state(), control_rng)
    return control, candidate


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0], [1.0, 3.0, 6.0, 0.0]],
        dtype=torch.float32,
    )
    quantities = torch.tensor(
        [[2.0, 5.0, 11.0, 23.0], [1.0, 4.0, 13.0, 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, True, True, True], [True, True, True, False]],
    )
    return dts, quantities, mask


def common_named_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name != "interlayer_alpha_raw"
    }


def test_zero_alpha_preserves_b_initialization_rng_and_parameter_state() -> None:
    control, candidate = build_pair()
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()

    assert candidate.memory_mode == "static_hard_lmm"
    assert candidate.interlayer_alpha.item() == 0.0
    assert set(candidate_state) == set(control_state) | {"interlayer_alpha_raw"}
    for name, tensor in control_state.items():
        assert torch.equal(candidate_state[name], tensor), name


def test_b_checkpoint_load_has_exactly_one_expected_missing_key() -> None:
    control, candidate = build_pair(seed=19)
    with torch.no_grad():
        candidate.interlayer_alpha_raw.fill_(0.7)

    incompatible = candidate.load_state_dict(control.state_dict(), strict=False)

    assert incompatible.missing_keys == ["interlayer_alpha_raw"]
    assert incompatible.unexpected_keys == []
    # Loading a B state does not silently choose the candidate behavior.  The
    # caller must retain/reset the explicit gate according to the run contract.
    assert candidate.interlayer_alpha_raw.item() == pytest.approx(0.7)


def test_zero_alpha_preserves_outputs_and_all_common_gradients() -> None:
    control, candidate = build_pair()
    control.train()
    candidate.train()
    dts, quantities, mask = batch()

    # Match dropout RNG exactly; the extra zero-gated memory read itself must
    # consume no randomness or perturb any common forward/gradient value.
    torch.manual_seed(902)
    control_outputs = target_outputs(
        control,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    torch.manual_seed(902)
    candidate_outputs = target_outputs(
        candidate,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    for name in (
        "joint_loss",
        "time_loss",
        "quantity_train_loss",
        "pred_qty",
        "true_qty",
    ):
        assert torch.equal(candidate_outputs[name], control_outputs[name]), name

    control_outputs["joint_loss"].sum().backward()
    candidate_outputs["joint_loss"].sum().backward()
    control_parameters = common_named_parameters(control)
    candidate_parameters = common_named_parameters(candidate)
    assert control_parameters.keys() == candidate_parameters.keys()
    for name, control_parameter in control_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if control_parameter.grad is None:
            assert candidate_parameter.grad is None, name
        else:
            assert torch.equal(candidate_parameter.grad, control_parameter.grad), name

    alpha_gradient = candidate.interlayer_alpha_raw.grad
    assert alpha_gradient is not None
    assert torch.isfinite(alpha_gradient)
    assert alpha_gradient.abs().item() > 0.0


def test_open_alpha_changes_hidden_and_prediction_and_keeps_final_read() -> None:
    _, candidate = build_pair(seed=11)
    candidate.eval()
    dts, quantities, mask = batch()
    with torch.no_grad():
        candidate.quantity_head.weight.fill_(0.1)
        closed_state = candidate.encode(dts, quantities, mask)
        closed_prediction = candidate.predict_quantity(closed_state)[1]

        candidate.interlayer_alpha_raw.fill_(math.atanh(0.5))
        calls = 0

        def count_final_read(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            nonlocal calls
            calls += 1

        handle = candidate.lmm.register_forward_hook(count_final_read)
        try:
            open_state = candidate.encode(dts, quantities, mask)
        finally:
            handle.remove()
        open_prediction = candidate.predict_quantity(open_state)[1]

    assert calls == 1
    assert not torch.equal(open_state[mask], closed_state[mask])
    assert not torch.equal(open_prediction[mask], closed_prediction[mask])
    assert torch.count_nonzero(open_state[~mask]) == 0


def test_open_candidate_is_causal_and_padding_invariant() -> None:
    _, candidate = build_pair(seed=31)
    candidate.eval()
    with torch.no_grad():
        candidate.interlayer_alpha_raw.fill_(math.atanh(0.5))

    dts = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    quantities = torch.tensor([[2.0, 4.0, 8.0, 16.0]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[:, 3] = 1000.0
    changed_quantities[:, 3] = 10000.0

    padded_dts = torch.tensor([[1.0, 2.0, 3.0, 99999.0]])
    padded_quantities = torch.tensor([[2.0, 4.0, 8.0, 99999.0]])
    padded_mask = torch.tensor([[True, True, True, False]])
    zero_padded_dts = padded_dts.clone()
    zero_padded_quantities = padded_quantities.clone()
    zero_padded_dts[:, 3] = 0.0
    zero_padded_quantities[:, 3] = 0.0

    with torch.no_grad():
        original = candidate.encode(dts, quantities, mask)
        future_changed = candidate.encode(changed_dts, changed_quantities, mask)
        padded = candidate.encode(padded_dts, padded_quantities, padded_mask)
        zero_padded = candidate.encode(
            zero_padded_dts,
            zero_padded_quantities,
            padded_mask,
        )

    assert torch.equal(original[:, :3], future_changed[:, :3])
    assert torch.equal(padded, zero_padded)
    assert torch.count_nonzero(padded[:, 3]) == 0


@pytest.mark.parametrize("magnitude", [0.0, 1e20])
def test_extreme_finite_inputs_keep_forward_and_gradients_finite(
    magnitude: float,
) -> None:
    _, candidate = build_pair(seed=71)
    candidate.train()
    with torch.no_grad():
        candidate.interlayer_alpha_raw.fill_(math.atanh(0.5))
    dts, quantities, mask = batch()
    dts = dts * magnitude
    quantities = quantities * magnitude

    encoded = candidate.encode(dts, quantities, mask)
    loss = encoded.square().mean()
    loss.backward()

    assert torch.isfinite(encoded).all()
    assert torch.isfinite(loss)
    assert candidate.interlayer_alpha_raw.grad is not None
    assert torch.isfinite(candidate.interlayer_alpha_raw.grad)
    for parameter in candidate.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("initial", [float("nan"), -1.0, 1.0])
def test_invalid_interlayer_alpha_initialization_is_rejected(initial: float) -> None:
    with pytest.raises(ValueError, match="interlayer_alpha_init"):
        CountAwareTitanInterLayerMemoryTPP(
            **model_kwargs(),
            interlayer_alpha_init=initial,
        )


def test_deepcopy_and_state_roundtrip_preserve_open_candidate() -> None:
    _, candidate = build_pair(seed=83)
    candidate.eval()
    with torch.no_grad():
        candidate.interlayer_alpha_raw.fill_(math.atanh(0.25))
    restored = copy.deepcopy(candidate)
    restored.load_state_dict(candidate.state_dict(), strict=True)
    restored.eval()
    dts, quantities, mask = batch()

    with torch.no_grad():
        expected = candidate.encode(dts, quantities, mask)
        actual = restored.encode(dts, quantities, mask)

    assert torch.equal(actual, expected)
    assert restored.interlayer_alpha.item() == pytest.approx(0.25)
