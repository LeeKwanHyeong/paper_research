from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.value_conditioning import predict_value_for_marks
from models.Titan import TitanConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
from simple_lab_test.search.common.runner import build_run_paths


def make_model(
    mode: str,
    qty_mark_gradient_mode: str = "coupled",
    value_encoder_gradient_mode: str = "coupled",
    use_lmm: bool = False,
) -> TitanTPP:
    rmtpp_cfg = RMTPPConfig(
        num_marks=5,
        mark_emb_dim=8,
        scale_base=2.0,
        value_head_activation="identity",
        value_head_mode=mode,
        qty_mark_gradient_mode=qty_mark_gradient_mode,
        value_encoder_gradient_mode=value_encoder_gradient_mode,
        value_input_mode="residual",
        loss_mode="hybrid",
        train_loss_scope="target_only",
    )
    titan_cfg = TitanConfig(
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
        memory_mode="static_lmm" if use_lmm else "none",
        contextual_mem_size=0,
        persistent_mem_size=4 if use_lmm else 0,
        use_lmm=use_lmm,
        mem_size=8,
        mem_topk=2,
        use_causal=True,
        max_len=32,
    )
    return TitanTPP(rmtpp_cfg, titan_cfg)


def test_zero_initialized_experts_match_shared_head() -> None:
    torch.manual_seed(7)
    shared = make_model("shared")
    torch.manual_seed(7)
    expert = make_model("mark_conditioned_experts")

    expert_state = expert.state_dict()
    for name, value in shared.state_dict().items():
        assert name in expert_state
        assert torch.equal(value, expert_state[name]), name

    hidden = torch.randn(2, 3, 16)
    logits = torch.randn(2, 3, 4)
    shared_value = shared.predict_value(hidden)
    expert_values = expert.predict_value_by_mark(hidden)

    assert expert_values.shape == (2, 3, 4)
    assert torch.allclose(
        expert_values,
        shared_value.unsqueeze(-1).expand_as(expert_values),
    )
    assert torch.allclose(
        shared.expected_qty_from_logits(logits, shared_value),
        expert.expected_qty_from_logits(logits, expert_values),
    )


def test_mark_conditioned_nll_is_finite_and_has_expected_shape() -> None:
    torch.manual_seed(11)
    model = make_model("mark_conditioned_experts")
    marks = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2]], dtype=torch.long)
    dts = torch.tensor([[0.0, 1.0, 2.0, 1.0], [0.0, 3.0, 1.0, 2.0]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.3, 0.2]])
    mask = torch.ones_like(marks, dtype=torch.bool)

    out = model.nll(marks, dts, values=values, mask=mask)

    assert out["value_by_mark"].shape == (2, 3, 4)
    for name in ("nll", "nll_marker", "nll_time", "value_loss", "qty_loss", "total_loss"):
        assert torch.isfinite(out[name]), name


def test_residual_loss_routes_gradient_to_selected_experts() -> None:
    torch.manual_seed(13)
    model = make_model("mark_conditioned_experts")
    hidden = torch.randn(3, 16)
    target_marks = torch.tensor([0, 2, 2])
    target_values = torch.tensor([0.2, 0.4, 0.6])

    values_by_mark = model.predict_value_by_mark(hidden)
    selected = predict_value_for_marks(model, hidden, target_marks)
    assert torch.allclose(
        selected,
        values_by_mark.gather(-1, target_marks.unsqueeze(-1)).squeeze(-1),
    )

    F.huber_loss(selected, target_values).backward()
    grad = model.value_mark_delta_head.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[0]) > 0
    assert torch.count_nonzero(grad[2]) > 0
    assert torch.count_nonzero(grad[1]) == 0
    assert torch.count_nonzero(grad[3]) == 0


def test_explicit_marks_select_their_residual_experts() -> None:
    model = make_model("mark_conditioned_experts")
    with torch.no_grad():
        model.value_head.weight.zero_()
        model.value_head.bias.zero_()
        model.value_mark_delta_head.weight.zero_()
        model.value_mark_delta_head.bias.copy_(torch.tensor([0.0, 0.1, 0.2, 0.3]))

    hidden = torch.randn(2, 16)
    selected = predict_value_for_marks(model, hidden, torch.tensor([3, 1]))

    assert torch.allclose(selected, torch.tensor([0.3, 0.1]))


def test_appended_target_mark_does_not_change_previous_prediction() -> None:
    torch.manual_seed(17)
    model = make_model("mark_conditioned_experts").eval()
    marks = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    changed_marks = marks.clone()
    changed_marks[:, -1] = 0
    dts = torch.tensor([[0.0, 1.0, 2.0, 1.0]])

    with torch.no_grad():
        original_hidden = model.forward(marks, dts)[:, -2]
        changed_hidden = model.forward(changed_marks, dts)[:, -2]
        original_logits = model.mark_head(original_hidden)
        changed_logits = model.mark_head(changed_hidden)

    assert torch.allclose(original_hidden, changed_hidden, atol=1e-6)
    assert torch.allclose(original_logits, changed_logits, atol=1e-6)


def test_v3_uses_a_distinct_run_path(tmp_path) -> None:
    run_cfg = RunConfig(
        dataset_name="intermittent",
        dataset_kind="marked_target",
        model_name="titantpp",
        candidate_name="small_lmm",
        candidate=SimpleNamespace(name="small_lmm"),
        seed=42,
        epochs=1,
        scale_base=2.0,
        titan_profile="dataset_best",
    )
    shared_cfg = ExperimentConfig(base_dir=str(tmp_path), value_head_mode="shared")
    expert_cfg = ExperimentConfig(
        base_dir=str(tmp_path),
        value_head_mode="mark_conditioned_experts",
    )
    detached_cfg = ExperimentConfig(
        base_dir=str(tmp_path),
        value_head_mode="mark_conditioned_experts",
        qty_mark_gradient_mode="detached",
    )
    value_encoder_detached_cfg = ExperimentConfig(
        base_dir=str(tmp_path),
        value_head_mode="mark_conditioned_experts",
        qty_mark_gradient_mode="detached",
        value_encoder_gradient_mode="detached",
    )

    shared_path = build_run_paths(shared_cfg, run_cfg).run_dir
    expert_path = build_run_paths(expert_cfg, run_cfg).run_dir
    detached_path = build_run_paths(detached_cfg, run_cfg).run_dir
    value_encoder_detached_path = build_run_paths(
        value_encoder_detached_cfg,
        run_cfg,
    ).run_dir

    assert shared_path != expert_path
    assert expert_path != detached_path
    assert detached_path != value_encoder_detached_path
    assert "valuehead_mark_conditioned_experts" in str(expert_path)
    assert "qtymarkgrad_detached" in str(detached_path)
    assert "valueencgrad_detached" in str(value_encoder_detached_path)
    assert "qtymarkgrad_" not in str(expert_path)
    assert "valueencgrad_" not in str(detached_path)
    assert "valuehead_" not in str(shared_path)


def test_v3a_v3b_forward_and_loss_values_are_identical() -> None:
    torch.manual_seed(19)
    coupled = make_model("mark_conditioned_experts", "coupled")
    torch.manual_seed(19)
    detached = make_model("mark_conditioned_experts", "detached")

    detached_state = detached.state_dict()
    for name, value in coupled.state_dict().items():
        assert torch.equal(value, detached_state[name]), name

    marks = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2]], dtype=torch.long)
    dts = torch.tensor([[0.0, 1.0, 2.0, 1.0], [0.0, 3.0, 1.0, 2.0]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.3, 0.2]])
    mask = torch.ones_like(marks, dtype=torch.bool)

    coupled_out = coupled.nll(marks, dts, values=values, mask=mask)
    detached_out = detached.nll(marks, dts, values=values, mask=mask)

    for key in (
        "nll",
        "nll_marker",
        "nll_time",
        "value_loss",
        "qty_loss",
        "total_loss",
        "value_hat",
        "value_by_mark",
    ):
        assert torch.equal(coupled_out[key], detached_out[key]), key


def _module_grad_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.abs().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _upstream_grad_norm(model: TitanTPP) -> float:
    modules = [model.emb, model.encoder]
    if hasattr(model, "value_input_proj"):
        modules.append(model.value_input_proj)
    if hasattr(model, "lmm"):
        modules.append(model.lmm)
    return sum(_module_grad_norm(module) for module in modules)


def _quantity_gradient_model(
    mode: str,
    value_encoder_gradient_mode: str = "coupled",
    use_lmm: bool = False,
) -> tuple[TitanTPP, dict[str, torch.Tensor]]:
    torch.manual_seed(23)
    model = make_model(
        "mark_conditioned_experts",
        mode,
        value_encoder_gradient_mode,
        use_lmm,
    )
    marks = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2]], dtype=torch.long)
    dts = torch.tensor([[0.0, 1.0, 2.0, 1.0], [0.0, 3.0, 1.0, 2.0]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.3, 0.2]])
    mask = torch.ones_like(marks, dtype=torch.bool)
    return model, model.nll(marks, dts, values=values, mask=mask)


def test_detached_qty_loss_blocks_mark_head_but_keeps_value_gradients() -> None:
    coupled, coupled_out = _quantity_gradient_model("coupled")
    coupled.zero_grad(set_to_none=True)
    coupled_out["qty_loss"].backward()

    detached, detached_out = _quantity_gradient_model("detached")
    detached.zero_grad(set_to_none=True)
    detached_out["qty_loss"].backward()

    assert _module_grad_norm(coupled.mark_head) > 0.0
    assert _module_grad_norm(detached.mark_head) == 0.0
    assert _module_grad_norm(detached.value_head) > 0.0
    assert _module_grad_norm(detached.value_mark_delta_head) > 0.0
    assert _module_grad_norm(detached.encoder) > 0.0


def test_detached_full_loss_still_trains_mark_head() -> None:
    model, out = _quantity_gradient_model("detached")
    model.zero_grad(set_to_none=True)
    out["total_loss"].backward()

    assert _module_grad_norm(model.mark_head) > 0.0


def test_v3b_v3c_forward_and_loss_values_are_identical() -> None:
    torch.manual_seed(29)
    v3b = make_model("mark_conditioned_experts", "detached", "coupled")
    torch.manual_seed(29)
    v3c = make_model("mark_conditioned_experts", "detached", "detached")

    v3b_state = v3b.state_dict()
    v3c_state = v3c.state_dict()
    assert v3b_state.keys() == v3c_state.keys()
    for name, value in v3b_state.items():
        assert torch.equal(value, v3c_state[name]), name

    marks = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2]], dtype=torch.long)
    dts = torch.tensor([[0.0, 1.0, 2.0, 1.0], [0.0, 3.0, 1.0, 2.0]])
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.3, 0.2]])
    mask = torch.ones_like(marks, dtype=torch.bool)

    assert torch.equal(
        v3b.forward(marks, dts, values=values, mask=mask),
        v3c.forward(marks, dts, values=values, mask=mask),
    )
    v3b_out = v3b.nll(marks, dts, values=values, mask=mask)
    v3c_out = v3c.nll(marks, dts, values=values, mask=mask)
    for key in (
        "nll",
        "nll_marker",
        "nll_time",
        "value_loss",
        "qty_loss",
        "total_loss",
        "value_hat",
        "value_by_mark",
    ):
        assert torch.equal(v3b_out[key], v3c_out[key]), key


@pytest.mark.parametrize("loss_name", ["value_loss", "qty_loss"])
def test_v3c_auxiliary_losses_stop_at_value_encoder_boundary(loss_name: str) -> None:
    model, out = _quantity_gradient_model("detached", "detached")
    model.zero_grad(set_to_none=True)
    out[loss_name].backward()

    assert _module_grad_norm(model.value_head) > 0.0
    assert _module_grad_norm(model.value_mark_delta_head) > 0.0
    assert _module_grad_norm(model.mark_head) == 0.0
    assert _upstream_grad_norm(model) == 0.0


@pytest.mark.parametrize(
    ("loss_name", "head_name"),
    [("nll_marker", "mark_head"), ("nll_time", "v_t")],
)
def test_v3c_primary_losses_still_train_encoder(
    loss_name: str,
    head_name: str,
) -> None:
    model, out = _quantity_gradient_model("detached", "detached")
    model.zero_grad(set_to_none=True)
    out[loss_name].backward()

    assert _upstream_grad_norm(model) > 0.0
    assert _module_grad_norm(getattr(model, head_name)) > 0.0
    assert _module_grad_norm(model.value_head) == 0.0
    assert _module_grad_norm(model.value_mark_delta_head) == 0.0


def test_v3c_full_loss_trains_primary_and_value_branches() -> None:
    model, out = _quantity_gradient_model("detached", "detached")
    model.zero_grad(set_to_none=True)
    out["total_loss"].backward()

    assert _upstream_grad_norm(model) > 0.0
    assert _module_grad_norm(model.mark_head) > 0.0
    assert _module_grad_norm(model.v_t) > 0.0
    assert _module_grad_norm(model.value_head) > 0.0
    assert _module_grad_norm(model.value_mark_delta_head) > 0.0


def test_v3c_static_lmm_obeys_value_encoder_boundary() -> None:
    auxiliary_model, auxiliary_out = _quantity_gradient_model(
        "detached",
        "detached",
        use_lmm=True,
    )
    auxiliary_model.zero_grad(set_to_none=True)
    auxiliary_out["qty_loss"].backward()

    assert _module_grad_norm(auxiliary_model.value_head) > 0.0
    assert _module_grad_norm(auxiliary_model.lmm) == 0.0
    assert _upstream_grad_norm(auxiliary_model) == 0.0

    marker_model, marker_out = _quantity_gradient_model(
        "detached",
        "detached",
        use_lmm=True,
    )
    marker_model.zero_grad(set_to_none=True)
    marker_out["nll_marker"].backward()

    assert _module_grad_norm(marker_model.lmm) > 0.0
    assert _upstream_grad_norm(marker_model) > 0.0


@pytest.mark.parametrize(
    ("value_head_mode", "qty_mark_gradient_mode"),
    [("shared", "detached"), ("mark_conditioned_experts", "coupled")],
)
def test_v3c_rejects_incomplete_gradient_mode_combinations(
    value_head_mode: str,
    qty_mark_gradient_mode: str,
) -> None:
    with pytest.raises(ValueError, match="value_encoder_gradient_mode='detached' requires"):
        make_model(
            value_head_mode,
            qty_mark_gradient_mode,
            "detached",
        )
