"""Current-source characterization, not a replay of historical metrics."""

import pytest
import torch

from phen.legacy import translate_legacy_state
from phen.registry import MODELS, build_model


HEADS = dict(time_scale=3.0, time_initial_location=0.2, time_initial_scale=0.8,
             time_observation_contract={"mode": "positive_integer_round_clamp_v1",
                                        "unit": "week", "top_code": None})


def legacy_model(name, length):
    pytest.importorskip("models.TPPs.CountAwareTPP", reason="Legacy references are not included in a source-only archive")
    kwargs = dict(hidden_dim=64, train_log_mean=1.2, **HEADS,
                  time_head_mode="heteroscedastic_lognormal_duration")
    if name == "phen":
        from models.TPPs.CountAwareTitanSharedHistoryMLP import CountAwareTitanSharedHistoryMLP
        return CountAwareTitanSharedHistoryMLP(max_seq_len=length, **kwargs)
    from models.TPPs.CountAwareTPP import CountAwareRMTPP, CountAwareTHP
    from models.TPPs.NeuralHawkesTPP import CountAwareNHP
    from models.TPPs.SelfAttentiveHawkesTPP import CountAwareSAHP
    from models.TPPs.CountAwareAdditionalTPP import CountAwareS2P2, CountAwareAttNHP
    return {"rmtpp": CountAwareRMTPP, "thp": CountAwareTHP, "nhp": CountAwareNHP,
            "sahp": CountAwareSAHP, "s2p2_matched_head": CountAwareS2P2,
            "attnhp_matched_head": CountAwareAttNHP}[name](**kwargs)


@pytest.mark.parametrize("name", MODELS)
def test_seeded_initialization_forward_and_gradient(name):
    length = 12
    torch.manual_seed(42)
    old = legacy_model(name, length)
    old_rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    new = build_model(name, hidden_dim=64, train_log_mean=1.2, max_seq_len=length, **HEADS)
    assert torch.equal(old_rng, torch.random.get_rng_state())
    old_state = translate_legacy_state(old.state_dict()) if name == "phen" else old.state_dict()
    assert set(new.state_dict()) == set(old_state)
    for key, value in new.state_dict().items():
        torch.testing.assert_close(value, old_state[key], rtol=0, atol=0)
    # Exercise trained, nonzero correction and quantity paths, not just their zero initialization.
    with torch.no_grad():
        old.quantity_head.weight.normal_(std=0.01)
        if name == "phen":
            for projection in old.multilag_detail.output_projections:
                projection.weight.normal_(std=0.01)
    state = translate_legacy_state(old.state_dict()) if name == "phen" else old.state_dict()
    new.load_state_dict(state, strict=True)
    old.eval()
    new.eval()
    dts = torch.arange(1, length + 1).float().repeat(2, 1)
    qty = torch.arange(length).float().repeat(2, 1)
    mask = torch.ones_like(dts, dtype=torch.bool)
    mask[1, -4:] = False
    old_hidden = old.encode(dts, qty, mask)[:, 4]
    new_hidden = new.encode(dts, qty, mask)[:, 4]
    torch.testing.assert_close(new_hidden, old_hidden, rtol=1e-6, atol=1e-6)
    targets = torch.tensor([3., 5.])
    for model, hidden in ((old, old_hidden), (new, new_hidden)):
        loss = (-model.log_observation_dt(hidden, targets)
                + model.quantity_outputs(hidden, targets)["train_loss"]).mean()
        loss.backward()
    torch.testing.assert_close(new.predict_quantity(new_hidden)[1], old.predict_quantity(old_hidden)[1])
    torch.testing.assert_close(new.predict_time_median(new_hidden), old.predict_time_median(old_hidden), rtol=1e-7, atol=1e-7)
    old_grads = {key: parameter.grad for key, parameter in old.named_parameters()}
    if name == "phen":
        old_grads = translate_legacy_state(old_grads)
    for key, parameter in new.named_parameters():
        reference = old_grads[key]
        assert (parameter.grad is None) == (reference is None), key
        if reference is not None:
            torch.testing.assert_close(parameter.grad, reference, rtol=2e-5, atol=2e-6, msg=key)


def test_history_gates_and_nonzero_correction_match_with_holes_and_resets():
    pytest.importorskip("models.TPPs.CountAwareTitanSharedHistoryMLP")
    from phen.models.components import HistoryCorrection
    from models.TPPs.CountAwareTitanSharedHistoryMLP import SharedHistoryCorrection
    torch.manual_seed(52)
    old, new = SharedHistoryCorrection(), HistoryCorrection()
    with torch.no_grad():
        for output in old.output_projections:
            output.weight.normal_(std=0.1)
    new.load_state_dict(old.state_dict())
    hidden = torch.randn(2, 150, 64, requires_grad=True)
    valid = torch.ones(2, 150, dtype=torch.bool)
    valid[0, :4] = False
    valid[1, 7:11] = False
    observed = valid.clone()
    observed[1, 65] = False
    actual = new(hidden, valid, observed=observed)
    expected = old(hidden, valid, memory_write_mask=observed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected.sum(), hidden, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.sum(), hidden)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_phen_stochastic_forward_gradient_and_optimizer_step():
    torch.manual_seed(62)
    old = legacy_model("phen", 140)
    new = build_model("phen", hidden_dim=64, train_log_mean=1.2, max_seq_len=140, **HEADS)
    with torch.no_grad():
        old.quantity_head.weight.normal_(std=0.01)
        for output in old.multilag_detail.output_projections:
            output.weight.normal_(std=0.01)
    new.load_state_dict(translate_legacy_state(old.state_dict()), strict=True)
    dts = torch.ones(2, 140)
    qty = torch.arange(140).float().repeat(2, 1)
    mask = torch.ones_like(dts, dtype=torch.bool)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001) for model in (old, new)]
    states = []
    for model, optimizer in zip((old, new), optimizers):
        model.train()
        torch.manual_seed(72)
        hidden = model.encode(dts, qty, mask)[:, -1]
        states.append(hidden.detach())
        loss = (-model.log_observation_dt(hidden, torch.tensor([2., 3.]))
                + model.quantity_outputs(hidden, torch.tensor([7., 9.]))["train_loss"]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    before = translate_legacy_state(old.state_dict())
    for key, tensor in new.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=2e-5, atol=2e-6, msg=key)
