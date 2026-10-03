"""Synthetic CPU checks of proposed controls; not production qualification."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from design_reference import CorrectionReference, eligibility, nb_log_mass

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REGISTRY = json.loads((ROOT / "reports/titantpp_independent_evaluation_preparation_20261001_v1/evaluation_registry.json").read_text())
FROZEN = ROOT / REGISTRY["bundles"]["core"]["source_root"]
sys.path.insert(0, str(FROZEN))
from models.TPPs.CountAwareTitanCoreAblation import HistoryCorrection
from models.TPPs.CountAwareTitanMultiLagDetail import lag_source_indices

torch.set_num_threads(2)


def activate(model):
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for layer in model.output_projections:
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * .1)


def test_frozen_source_hashes():
    for rel in ["models/TPPs/CountAwareTitanCoreAblation.py", "models/TPPs/CountAwareTitanMultiLagDetail.py"]:
        assert hashlib.sha256((FROZEN / rel).read_bytes()).hexdigest() == REGISTRY["bundles"]["core"]["source_files"][rel]
    assert sys.modules[HistoryCorrection.__module__].__file__.startswith(str(FROZEN))


@pytest.mark.parametrize("mode", ["baseline", "current_only", "all_available"])
def test_equal_parameter_count_zero_residual_and_rng(mode):
    torch.manual_seed(11)
    state = torch.get_rng_state().clone()
    m = CorrectionReference(mode)
    assert torch.equal(state, torch.get_rng_state())
    assert sum(p.numel() for p in m.parameters()) == 6144
    h = torch.randn(2, 132, 64)
    valid = torch.ones(2, 132, dtype=torch.bool)
    assert torch.equal(m(h, valid), torch.zeros_like(h))


def test_reference_matches_frozen_nonzero_computation_and_masks():
    torch.manual_seed(29)
    base = HistoryCorrection(64, "mlp")
    ref = CorrectionReference()
    activate(ref)
    base.load_state_dict(ref.state_dict(), strict=True)
    h = torch.randn(2, 140, 64)
    valid = torch.ones(2, 140, dtype=torch.bool)
    valid[0, 2:5] = False
    write = valid.clone()
    write[1, 7] = False
    src, mask = eligibility(valid, write)
    actual_src, actual_mask = lag_source_indices(valid, memory_write_mask=write, mode="local")
    assert torch.equal(src, actual_src) and torch.equal(mask, actual_mask)
    assert torch.allclose(ref(h, valid, write), base(h, valid, memory_write_mask=write), atol=2e-7, rtol=2e-6)


def test_threshold_boundaries():
    valid = torch.ones(1, 130, dtype=torch.bool)
    _, mask = eligibility(valid)
    for b, t in enumerate([1, 2, 4, 8, 16, 32, 64, 128]):
        assert not mask[0, t - 1, b]
        assert mask[0, t, b]


def test_all_available_uses_actual_predecessor_for_newly_enabled_branches():
    valid = torch.tensor([[True, False, True, True]])
    source, mask = eligibility(valid, all_available=True)
    assert not mask[0, 0].any() and not mask[0, 1].any()
    assert mask[0, 3].all() and source[0, 3].tolist() == [2] * 8
    frozen_source, _ = lag_source_indices(valid, mode="local")
    assert frozen_source[0, 3, 7] == 0  # simply unmasking these indices would be wrong


def test_withheld_event_resets_local_availability():
    valid = torch.tensor([[True, True, True, False, True, True]])
    write = torch.tensor([[True, True, False, False, True, True]])
    source, mask = eligibility(valid, write, all_available=True)
    assert not mask[0, 2:5].any()
    assert mask[0, 5].all() and source[0, 5].tolist() == [4] * 8


@pytest.mark.parametrize("mode", ["baseline", "current_only", "all_available"])
def test_future_and_unobserved_values_do_not_change_prefix(mode):
    m = CorrectionReference(mode)
    activate(m)
    h = torch.randn(1, 140, 64)
    v = torch.ones(1, 140, dtype=torch.bool)
    w = v.clone()
    w[0, 2] = False
    old = m(h, v, w)
    modified = h.clone()
    modified[:, 80:] += 100
    modified[:, 2] = float("nan")
    assert torch.equal(old[:, :80], m(modified, v, w)[:, :80])


def test_current_only_has_no_direct_predecessor_dependence():
    m = CorrectionReference("current_only")
    activate(m)
    h = torch.randn(1, 4, 64)
    valid = torch.ones(1, 4, dtype=torch.bool)
    old = m(h, valid)[:, 3].clone()
    h[:, 2] += 5
    assert torch.equal(old, m(h, valid)[:, 3])


def test_all_available_reduces_to_baseline_after_all_thresholds():
    base = CorrectionReference()
    activate(base)
    full = CorrectionReference("all_available")
    full.load_state_dict(base.state_dict())
    h = torch.randn(1, 140, 64)
    valid = torch.ones(1, 140, dtype=torch.bool)
    assert torch.equal(base(h, valid)[:, 128:], full(h, valid)[:, 128:])


@pytest.mark.parametrize("mode", ["current_only", "all_available"])
def test_zero_initialization_has_nonzero_output_projection_gradient(mode):
    m = CorrectionReference(mode)
    h = torch.randn(1, 140, 64)
    valid = torch.ones(1, 140, dtype=torch.bool)
    m(h, valid).square().sum()  # no optimizer and no scientific updates
    m(h, valid).sum().backward()
    assert any(p.weight.grad.abs().sum() > 0 for p in m.output_projections)
    assert all(torch.count_nonzero(p.weight.grad) == 0 for p in m.input_projections)


def test_shifted_nb_matches_torch_probability_and_analytic_mean():
    mu = torch.tensor([.01, 1., 5., 100.], dtype=torch.float64)
    alpha = torch.tensor([.1, 1., .3, 2.], dtype=torch.float64)
    raw_q = torch.tensor([1., 2., 9., 200.], dtype=torch.float64)
    dist = torch.distributions.NegativeBinomial(total_count=1 / alpha, logits=torch.log(alpha * mu))
    assert torch.allclose(nb_log_mass(raw_q - 1, mu, alpha), dist.log_prob(raw_q - 1), atol=1e-11)
    assert torch.allclose(1 + dist.mean, 1 + mu)


def test_topcode_tail_uses_shifted_support_not_point_mass():
    # alpha=1 gives a geometric NB; survival is available in closed form.
    mu = torch.tensor(15., dtype=torch.float64)
    alpha = torch.ones((), dtype=torch.float64)
    k = torch.arange(29, dtype=torch.float64)
    p_below_30 = nb_log_mass(k, mu, alpha).exp().sum()  # D=1,...,29
    p_at_or_above_30 = (mu / (1 + mu)) ** 29  # D>=30 iff NB>=29
    assert torch.allclose(p_below_30 + p_at_or_above_30, torch.ones(()).double(), atol=1e-12)
    assert p_at_or_above_30 > nb_log_mass(torch.tensor(29.), mu, alpha).exp()


def test_negative_binomial_rejects_noninteger_or_nonpositive_target():
    for k in [-1., .5]:
        with pytest.raises(ValueError):
            nb_log_mass(torch.tensor(k), torch.tensor(1.), torch.tensor(1.))
