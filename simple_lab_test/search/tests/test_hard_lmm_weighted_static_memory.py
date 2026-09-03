"""Single-change retrieval, legacy compatibility, and causal model contracts."""

import io
import os

import pytest
import polars as pl
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.Titan.common.memory import (
    HardLocalMemoryMatcher,
    LMM,
    SimilarityWeightedLocalMemoryMatcher,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    MODEL_ROLE_WEIGHTED_STATIC,
    VARIANT,
    validate_model_role_contract,
)
from paper.scripts.count_aware_tpp_backbone.core import load_train_validation_frame, target_outputs

CANDIDATE = "titantpp_weighted_static_memory"
DEVICE = os.environ.get("WEIGHTED_STATIC_TEST_DEVICE", "cpu")


def build(backbone=CANDIDATE):
    torch.manual_seed(42)
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=16, train_log_mean=1.5, max_seq_len=8,
        quantity_variant=VARIANT, lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
    )
    return model.to(DEVICE), metadata


def manual_retrieval(x, bank, k, weighted):
    bank = bank.expand(x.size(0), -1, -1)
    scores, indices = (F.normalize(x, dim=-1) @ F.normalize(bank, dim=-1)
                       .transpose(-1, -2)).topk(k, dim=-1)
    selected = bank.unsqueeze(1).expand(-1, x.size(1), -1, -1).gather(
        2, indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))
    residual = (selected * scores.softmax(-1).unsqueeze(-1)).sum(2) if weighted else selected.mean(2)
    return residual, indices


@pytest.mark.parametrize("weighted", [False, True])
def test_exact_formula_selection_and_gradients(weighted):
    torch.manual_seed(7)
    cls = SimilarityWeightedLocalMemoryMatcher if weighted else HardLocalMemoryMatcher
    module = cls(8, 12, 4).to(DEVICE)
    x = torch.randn(2, 5, 8, device=DEVICE, requires_grad=True)
    residual, trace = module.retrieve(x)
    expected, indices = manual_retrieval(x, module.mem, 4, weighted)
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
    assert torch.equal(trace["prototype_indices"], indices)
    actual_grads = torch.autograd.grad((x + residual).square().sum(), (x, module.mem), retain_graph=True)
    reference_grads = torch.autograd.grad((x + expected).square().sum(), (x, module.mem))
    for actual, reference in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert torch.isfinite(actual).all()


def test_same_membership_different_scores_changes_only_weighted_residual():
    bank = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]], device=DEVICE)
    x = torch.tensor([[[1., .1], [.1, 1.]]], device=DEVICE, requires_grad=True)
    hard = HardLocalMemoryMatcher(2, 4, 4).to(DEVICE)
    weighted = SimilarityWeightedLocalMemoryMatcher(2, 4, 4).to(DEVICE)
    a, ta = hard.retrieve(x, bank)
    b, tb = weighted.retrieve(x, bank)
    assert torch.equal(ta["prototype_indices"], tb["prototype_indices"])
    assert torch.equal(a[:, 0], a[:, 1])
    assert not torch.allclose(b[:, 0], b[:, 1])
    b.square().sum().backward()
    assert x.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("topk,mem_size", [(0, 4), (4, 0), (1, 4), (4, 4)])
def test_empty_singleton_and_equal_similarity(topk, mem_size):
    module = SimilarityWeightedLocalMemoryMatcher(8, mem_size, topk).to(DEVICE)
    x = torch.zeros(2, 3, 8, device=DEVICE)
    residual, trace = module.retrieve(x)
    hard = HardLocalMemoryMatcher(8, mem_size, topk).to(DEVICE)
    hard.load_state_dict(module.state_dict())
    assert torch.equal(residual, hard.retrieve(x)[0])
    assert torch.isfinite(residual).all()
    if not topk or not mem_size:
        assert trace["topk_similarity"].size(-1) == 0
        assert torch.equal(module(x), x)


def test_external_banks_and_extreme_finite_values():
    torch.manual_seed(19)
    module = SimilarityWeightedLocalMemoryMatcher(8, 8, 4).to(DEVICE)
    x = (torch.randn(2, 3, 8, device=DEVICE) * 1e10).requires_grad_()
    bank = torch.randn(2, 9, 8, device=DEVICE, requires_grad=True)
    residual, _ = module.retrieve(x, bank)
    expected, _ = manual_retrieval(x, bank, 4, True)
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
    residual.square().mean().backward()
    assert torch.isfinite(residual).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(bank.grad).all()


def test_no_new_parameters_identical_initialization_and_legacy_metadata():
    old, old_meta = build("titantpp")
    new, new_meta = build()
    assert LMM is HardLocalMemoryMatcher
    assert type(old.lmm) is HardLocalMemoryMatcher
    assert type(new.lmm) is SimilarityWeightedLocalMemoryMatcher
    assert old.state_dict().keys() == new.state_dict().keys()
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key], new.state_dict()[key]), key
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    assert all(p.requires_grad for p in new.parameters())
    assert "static_retrieval_contract_id" not in old_meta
    assert old_meta["backbone_contract_id"] == "B0"
    assert new_meta["backbone_contract_id"] == "W0"
    assert new_meta["static_retrieval_temperature"] == 1.0
    for key in ("persistent_mem_size", "lmm_mem_size", "lmm_topk", "n_layers", "n_heads", "d_model"):
        assert old_meta[key] == new_meta[key]


def batch():
    dts = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 0.]], device=DEVICE)
    qty = torch.tensor([[1., 5., 9., 20.], [2., 6., 12., 0.]], device=DEVICE)
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]], device=DEVICE)
    return dts, qty, mask


def test_target_future_padding_and_series_isolation():
    model, _ = build()
    model.eval()
    with torch.no_grad():
        model.quantity_head.weight.normal_(0, .1)
    dts, qty, mask = batch()
    before = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    changed = qty.clone()
    changed[0, 3], changed[1, 2], changed[1, 3] = 90000., 80000., 70000.
    changed_dts = dts.clone()
    changed_dts[0, 3], changed_dts[1, 2], changed_dts[1, 3] = 60000., 50000., 40000.
    after = target_outputs(model, changed_dts, mask, changed, lambda_log_qty=1.)
    torch.testing.assert_close(before["pred_qty"], after["pred_qty"], rtol=0, atol=0)
    states = model.encode(dts, qty, mask)
    changed_states = model.encode(changed_dts, changed, mask)
    torch.testing.assert_close(states[:, :2], changed_states[:, :2], rtol=0, atol=0)
    for row in range(2):
        isolated = model.encode(dts[row:row+1], qty[row:row+1], mask[row:row+1])
        torch.testing.assert_close(states[row:row+1], isolated, rtol=1e-5, atol=1e-6)
    assert torch.equal(states[1, 3], torch.zeros_like(states[1, 3]))
    model.encode(changed_dts * 2, changed * 2, mask)
    torch.testing.assert_close(states, model.encode(dts, qty, mask), rtol=0, atol=0)


def test_finite_full_model_training_checkpoint_and_retrieval_effect():
    model, metadata = build()
    dts, qty, mask = batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        output = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
        assert all(torch.isfinite(value).all() for value in output.values())
        output["joint_loss"].mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.lmm.mem.grad.abs().sum() > 0
        optimizer.step()
    model.eval()
    original = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    buffer = io.BytesIO()
    torch.save({"backbone": CANDIDATE, "metadata": metadata, "state": model.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location=DEVICE, weights_only=True)
    restored, _ = build(saved["backbone"])
    restored.load_state_dict(saved["state"])
    restored.eval()
    replay = target_outputs(restored, dts, mask, qty, lambda_log_qty=1.)
    for key in original:
        torch.testing.assert_close(original[key], replay[key], rtol=0, atol=0)
    hard, _ = build("titantpp")
    hard.load_state_dict(saved["state"])
    hard.eval()
    assert not torch.allclose(model.encode(dts, qty, mask), hard.encode(dts, qty, mask), rtol=1e-7, atol=1e-8)


def test_experiment_role_is_strict():
    kwargs = dict(model_role=MODEL_ROLE_WEIGHTED_STATIC, backbones=(CANDIDATE,),
                  quantity_variants=(VARIANT,), time_head_mode="legacy_clamped_rmtpp", lambda_tail=0.)
    validate_model_role_contract(**kwargs)
    for update in ({"backbones": ("titantpp",)}, {"lambda_tail": .1},
                   {"quantity_variants": ("tail_shared",)}, {"time_head_mode": "scaled_exact"}):
        with pytest.raises(ValueError):
            validate_model_role_contract(**(kwargs | update))


def test_dataset_filter_preserves_validation_and_excludes_heldout(tmp_path):
    path = tmp_path / "split.parquet"
    frame = pl.DataFrame({"oper_part_no": ["a"] * 4, "seq": [4, 2, 1, 3],
                          "chronological_split": ["test", "train", "train", "validation"],
                          "demand_qty": [float("nan"), 4., 2., 6.]})
    frame.write_parquet(path)
    observed = load_train_validation_frame(path)
    expected = frame.filter(pl.col("chronological_split") != "test").sort(["oper_part_no", "seq"])
    assert observed.equals(expected)
    assert observed["chronological_split"].to_list() == ["train", "train", "validation"]
