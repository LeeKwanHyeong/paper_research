import json
from unittest.mock import patch

import polars as pl
import pytest
import torch
from torch.nn import functional as F

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.hard_lmm_frozen_probe import (
    FrozenResidualProbe, acceptance, extract_features, fit_probe, metric_values,
    predict, sample_indices, summarize,
)
from paper.scripts.run_hard_lmm_frozen_probe import (
    fresh_directory, gpu_preflight, make_dataset, run_guarded,
    verify_baseline_replay, verify_hashes, visible_frame,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


@pytest.fixture
def frozen_model():
    torch.manual_seed(23)
    model, _ = build_count_aware_model("titantpp", hidden_dim=16, train_log_mean=1.0,
                                     train_log_std=0.5, max_seq_len=6)
    with torch.no_grad():
        model.quantity_head.weight.normal_(std=0.1)
    return model.requires_grad_(False).eval()


def batch():
    dts = torch.tensor([[0., 0., 1., 2., 3., 1.], [0., 1., 1., 2., 1., 4.]])
    mask = torch.tensor([[False, False, True, True, True, True], [False, True, True, True, True, True]])
    qty = torch.tensor([[0., 0., 1., 2., 4., 5.], [0., 3., 2., 7., 1., 14.]])
    return dts, mask, qty


@pytest.mark.parametrize("candidate", ["calibration", "shrinkage"])
def test_identity_and_nonzero_first_gradient(frozen_model, candidate):
    cache = extract_features(frozen_model, *batch())
    probe = FrozenResidualProbe(42, candidate)
    z, gate, correction = probe(cache["features"], cache["z"], cache["projection"])
    assert torch.equal(z, cache["z"])
    assert torch.equal(gate, torch.ones_like(gate))
    assert torch.count_nonzero(correction) == 0
    loss = (F.softplus(z) - cache["quantity"].log1p()).square().mean()
    loss.backward()
    assert probe.network[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in frozen_model.parameters())


@pytest.mark.parametrize("candidate", ["calibration", "shrinkage"])
@pytest.mark.parametrize("bias", [-100., 100.])
def test_bounds(candidate, bias):
    probe = FrozenResidualProbe(3, candidate)
    with torch.no_grad():
        probe.network[-1].bias.fill_(bias)
    z, gate, correction = probe(torch.randn(5, 3), torch.zeros(5), torch.ones(5))
    assert bool(torch.isfinite(z).all())
    if candidate == "calibration":
        assert bool((correction.abs() <= 0.05).all())
    else:
        assert bool(((gate >= 0.8) & (gate <= 1)).all())
        torch.testing.assert_close(correction, gate - 1)


def test_official_predictions_time_and_causal_feature_invariance(frozen_model):
    dts, mask, qty = batch()
    cache = extract_features(frozen_model, dts, mask, qty)
    with torch.no_grad():
        official = target_outputs(frozen_model, dts, mask, qty, lambda_log_qty=1)
    torch.testing.assert_close(F.softplus(cache["z"]).expm1(), official["pred_qty"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cache["time_nll"], official["time_loss"], rtol=1e-6, atol=1e-6)
    changed_qty, changed_dt = qty.clone(), dts.clone()
    changed_qty[:, -1] = 1e12
    changed_dt[:, -1] = 1e5
    changed_qty[~mask] = 1e15
    changed_dt[~mask] = 1e15
    changed = extract_features(frozen_model, changed_dt, mask, changed_qty)
    for key in ("features", "z", "projection", "history_length"):
        assert torch.equal(cache[key], changed[key]), key


def test_series_independence_and_base_freeze(frozen_model):
    dts, mask, qty = batch()
    before = canonical_state_dict_sha256(frozen_model.state_dict())
    original = extract_features(frozen_model, dts[:1], mask[:1], qty[:1])
    extract_features(frozen_model, dts[1:], mask[1:], qty[1:])
    again = extract_features(frozen_model, dts[:1], mask[:1], qty[:1])
    assert torch.equal(original["features"], again["features"])
    assert before == canonical_state_dict_sha256(frozen_model.state_dict())
    frozen_model.train()
    with pytest.raises(ValueError, match="eval"):
        extract_features(frozen_model, dts, mask, qty)


@pytest.mark.parametrize("candidate", ["calibration", "shrinkage"])
def test_fit_updates_only_probe_and_replays_checkpoint(frozen_model, candidate, tmp_path):
    cache = extract_features(frozen_model, *batch())
    before = canonical_state_dict_sha256(frozen_model.state_dict())
    policy = {"learning_rate": .001, "shuffle_seed": 42, "maximum_epochs": 2,
              "minimum_epochs": 1, "patience": 2, "batch_size": 2, "gradient_clip": 1}
    probe, history, selection = fit_probe(candidate, cache, cache, policy, lambda value: None)
    assert len(history) == 3 and selection["first_gradient_norm"] > 0
    assert before == canonical_state_dict_sha256(frozen_model.state_dict())
    torch.save(probe.state_dict(), tmp_path / "adapter.pt")
    restored = FrozenResidualProbe(42, candidate)
    restored.load_state_dict(torch.load(tmp_path / "adapter.pt", weights_only=True))
    assert all(torch.equal(a, b) for a, b in zip(predict(probe, cache), predict(restored, cache)))
    assert metric_values(predict(probe, cache)[0], cache["quantity"], cache["time_nll"])["joint_objective"] <= history[0]["joint_objective"]


def test_train_sampling_reproducible_and_full_below_cap():
    assert torch.equal(sample_indices(10, 20), torch.arange(10))
    sampled = sample_indices(1000, 128)
    assert torch.equal(sampled, sample_indices(1000, 128))
    assert len(sampled.unique()) == 128
    assert bool((sampled[1:] > sampled[:-1]).all())


def test_test_rows_excluded_before_features_and_target_split_is_fixed(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"oper_part_no": ["a"] * 6, "seq": [1, 2, 3, 4, 5, 6], "delta_t": [1.] * 6,
        "demand_qty": [1., 2., 3., 4., 5., float("nan")],
        "chronological_split": ["train"] * 3 + ["validation"] * 2 + ["test"]}).write_parquet(path)
    frame = visible_frame(path)
    assert frame.height == 5
    row = {"lookback": 10, "max_seq_len": 6}
    training = make_dataset(frame, row, "train")
    validation = make_dataset(frame, row, "validation")
    assert len(training) == 2 and len(validation) == 2
    assert not set(training.index) & set(validation.index)
    with pytest.raises(ValueError, match="Held-out"):
        make_dataset(frame, row, "test")


def test_empty_strata_and_smoke_cannot_pass(frozen_model):
    cache = extract_features(frozen_model, *batch())
    z = cache["z"]
    scopes = summarize(cache, z, torch.ones_like(z), torch.zeros_like(z), [20, 30, 40, 50])
    assert scopes["gt_p99"] == {"count": 0, "status": "empty"}
    assert acceptance(scopes, 1, False)["status"] == "not_assessed_smoke"
    assert acceptance(scopes, 1, True)["status"] == "not_assessable_empty_stratum"


def test_artifacts_are_immutable_and_failure_status_is_durable(tmp_path):
    output = tmp_path / "failed"
    def fail(_status):
        raise RuntimeError("synthetic OOM")
    with pytest.raises(RuntimeError):
        run_guarded(output, fail, {})
    status = json.loads((output / "probe_status.json").read_text())
    assert status["status"] == "failed" and "OOM" in status["error"]
    with pytest.raises(FileExistsError):
        fresh_directory(output)


def test_checksum_and_baseline_replay_reject_mismatch(tmp_path):
    (tmp_path / "source.py").write_text("changed")
    with pytest.raises(ValueError, match="Checksum"):
        verify_hashes(tmp_path, {"source.py": "0" * 64})
    with pytest.raises(AssertionError, match="replay"):
        verify_baseline_replay({"qty_mae": 2.0}, {"best_val_qty_mae": 1.0})


@pytest.mark.parametrize("bad", ["vram", "busy", "gdm", "desktop"])
def test_gpu_preflight_fails_before_training(bad):
    class Result:
        returncode = 0
        stderr = ""
        def __init__(self, stdout):
            self.stdout = stdout
    results = [Result("NVIDIA GeForce RTX 5080, " + ("1000" if bad == "vram" else "15800")),
               Result("123" if bad == "busy" else ""), Result("active" if bad == "gdm" else "inactive"),
               Result("gnome-shell" if bad == "desktop" else "python sshd")]
    with patch("paper.scripts.run_hard_lmm_frozen_probe.subprocess.run", side_effect=results):
        with pytest.raises(RuntimeError):
            gpu_preflight()


def test_nonfinite_raw_prediction_is_rejected():
    with pytest.raises(FloatingPointError):
        metric_values(torch.tensor([1000.]), torch.tensor([1.]), torch.tensor([0.]))
