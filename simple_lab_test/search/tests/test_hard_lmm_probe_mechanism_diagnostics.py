import json

import pytest
import torch

from paper.scripts.diagnose_hard_lmm_frozen_probes import (
    activity, fit_constant, load_cache, taxi_comparison, trace_shrinkage,
)
from paper.scripts.hard_lmm_frozen_probe import FrozenResidualProbe
from paper.scripts.run_hard_lmm_frozen_probe import sha256_file


def cache(quantity=100.):
    n = 16
    return {"features": torch.ones(n, 3), "z": torch.ones(n),
        "projection": torch.ones(n), "quantity": torch.full((n,), quantity),
        "time_nll": torch.full((n,), 2.), "history_length": torch.full((n,), 12),
        "target_index": torch.arange(n)}


POLICY = {"learning_rate": .001, "shuffle_seed": 42, "batch_size": 4, "gradient_clip": 1}
BOUNDARIES = [1., 2., 3., 4.]


def test_trace_distinguishes_dead_clamp_from_healthy_gate(tmp_path):
    dead = trace_shrinkage(cache(), BOUNDARIES, POLICY, tmp_path / "dead.jsonl", epochs=2)
    live = trace_shrinkage(cache(0.), BOUNDARIES, POLICY, tmp_path / "live.jsonl", epochs=2)
    assert dead["first_steps"][0]["gradient_norm"] > 0
    assert dead["history"][-1]["zero_gradient_batch_fraction"] == 1
    assert dead["snapshots"][-1]["scopes"]["overall"]["identity_fraction"] == 1
    assert live["history"][-1]["zero_gradient_batch_fraction"] == 0
    assert live["snapshots"][-1]["scopes"]["overall"]["relative_residual_norm_reduction_mean"] > 0
    assert dead["validation_evaluated"] is False
    assert dead["checkpoint_replaced"] is False
    assert dead["snapshots"][-1]["scopes"]["gt_p99"]["count"] == 16


def test_constant_fit_is_bounded_and_only_depends_on_train_logits_and_labels():
    high, curve = fit_constant(torch.zeros(16), torch.full((16,), 100.))
    low, _ = fit_constant(torch.ones(16), torch.zeros(16))
    assert high["offset"] == .05 and low["offset"] == -.05
    assert len(curve) == 1001 and any(row["offset"] == 0 for row in curve)
    with pytest.raises(ValueError):
        fit_constant(torch.zeros(16), torch.zeros(16), points=100)


def test_activity_measures_relative_and_projected_reduction():
    row = activity(torch.tensor([0., .5, 1.]), torch.tensor([1., .9, .8]), torch.tensor([0., -.3, .4]))
    assert row["identity_fraction"] == pytest.approx(1 / 3)
    assert row["relative_residual_norm_reduction_mean"] == pytest.approx(.1)
    assert row["projected_residual_reduction_abs_mean"] == pytest.approx(.7 / 3)


def test_taxi_identity_reconciles_bins_preserves_time_and_empty_strata():
    train = cache(1.)
    probe = FrozenResidualProbe(3, "calibration").eval()
    result = taxi_comparison(train, probe, 0., BOUNDARIES)
    assert result["bins"]["overall"]["prediction_gap_abs_mean"] == 0
    assert result["bins"]["gt_p99"]["status"] == "empty"
    assert result["constant_gain_fraction_of_mlp"]["squared_error_reduction_sum"] is None
    for rows in result["scopes"].values():
        assert rows["overall"]["baseline"] == rows["overall"]["candidate"]
    assert result["time_route_changed"] is False


def test_cache_rejects_test_other_validation_and_digest_mismatch(tmp_path):
    for dataset, split in (("yellow_trip_hourly", "test"), ("raf_spare_parts", "validation")):
        with pytest.raises(ValueError, match="permitted"):
            load_cache(tmp_path, dataset, split, {})
    folder = tmp_path / "yellow_trip_hourly"
    folder.mkdir()
    path = folder / "train_cache.pt"
    torch.save(cache(), path)
    audit = {"cache": {"train": {"sha256": sha256_file(path), "selected_targets": 16}}}
    (folder / "baseline_audit.json").write_text(json.dumps(audit))
    hashes = {}
    assert len(load_cache(tmp_path, "yellow_trip_hourly", "train", hashes)["z"]) == 16
    assert str(path) in hashes
    torch.save(cache(0.), path)
    with pytest.raises(ValueError, match="checksum"):
        load_cache(tmp_path, "yellow_trip_hourly", "train", {})


def test_nonfinite_and_output_overwrite_rejected(tmp_path):
    with pytest.raises(FloatingPointError):
        activity(torch.tensor([float("nan")]), torch.ones(1), torch.zeros(1))
    path = tmp_path / "trace.jsonl"
    path.touch()
    with pytest.raises(FileExistsError):
        trace_shrinkage(cache(), BOUNDARIES, POLICY, path, epochs=1)
