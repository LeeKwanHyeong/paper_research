"""Synthetic-only width diagnostic checks; no real checkpoints or held-out data."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from paper.scripts import evaluate_titantpp_history_width as m


BOUNDS = [7., 686., 1562., 3449.]


def test_metrics_strict_tail_weighted_errors_and_empty_bins():
    metrics = m.QuantityMetrics(BOUNDS)
    metrics.add([7., 3449., 4000.], [9., 3446., 4010.], [1., 2., 3.], [2., 3., 5.])
    out = metrics.finish()
    assert out["count"] == 3 and out["represented_n"] == 10
    assert out["qty_rmse"] == pytest.approx(np.sqrt((2 * 4 + 3 * 9 + 5 * 100) / 10))
    assert out["qty_mae"] == pytest.approx((2 * 2 + 3 * 3 + 5 * 10) / 10)
    assert out["qty_bias"] == pytest.approx((2 * 2 - 3 * 3 + 5 * 10) / 10)
    assert out["time_nll"] == 2.3
    assert out["tail"]["count"] == 1 and out["tail"]["qty_rmse"] == 10
    assert out["quantity_cells"][3]["count"] == 1  # Exactly 3449 is outside tail.
    empty = out["quantity_cells"][1]
    assert empty["count"] == empty["qty_sse"] == 0
    assert empty["qty_rmse"] is empty["qty_bias"] is empty["time_nll"] is None


@pytest.mark.parametrize("q,p,t,w", [([1], [np.inf], [0], [1]), ([1], [1], [np.nan], [1]),
                                     ([1], [1], [0], [0]), ([1], [-1], [0], [1])])
def test_invalid_metric_values_fail_closed(q, p, t, w):
    with pytest.raises(ValueError, match="Invalid metric"):
        m.QuantityMetrics(BOUNDS).add(q, p, t, w)


def test_streaming_matches_single_batch():
    rng = np.random.default_rng(17)
    q = rng.integers(0, 5000, 1000)
    p = rng.uniform(0, 5000, 1000)
    t = rng.normal(size=1000)
    w = rng.uniform(1, 4, 1000)
    one, batched = m.QuantityMetrics(BOUNDS), m.QuantityMetrics(BOUNDS)
    one.add(q, p, t, w)
    for start in range(0, 1000, 17):
        sl = slice(start, start + 17)
        batched.add(q[sl], p[sl], t[sl], w[sl])
    for k in ("qty_rmse", "qty_mae", "qty_bias", "time_nll", "qty_sse"):
        assert one.finish()[k] == pytest.approx(batched.finish()[k])


def test_sample_exactly_reproduces_diagnostic_rng_and_inverse_sampling_weights():
    q = np.concatenate([np.repeat(v, n) for v, n in zip([7, 686, 1562, 3449, 4000], [3, 4097, 1, 2050, 2200])])
    ix, w, receipt = m.frozen_train_sample(q, BOUNDS)
    # Independent reproduction of diagnose.py's choices, not a sample of a sample.
    rng = np.random.default_rng(20261003)
    expected, expected_w = [], []
    for value in (7, 686, 1562, 3449, 4000):
        candidates = np.flatnonzero(q == value)
        chosen = np.sort(rng.choice(candidates, min(2048, len(candidates)), replace=False))
        expected.extend(chosen)
        expected_w.extend([len(candidates) / len(chosen)] * len(chosen))
    assert np.array_equal(ix, expected) and np.array_equal(w, expected_w)
    assert np.all(np.diff(ix) > 0)
    assert receipt["represented_n"] == pytest.approx(len(q))
    assert receipt["sample_indices_sha256"] == hashlib.sha256(np.array(expected, dtype="<i8").tobytes()).hexdigest()
    assert receipt["weights_sha256"] == hashlib.sha256(np.array(expected_w, dtype="<f8").tobytes()).hexdigest()
    with pytest.raises(ValueError, match="sampling contract"):
        m.frozen_train_sample(q, BOUNDS, {"train_sampling_seed": 1, "max_per_bin": 2048})


def fake_population(split):
    q = np.array([7., 686., 1562., 3449., 4000.])
    dataset = SimpleNamespace(index=[(0, i) for i in range(5)], parts=["entity"],
                              seq_lists=[list(range(6))], val_lists=[[0.] + q.tolist()],
                              split_lists=[[split] * 6])
    h = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode() + b"\0")
    h.update(json.dumps(["entity"], separators=(",", ":")).encode())
    for a in (np.zeros(5, dtype="<i8"), np.arange(1, 6, dtype="<i8"), np.arange(1, 6, dtype="<i8")):
        h.update(a.tobytes())
    pop = {"target_count": 5, "target_identity_sha256": h.hexdigest(),
           "target_quantity_sha256": hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + q.astype("<f8").tobytes()).hexdigest()}
    return dataset, q, pop


def test_population_identity_detects_reordering_and_wrong_split():
    dataset, q, pop = fake_population("validation")
    assert np.array_equal(m.population_identity(dataset, "validation", pop)[0], q)
    dataset.index.reverse()
    with pytest.raises(ValueError, match="population identity"):
        m.population_identity(dataset, "validation", pop)
    with pytest.raises(ValueError, match="Only train/validation"):
        m.population_identity(dataset, "test", pop)


def endpoint_fixture(q):
    def cell(values):
        n = len(values)
        return {"count": n, "qty_mae": 1. if n else None, "qty_rmse": 1. if n else None,
                "qty_sse": float(n), "time_nll": 2. if n else None}
    groups = np.searchsorted(BOUNDS, q, side="left")
    return {**cell(q), "body": cell(q[q <= BOUNDS[2]]), "tail": cell(q[q > BOUNDS[3]]),
            "quantity_boundaries": BOUNDS, "history_boundaries": [2., 4.],
            "quantity_cells": [{"bin": i, **cell(q[groups == i])} for i in range(5)],
            "history_cells": [{"bin": i, **cell(q[lo:hi])} for i, (lo, hi) in enumerate([(0, 2), (2, 4), (4, 5)])],
            "state_sha256": "state", "evaluation_scope": "validation_only", "held_out_test_evaluated": False}


def test_cached_endpoint_keeps_metrics_without_inference_and_marks_missing_bias(monkeypatch):
    ds, q, pop = fake_population("validation")
    data = {"dataset_id": "yellow_trip_hourly", "quantity_boundaries_all_train_rows": BOUNDS,
            "inherited_data_identity": {"populations": {"validation": pop}}}
    payload = {"backbone": "titantpp_history_mlp_width16", "seed": 42, "epoch": 19,
               "checkpoint_monitor": "validation_raw_quantity_rmse"}
    monkeypatch.setattr(m, "_payload", lambda *_: (payload, "state"))
    monkeypatch.setattr(m, "_loader", lambda *_: SimpleNamespace(dataset=ds))
    def forbidden(*_, **__):
        pytest.fail("Cached endpoint must not construct a model or run inference")
    monkeypatch.setattr(m, "_build_model", forbidden)
    monkeypatch.setattr(m, "_infer", forbidden)
    out = m.evaluate_checkpoint(data, "synthetic.pt", None, validation_replay=endpoint_fixture(q))
    assert out["epoch"] == 19 and out["train_sample"] is None
    assert out["validation"]["qty_rmse"] == out["validation"]["tail"]["qty_rmse"] == 1
    assert out["validation"]["qty_bias"] is None and out["validation"]["bias_available"] is False
    assert out["validation"]["additional_validation_forward_pass"] is False
    assert out["provenance"]["raw_predictions_written"] is False


@pytest.mark.parametrize("change", ["state", "scope", "bounds", "population", "strata"])
def test_cached_endpoint_rejects_foreign_evidence(change):
    _, q, pop = fake_population("validation")
    data = {"dataset_id": "yellow_trip_hourly", "quantity_boundaries_all_train_rows": BOUNDS}
    replay = endpoint_fixture(q)
    if change == "state": replay["state_sha256"] = "foreign"
    if change == "scope": replay["held_out_test_evaluated"] = True
    if change == "bounds": replay["quantity_boundaries"] = [7., 686., 1562., 3448.]
    if change == "population": pop = {**pop, "target_count": 6}
    if change == "strata": q = np.array([7., 686., 1562., 3449., 3449.])
    with pytest.raises((ValueError, RuntimeError)):
        m.reuse_validation_replay(replay, data, pop, "state", q)


def comparison_fixture():
    _, q, pop = fake_population("train")
    ix, w, sampling = m.frozen_train_sample(q, BOUNDS)
    a = m.QuantityMetrics(BOUNDS)
    a.add(q[ix], q[ix] + 1, np.repeat(2., len(ix)), w)
    train = {**a.finish(), "population": pop, "sampling": sampling}
    validation = {**a.finish(), "population": {**pop, "target_identity_sha256": "validation"}}
    return {"dataset": "yellow_trip_hourly", "model": "titantpp_history_mlp", "seed": 42,
            "epoch": 7, "provenance": {"state_sha256": "baseline"}, "train_sample": train, "validation": validation}


def test_compare_enforces_all_population_sample_and_weight_identities():
    a = comparison_fixture()
    b = deepcopy(a)
    b.update(model="titantpp_history_mlp_width16", epoch=12)
    b["validation"]["qty_rmse"] = .8
    result = m.compare_evaluations(a, b)
    assert result["candidate_minus_baseline"]["validation"]["overall"]["qty_rmse"] == pytest.approx(-.2)
    assert result["baseline"]["epoch"] == 7 and result["candidate"]["epoch"] == 12
    for key in ("sample_indices_sha256", "weights_sha256", "train_sampling_seed"):
        changed = deepcopy(b)
        changed["train_sample"]["sampling"][key] = "changed"
        with pytest.raises(ValueError, match="TRAIN sample"):
            m.compare_evaluations(a, changed)
    changed = deepcopy(b)
    changed["validation"]["population"]["target_identity_sha256"] = "changed"
    with pytest.raises(ValueError, match="Validation population"):
        m.compare_evaluations(a, changed)


def test_inference_stream_uses_actual_targets_and_weights_without_gradients(monkeypatch):
    import data_loader.event_seq_data_module as dl
    from paper.scripts.count_aware_tpp_backbone import core
    q = np.array([7., 8., 4000.])
    seen = []
    def collate(items):
        x = torch.tensor([[q[i]] for i in items], dtype=torch.float32)
        return None, torch.ones_like(x), torch.ones_like(x, dtype=torch.bool), None, x
    def target_outputs(model, dt, mask, quantity, **kwargs):
        assert not torch.is_grad_enabled()
        seen.extend(quantity[:, 0].tolist())
        return {"true_qty": quantity[:, 0], "pred_qty": quantity[:, 0] + 1,
                "time_loss": torch.full_like(quantity[:, 0], 2.)}
    monkeypatch.setattr(dl, "collate_week_lookback", collate)
    monkeypatch.setattr(core, "target_outputs", target_outputs)
    out = m._infer(None, [0, 1, 2], np.array([0, 2]), np.array([2., 5.]), q, BOUNDS, 1, "cpu", lambda: None)
    assert seen == [7., 4000.]
    assert out["count"] == 2 and out["represented_n"] == 7
    assert out["qty_rmse"] == out["qty_bias"] == 1 and out["tail"]["count"] == 1


def test_frame_with_heldout_rows_is_rejected_before_loader():
    import polars as pl
    with pytest.raises(ValueError, match="Held-out rows"):
        m._loader({}, pl.DataFrame({"chronological_split": ["train", "test"]}), "validation")


def test_payload_rejects_selection_or_checkpoint_state_drift(monkeypatch):
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone.training import checkpoint_monitor_spec
    from simple_lab_test.search.common import runner
    state = {"weight": torch.tensor([1.])}
    selector = checkpoint_monitor_spec("validation_raw_quantity_rmse")
    payload = {"backbone": "titantpp_history_mlp", "variant": "count_only_log_regression",
               "model_state_dict": state, "model_state_sha256": runner.canonical_state_dict_sha256(state),
               "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
               "interface_meta": {"time_head": {"observation_likelihood": {"unit": "hour"}}},
               "checkpoint_monitor": "validation_raw_quantity_rmse",
               "checkpoint_monitor_history_key": selector["history_key"], "checkpoint_selection": selector["selection"]}
    monkeypatch.setattr(factory, "validate_checkpoint_route", lambda *_: None)
    monkeypatch.setattr(runner, "torch_load_checkpoint", lambda *_, **__: payload)
    data = {"model": {"time_observation_contract": {"unit": "hour"}}}
    assert m._payload("synthetic.pt", data)[1] == payload["model_state_sha256"]
    payload["checkpoint_selection"] = "last_minimum"
    with pytest.raises(ValueError, match="selector changed"):
        m._payload("synthetic.pt", data)
    payload["checkpoint_selection"] = selector["selection"]
    payload["model_state_dict"]["weight"].add_(1)
    with pytest.raises(ValueError, match="state digest"):
        m._payload("synthetic.pt", data)
