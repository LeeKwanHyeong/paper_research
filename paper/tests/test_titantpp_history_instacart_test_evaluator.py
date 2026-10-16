"""Operational safeguards for the approved selected-checkpoint Test adapter."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_titantpp_history_instacart_frozen_test.py"
SPEC = importlib.util.spec_from_file_location("_history_instacart_test_adapter", SCRIPT)
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def metrics():
    return {"count": 7, "qty_mae": 1., "qty_rmse": 2., "time_nll": .5,
            "tail": {"count": 2, "qty_mae": 3., "qty_rmse": 4., "time_nll": .7}}


def fixture_job(study="history", dataset="yellow_trip_hourly", arm="titantpp_cnn_gru54", seed=42):
    return {"id": f"{dataset}__{seed}__{arm}", "study": study, "dataset": dataset, "arm": arm, "seed": seed,
        "selected_epoch": 9, "initial_state_sha256": "a" * 64, "checkpoint_state_sha256": "b" * 64,
        "checkpoint": {"relative_path": "checkpoints/x.pt", "sha256": "c" * 64},
        "frozen_source_closure": "d" * 64, "selected_metric_value": 2., "source_revision_history": ["revision"],
        "validation_reference": {"split": "validation", **metrics()}}


def payload(job):
    return {"best_epoch": job["selected_epoch"], "backbone": job["arm"], "seed": job["seed"],
        "variant": "count_only_log_regression", "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "checkpoint_monitor": "validation_raw_quantity_rmse", "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": "best_validation_raw_quantity_rmse", "initial_state_sha256": job["initial_state_sha256"],
        "source_revision": "revision", "source_revision_history": ["revision"], "selected_metric_value": 2.,
        "model_state_sha256": job["checkpoint_state_sha256"], "interface_meta": {"time_head": {
            "mode": "duration", "time_scale": 1., "observation_likelihood": {"mode": "recorded", "top_code": None}}}}


def receipt(job, contract):
    return {"status": "complete", "contract_sha256": contract["contract_sha256"], "split": "validation",
        **{key: job[key] for key in ("id", "study", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256")},
        "checkpoint_sha256": job["checkpoint"]["sha256"], "state_sha256": job["checkpoint_state_sha256"],
        "source_closure": job["frozen_source_closure"], "retrained": False, "reselected": False,
        "raw_predictions_written": False, "full_population": True, "validation_match": "PASS",
        "metrics": metrics(), "qualification": dict.fromkeys(ev.QUALIFICATION_KEYS, True), "target_identity_sha256": "e" * 64,
        "truth_sha256": "f" * 64}


def global_gate(tmp_path):
    jobs = [fixture_job(*entry) for entry in sorted(ev.expected_jobs())]
    contract = {"contract_sha256": "contract", "jobs": jobs}
    bindings = {}
    for job in jobs:
        path = tmp_path / "results" / job["id"] / "validation_replay.json"
        ev.write_exclusive(path, {**receipt(job, contract), "completed_unix": 10.})
        ev.write_exclusive(path.parent / "validation_worker_exit.json", {"job_id": job["id"],
            "contract_sha256": "contract", "exit_code": 0, "actual_child_exit_observed": True, "completed_unix": 11.})
        bindings[job["id"]] = ev.sha(path)
    ev.write_exclusive(tmp_path / "validation_gate.json", {"status": "PASS", "contract_sha256": "contract",
        "job_count": 36, "validation_receipts": bindings, "created_unix": 12.})
    return contract


def test_import_does_not_load_scientific_runtime():
    # This adapter must be usable by preparation/control code before CUDA/data admission.
    assert "torch" not in ev.__dict__ and "polars" not in ev.__dict__


@pytest.mark.parametrize("study,arm,route", [
    ("history", "titantpp_cnn_gru54", ev.ROUTES["history"]),
    ("history", "titantpp_cnn_gru54_correction_disabled", ev.ROUTES["history"]),
    ("history", "titantpp_cnn_gru54_current_only", ev.ROUTES["history"]),
    ("instacart", "titantpp_history_mlp_width8", ev.ROUTES["instacart_mlp"]),
    ("instacart", "titantpp_history_mlp_width12", ev.ROUTES["instacart_mlp"]),
    ("instacart", "titantpp_cnn_gru54", ev.ROUTES["instacart_cnn"]),
])
def test_exact_routes(study, arm, route):
    dataset = "yellow_trip_hourly" if study == "history" else "insta_market_basket"
    assert ev.route_for(fixture_job(study, dataset, arm)) == route


@pytest.mark.parametrize("job", [fixture_job(arm="titantpp_history_mlp_width8"),
    fixture_job("instacart", "raf_spare_parts"), fixture_job("foreign")])
def test_route_rejects_cross_campaign(job):
    with pytest.raises(ValueError):
        ev.route_for(job)


@pytest.mark.parametrize("field,value", [("best_epoch", 10), ("seed", 52), ("backbone", "foreign"),
    ("evaluation_scope", "test"), ("checkpoint_selection", "last_epoch"),
    ("initial_state_sha256", "wrong"), ("source_revision_history", ["revision", "foreign"]),
    ("selected_metric_value", 1.9), ("model_state_sha256", "wrong")])
def test_payload_rejects_reselection_or_foreign_lineage(field, value):
    job = fixture_job()
    p = payload(job)
    p[field] = value
    definition = {"model": {"time_head_mode": "duration", "time_scale": 1.,
                              "time_observation_contract": {"mode": "recorded", "top_code": None}}}
    with pytest.raises(ValueError):
        ev.validate_payload(p, job, {"revision": "revision"}, definition)


def test_payload_whole_metadata_bound():
    job = fixture_job()
    p = payload(job)
    definition = {"model": {"time_head_mode": "duration", "time_scale": 1.,
                              "time_observation_contract": {"mode": "recorded", "top_code": None}}}
    ev.validate_payload(p, job, {"revision": "revision"}, definition, deepcopy(p))
    altered = deepcopy(p)
    altered["unrelated_model_option"] = True
    with pytest.raises(ValueError, match="complete metadata"):
        ev.validate_payload(p, job, {"revision": "revision"}, definition, altered)


@pytest.mark.parametrize("group,key,value", [(None, "count", 8), (None, "qty_mae", float("nan")),
    ("tail", "time_nll", .8), ("tail", "count", 1), (None, "qty_rmse", 2.001)])
def test_validation_gate_requires_exact_counts_and_finite_six_metrics(group, key, value):
    a, r = metrics(), metrics()
    cell = a if group is None else a[group]
    cell[key] = value
    with pytest.raises(ValueError):
        ev.compare_validation_metrics(a, r)


def test_validation_uses_max_isclose_rule_not_added_tolerances():
    a, r = metrics(), metrics()
    a["time_nll"] += .000011
    # abs=1e-5 and rel=5e-6: added-tolerance rule would pass; math.isclose must fail.
    with pytest.raises(ValueError):
        ev.compare_validation_metrics(a, r)
    a["time_nll"] = r["time_nll"] + .000009
    ev.compare_validation_metrics(a, r)


def test_global_gate_needs_all36(tmp_path):
    c = global_gate(tmp_path)
    ev.verify_global_validation_gate(tmp_path, c)
    gate = ev.read(tmp_path / "validation_gate.json")
    gate["validation_receipts"].pop(c["jobs"][0]["id"])
    (tmp_path / "validation_gate.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="missing or foreign"):
        ev.verify_global_validation_gate(tmp_path, c)


def test_global_gate_rejects_modified_receipt_bytes(tmp_path):
    c = global_gate(tmp_path)
    path = tmp_path / "results" / c["jobs"][0]["id"] / "validation_replay.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="bytes changed"):
        ev.verify_global_validation_gate(tmp_path, c)


@pytest.mark.parametrize("field,value", [("exit_code", 1), ("actual_child_exit_observed", False),
                                        ("completed_unix", 13.)])
def test_global_gate_requires_observed_exit0_before_test(field, value, tmp_path):
    c = global_gate(tmp_path)
    path = tmp_path / "results" / c["jobs"][0]["id"] / "validation_worker_exit.json"
    actual_exit = ev.read(path)
    actual_exit[field] = value
    path.write_text(json.dumps(actual_exit))
    with pytest.raises(ValueError, match="actual worker exit0"):
        ev.verify_global_validation_gate(tmp_path, c)


def test_global_gate_rejects_cross_seed_target_change_even_if_resealed(tmp_path):
    c = global_gate(tmp_path)
    job = c["jobs"][1]
    path = tmp_path / "results" / job["id"] / "validation_replay.json"
    r = ev.read(path)
    r["truth_sha256"] = "0" * 64
    path.write_text(json.dumps(r))
    gate = ev.read(tmp_path / "validation_gate.json")
    gate["validation_receipts"][job["id"]] = ev.sha(path)
    (tmp_path / "validation_gate.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="identical targets/truth"):
        ev.verify_global_validation_gate(tmp_path, c)


def test_source_checksum_and_confinement(tmp_path):
    path = tmp_path / "source" / "module.py"
    path.parent.mkdir()
    path.write_text("science = 1\n")
    files = {"module.py": ev.sha(path)}
    bundle = {"root": "source", "files": files, "closure": ev.canonical(files)}
    ev.verify_source(tmp_path, bundle)
    path.write_text("science = 2\n")
    with pytest.raises(ValueError, match="scientific source changed"):
        ev.verify_source(tmp_path, bundle)
    with pytest.raises(ValueError, match="escapes"):
        ev.confined(tmp_path, "../foreign")


def test_claims_never_overwrite_previous_attempt(tmp_path):
    path = tmp_path / "claim.json"
    ev.write_exclusive(path, {"attempt": 1})
    with pytest.raises(FileExistsError):
        ev.write_exclusive(path, {"attempt": 2})
    assert ev.read(path) == {"attempt": 1}


def test_raf_parent_must_preserve_exact_training_export():
    pl = pytest.importorskip("polars")
    full = pl.DataFrame({"oper_part_no": ["a", "a", "a"], "seq": [1, 2, 3],
                        "chronological_split": ["train", "validation", "test"], "demand_qty": [1., 2., 3.]})
    trainval = full.head(2)
    ev.require_raf_training_slice(full, trainval, pl)
    changed = trainval.with_columns(pl.col("demand_qty") + 1)
    with pytest.raises(ValueError, match="differs from frozen export"):
        ev.require_raf_training_slice(full, changed, pl)


def test_recorded_grid_checks_new_approved_split_without_relaxing_values():
    pl, np = pytest.importorskip("polars"), pytest.importorskip("numpy")
    frame = pl.DataFrame({"oper_part_no": ["a"] * 3, "seq": [1, 2, 3],
        "chronological_split": ["train", "validation", "test"], "demand_qty": [1., 2., 3.],
        "delta_t": [0., 1., 30.], "demand_dt": ["d1", "d2", "d3"]})
    d = {"model": {"time_observation_contract": {"top_code": 30}}}
    ev.validate_frame(frame, d, "test", np, pl)
    with pytest.raises(ValueError, match="Unapproved split"):
        ev.validate_frame(frame, d, "validation", np, pl)
    with pytest.raises(ValueError, match="top code"):
        ev.validate_frame(frame.with_columns(pl.lit(31.).alias("delta_t")), d, "test", np, pl)
    with pytest.raises(ValueError, match="integers"):
        ev.validate_frame(frame.with_columns(pl.lit(.5).alias("delta_t")), d, "test", np, pl)
