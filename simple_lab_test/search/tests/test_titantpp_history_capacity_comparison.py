"""Local proof and selection boundaries of the Validation-only linkage."""
import json

import pytest

from paper.scripts import compare_titantpp_history_capacity as compare


def endpoint(rmse):
    def cell(count):
        return {"count": count, "qty_sse": rmse ** 2 * count,
                "qty_rmse": rmse, "qty_mae": 1., "time_nll": .2}
    return {**cell(5), "body": cell(3), "tail": cell(1),
            "quantity_boundaries": [1, 2, 3, 4],
            "quantity_cells": [{"bin": i, **cell(1)} for i in range(5)],
            "history_cells": [{"bin": i, **cell(n)} for i, n in enumerate((2, 2, 1))],
            "history_boundaries": [64, 128], "state_sha256": str(rmse),
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "time_metric": "recorded_positive_integer_time_nll"}


def fixture_fit(tmp_path):
    job = {"dataset": "yellow_trip_hourly", "seed": 42, "arm": "titantpp_history_mlp_width8",
           "host": "5080", "id": "yellow_trip_hourly__42__titantpp_history_mlp_width8"}
    pop = {"target_count": 5, "target_identity_sha256": "v", "target_quantity_sha256": "qv"}
    train_pop = {"target_count": 10, "target_identity_sha256": "t", "target_quantity_sha256": "qt"}
    contract = {"datasets": [{"dataset_id": job["dataset"], "quantity_boundaries_all_train_rows": [1, 2, 3, 4],
                             "inherited_data_identity": {"populations": {"validation": pop, "train": train_pop}}}],
                "hosts": {"5080": {"root": "/owned/5080"}}, "diagnostic": {"train_sampling": {
                    "max_per_bin": 2048, "train_sampling_seed": 20261003}},
                "source": {"git_revision": "fixed", "files_sha256": "closure"}}
    folder = tmp_path / "fit"
    run = folder / "runs" / job["arm"] / "count_only_log_regression" / "seed_42"
    run.mkdir(parents=True)
    history = [{"epoch": i, "val_qty_rmse": 2. if i == 2 else 3.,
                "val_qty_mae": 1., "val_time_nll": .2} for i in range(1, 41)]
    selected, last = endpoint(2.), endpoint(3.)
    remote_run = "/owned/5080/run/" + job["id"] + "/runs/" + job["arm"] + "/count_only_log_regression/seed_42/"
    selected["checkpoint_path"] = remote_run + "best_val_qty_rmse_model.pt"
    last["checkpoint_path"] = remote_run + "last_epoch_state.pt"
    replay = {"status": "complete", "job": job, "evaluation_scope": "validation_only",
              "held_out_test_evaluated": False, "best_epoch": 2, "completed_epochs": 40,
              "selected": selected, "last": last}
    provenance = {"checkpoint_path": selected["checkpoint_path"], "state_sha256": selected["state_sha256"],
                  "held_out_test_evaluated": False, "selection_unchanged": True, "new_training": False}
    sampling = {**contract["diagnostic"]["train_sampling"], "sample_n": 5, "represented_n": 10.}
    train = {"split": "train", "held_out_test_evaluated": False, "new_training": False,
             "state_sha256": selected["state_sha256"], "population": train_pop,
             "sampling": sampling, "count": 5, "represented_n": 10., "qty_rmse": 2., "qty_mae": 1.,
             "tail": {"qty_rmse": 2.}, "time_nll": .2}
    diagnostic = {"dataset": job["dataset"], "seed": 42, "model": job["arm"], "epoch": 2,
                  "provenance": provenance, "validation": {**selected, "split": "validation",
                      "full_population": True, "population": pop}, "train_sample": train}
    paired = {"dataset": job["dataset"], "seed": 42, "population_and_sample_identity_verified": True,
              "selection_unchanged": True, "candidate": {"model": job["arm"], "epoch": 2, "provenance": provenance},
              "candidate_minus_baseline": {"train_sample": {"overall": {"qty_rmse": -.1}}}}
    for name, value in (("endpoint_replays.json", replay), ("history.json", {"history": history}),
                        ("selected_train_validation_diagnostic.json", diagnostic), ("width4_candidate_comparison.json", paired)):
        (run / name).write_text(json.dumps(value))
    (run / "best_val_qty_rmse_model.pt").write_bytes(b"selected-original")
    (run / "last_epoch_state.pt").write_bytes(b"last-original")
    manifest = {"status": "complete", "scientific_success": True, "contract_sha256": compare.sha_json(contract),
                "job": job, "completed_unix": 1., "files": {str(p.relative_to(folder)): compare.sha_file(p)
                for p in run.iterdir()}}
    (folder / "terminal_manifest.json").write_text(json.dumps(manifest))
    return folder, run, job, contract


def reseal(folder, path, value):
    path.write_text(json.dumps(value))
    m = compare.read(folder / "terminal_manifest.json")
    m["files"][str(path.relative_to(folder))] = compare.sha_file(path)
    (folder / "terminal_manifest.json").write_text(json.dumps(m))


def test_earliest_minimum_preserves_tie_rule():
    history = [{"epoch": i + 1, "val_qty_rmse": v} for i, v in enumerate((3., 2., 2.))]
    assert compare.first_minimum(history) == 2


def test_verified_fit_keeps_full_validation_and_sample_separate(tmp_path):
    folder, _, job, contract = fixture_fit(tmp_path)
    row, proof = compare.verified_fit(folder, job, contract)
    assert row["qty_rmse"] == 2.
    assert row["selected_epoch"] == 2
    assert row["train_sample_n"] == 5 and row["train_represented_n"] == 10.
    assert proof["checkpoint_binary_sha_verified"]
    assert not proof["cpu_inference_audit_performed_here"]


def test_checkpoint_byte_change_rejected(tmp_path):
    folder, run, job, contract = fixture_fit(tmp_path)
    (run / "best_val_qty_rmse_model.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA mismatch"):
        compare.verified_fit(folder, job, contract)


@pytest.mark.parametrize("modification,message", [
    (lambda r: r.update(best_epoch=3), "strict first minimum"),
    (lambda r: r["last"].update(qty_rmse=10.), "RMSE/SSE mismatch"),
    (lambda r: r["selected"].update(held_out_test_evaluated=True), "Validation scope"),
    (lambda r: r["selected"].update(checkpoint_path="/another/fit.pt"), "owned fit"),
])
def test_replay_scope_selection_and_last_endpoint_rejected(tmp_path, modification, message):
    folder, run, job, contract = fixture_fit(tmp_path)
    p = run / "endpoint_replays.json"
    replay = compare.read(p)
    modification(replay)
    reseal(folder, p, replay)
    with pytest.raises(ValueError, match=message):
        compare.verified_fit(folder, job, contract)


def test_sample_population_weight_failure_is_not_full_training_score(tmp_path):
    folder, run, job, contract = fixture_fit(tmp_path)
    p = run / "selected_train_validation_diagnostic.json"
    diag = compare.read(p)
    diag["train_sample"]["sampling"]["represented_n"] = 5.
    reseal(folder, p, diag)
    with pytest.raises(ValueError, match="N/n accounting"):
        compare.verified_fit(folder, job, contract)


@pytest.mark.parametrize("path", ["../outside.json", "/outside.json", "runs/test_predictions.json"])
def test_manifest_unsafe_or_heldout_path_rejected(tmp_path, path):
    folder, _, job, contract = fixture_fit(tmp_path)
    m = compare.read(folder / "terminal_manifest.json")
    m["files"][path] = "unused"
    (folder / "terminal_manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Unsafe terminal|Held-out file forbidden"):
        compare.verify_manifest(folder, job, compare.sha_json(contract))


def rows_for(width, values):
    return [{"dataset": "yellow_trip_hourly", "seed": s, "width": width,
             **{m: v for m in compare.METRICS}} for s, v in zip(compare.SEEDS, values)]


def test_no_three_seed_conclusion_with_one_missing():
    rows = rows_for(16, (3., 3., 3.)) + rows_for(8, (2., 2., 2.))[:2]
    result = compare.group_decision(rows, "yellow_trip_hourly", 8, 16)
    assert result["category"] == "invalid_or_incomplete"
    assert not result["three_seed_conclusion"] and "metrics" not in result


def test_joint_improvement_reports_worst_seed_and_sample_sd():
    rows = rows_for(16, (3., 3., 3.)) + rows_for(8, (2., 2., 4.))
    result = compare.group_decision(rows, "yellow_trip_hourly", 8, 16)
    assert result["category"] == "joint_improvement"
    primary = result["metrics"]["qty_rmse"]
    assert primary["improved_seed_count"] == 2 and primary["worst_paired_seed_delta"] == 1.
    assert primary["candidate_sample_sd"] == pytest.approx(1.1547005383792515)
    assert not result["statistical_significance_established"]


def test_time_damage_is_disclosed_as_tradeoff():
    rows = rows_for(16, (3., 3., 3.)) + rows_for(8, (2., 2., 2.))
    for row in rows:
        if row["width"] == 8:
            row["time_nll"] = 4.
    result = compare.group_decision(rows, "yellow_trip_hourly", 8, 16)
    assert result["category"] == "tradeoff"
    assert result["metrics"]["time_nll"]["mean_delta"] == 1.


def test_one_seed_gain_does_not_pass_majority_screen():
    rows = rows_for(16, (3., 3., 3.)) + rows_for(8, (0., 4., 4.))
    result = compare.group_decision(rows, "yellow_trip_hourly", 8, 16)
    assert result["category"] == "no_stable_capacity_gain"
    assert not result["majority_primary_screen_passed"]


def observation_fixture():
    job = {"host": "5080", "id": "approved_job"}
    observation = {"observed_unix": 12., "files": {
        "status.json": {"status": "running", "active_job": job, "supervisor_pid": 101},
        "run/approved_job/status.json": {"status": "training"},
        "progress.json": {"job": "approved_job", "pid": 102}},
        "gpu": "102, python3.12, 1200 MiB\n",
        "processes": ["101 python3.12 run_titantpp_history_capacity_campaign.py --mode dispatch",
                      "102 python3.12 run_titantpp_history_capacity_campaign.py --mode fit --job approved_job"]}
    return job, observation


def test_live_owned_worker_and_gpu_confirm_observed_training():
    job, observation = observation_fixture()
    assert compare.status_from_observation(job, {"5080": ("snapshot", observation)}) == (
        "training_confirmed_at_observation", 12.)


def test_host_failure_overrides_stale_training_json():
    job, observation = observation_fixture()
    observation["files"]["failure.json"] = {"error": "cost gate"}
    assert compare.status_from_observation(job, {"5080": ("snapshot", observation)})[0] == "host_failure_status_unconfirmed"


@pytest.mark.parametrize("change", [
    lambda o: o.update(processes=[]),
    lambda o: o.update(gpu=""),
    lambda o: o.update(gpu="999, unrelated.py, 100 MiB\n"),
    lambda o: o["files"]["progress.json"].update(pid=999),
])
def test_process_or_gpu_absence_leaves_training_unconfirmed(change):
    job, observation = observation_fixture()
    change(observation)
    assert compare.status_from_observation(job, {"5080": ("snapshot", observation)})[0] == "training_status_unconfirmed"
