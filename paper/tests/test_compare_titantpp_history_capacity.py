"""Local aggregate-reader contracts; no GPU, network, inference or held-out data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_titantpp_history_capacity.py"
spec = importlib.util.spec_from_file_location("capacity_comparison_under_test", SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_transferred_fit_uses_operational_owner_without_mutating_original(tmp_path):
    original = {"id": "intermittent__62__width8", "dataset": "intermittent", "seed": 62, "arm": "width8", "host": "5090"}
    operational_job = {**original, "host": "5080"}
    parent, operational = {"jobs": [original]}, {"jobs": [operational_job]}
    prior = deepcopy(parent)
    folder, job, contract = c.retrieved_fit(tmp_path / "parent", original, parent, tmp_path / "handoff",
                                           (operational, {"job_ids": [original["id"]]}))
    assert folder == tmp_path / "handoff/retrieved/5080/run" / original["id"]
    assert job == operational_job and contract == operational and parent == prior


def test_duplicate_original_transferred_terminal_is_rejected(tmp_path):
    job = {"id": "intermittent__62__width12", "host": "5090"}
    folder = tmp_path / "parent/retrieved/5090/run" / job["id"]
    folder.mkdir(parents=True)
    (folder / "terminal_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Duplicate"):
        c.retrieved_fit(tmp_path / "parent", job, {}, tmp_path / "handoff",
                        ({"jobs": [{**job, "host": "5080"}]}, {"job_ids": [job["id"]]}))


def test_untransferred_fit_retains_parent_folder_and_contract(tmp_path):
    job = {"id": "taxi__42__width8", "host": "5080"}
    parent = {"scientific": "parent"}
    folder, fit_job, contract = c.retrieved_fit(tmp_path / "parent", job, parent, tmp_path / "handoff",
                                               ({}, {"job_ids": ["intermittent__62__width8"]}))
    assert folder == tmp_path / "parent/retrieved/5080/run" / job["id"]
    assert fit_job is job and contract is parent


@pytest.mark.parametrize("gpu,compute", [("101, GPU-0, python, 2MiB\n", {}),
                                          ({"stdout": "GPU-0"}, {"returncode": 0, "stdout": "101, GPU-0, python, 2MiB\n"})])
def test_training_evidence_supports_legacy_and_structured_observation(gpu, compute):
    job = {"id": "taxi__42__width8", "host": "5080"}
    observation = {"observed_unix": 100, "gpu": gpu, "compute": compute,
                   "processes": ["101 python run_titantpp_history_capacity_campaign.py --mode fit --job taxi__42__width8",
                                 "99 python run_titantpp_history_capacity_campaign.py --mode dispatch"],
                   "files": {"status.json": {"active_job": job, "supervisor_pid": 99},
                             "progress.json": {"job": job["id"], "pid": 101}}}
    assert c.status_from_observation(job, {"5080": (None, observation)}) == ("training_confirmed_at_observation", 100)
    observation["processes"] = observation["processes"][:1]
    assert c.status_from_observation(job, {"5080": (None, observation)})[0] == "training_status_unconfirmed"


def test_recorded_selection_uses_earliest_finite_minimum():
    history = [{"epoch": i, "val_qty_rmse": value} for i, value in enumerate((3.0, 2.0, 2.0, 4.0), 1)]
    assert c.first_minimum(history) == 2
    history[2]["val_qty_rmse"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        c.first_minimum(history)


@pytest.mark.parametrize("name", ["test/result.json", "heldout/result.json", "../outside.json"])
def test_terminal_rejects_forbidden_paths_before_opening(tmp_path, name):
    manifest = {"status": "complete", "scientific_success": True, "contract_sha256": "contract",
                "job": {"id": "job"}, "files": {name: "sha"}}
    (tmp_path / "terminal_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="forbidden|Unsafe"):
        c.verify_manifest(tmp_path, manifest["job"], "contract")


def test_terminal_file_tampering_is_rejected(tmp_path):
    sealed = tmp_path / "history.json"
    sealed.write_text("{}")
    manifest = {"status": "complete", "scientific_success": True, "contract_sha256": "contract",
                "job": {"id": "job"}, "files": {"history.json": c.sha_file(sealed)}}
    (tmp_path / "terminal_manifest.json").write_text(json.dumps(manifest))
    assert c.verify_manifest(tmp_path, manifest["job"], "contract") == manifest
    sealed.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="SHA mismatch"):
        c.verify_manifest(tmp_path, manifest["job"], "contract")


def rows(width, values):
    return [{"dataset": "dataset", "seed": seed, "width": width, **dict.fromkeys(c.METRICS, value)}
            for seed, value in zip(c.SEEDS, values)]


def test_complete_seed_summary_uses_sample_sd_and_omits_incomplete_group():
    summary = c.seed_summary(rows(8, (1, 2, 3)) + rows(12, (1, 2)))
    assert len(summary) == 1 and summary[0]["width"] == 8 and summary[0]["sd_ddof"] == 1
    assert summary[0]["qty_rmse_mean"] == 2 and summary[0]["qty_rmse_sample_sd"] == 1


def test_adoption_discloses_worst_seed_and_time_tradeoff():
    baseline = rows(16, (10, 10, 10))
    candidate = rows(8, (9, 9, 11))
    for row in candidate:
        row["time_nll"] = 12
    group = c.group_decision(baseline + candidate, "dataset", 8, 16)
    assert group["category"] == "tradeoff" and group["majority_primary_screen_passed"] is True
    assert group["metrics"]["qty_rmse"]["worst_paired_seed"] == 62
    assert group["metrics"]["qty_rmse"]["worst_paired_seed_delta"] == 1
    assert group["statistical_significance_established"] is False
    incomplete = c.group_decision(baseline + candidate[:2], "dataset", 8, 16)
    assert incomplete["category"] == "invalid_or_incomplete" and incomplete["three_seed_conclusion"] is False
