"""Capacity scope, authorization and supervisor checks; no remote or real-data work."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from paper.scripts import prepare_titantpp_history_capacity as preparation
from paper.scripts import run_titantpp_history_capacity_campaign as campaign
from paper.scripts import run_titantpp_history_width as width


@pytest.fixture
def contract(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "synthetic.py").write_text("# synthetic source\n")
    monkeypatch.setattr(campaign, "ROOT", source)
    revision = "a" * 40
    monkeypatch.setenv("SOURCE_REVISION", revision)
    files = {"synthetic.py": campaign.common.sha_file(source / "synthetic.py")}
    datasets = [{"dataset_id": d} for ds in campaign.ASSIGNMENTS.values() for d in ds]
    jobs = [{"id": f"{d}__{s}__{a}", "host": h, "dataset": d, "seed": s, "arm": a}
            for h, ds in campaign.ASSIGNMENTS.items() for d in ds
            for s in campaign.SEEDS for a in campaign.ARMS]
    architecture = {"widths": [8, 12], "cnn_gru_included": False, "main_hidden_dimension": 64,
                    "residual_divisor": 8, "warm_start": False}
    return {"schema": campaign.SCHEMA, "arms": list(campaign.ARMS), "jobs": jobs,
            "roles": {a: campaign.role_for(a) for a in campaign.ARMS},
            "quantity_variants": {a: campaign.VARIANT for a in campaign.ARMS},
            "datasets": datasets, "dataset_sha256": {d["dataset_id"]: campaign.common.sha_json(d) for d in datasets},
            "training": {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40,
                         "patience": 40, "monitor": "validation_raw_quantity_rmse",
                         "tie": "strict_earliest_finite_minimum", "warm_start": False},
            "limits": {"total_wall_seconds": 168 * 3600, "per_condition_seconds": 36 * 3600,
                       "workers_per_host": 1, "automatic_retry": False},
            "reuse": [{"host": h, "dataset": d, "seed": s, "arm": "titantpp_history_mlp"}
                      for h, ds in campaign.ASSIGNMENTS.items() for d in ds for s in campaign.SEEDS],
            "baseline_replays": {d["dataset_id"]: {str(s): {} for s in campaign.SEEDS} for d in datasets},
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "source": {"git_revision": revision, "files": files, "files_sha256": campaign.common.sha_json(files)},
            "architecture": architecture, "design_sha256": campaign.common.sha_json(architecture),
            "reference_baseline": "titantpp_history_mlp_width16", "deferred_datasets": ["insta_market_basket"],
            "hosts": {h: {"root": str(tmp_path / h), "environment": {"PYTHONHASHSEED": "42"}}
                      for h in campaign.ASSIGNMENTS}}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def authorize(c, root, now=1000.):
    approval = {"approved": True, "contract_sha256": campaign.common.sha_json(c),
                "hosts": ["5080", "5090"], "user_instruction": preparation.USER_INSTRUCTION}
    start = {"contract_sha256": campaign.common.sha_json(c),
             "approval_sha256": campaign.common.sha_json(approval),
             "started_at_unix": now, "deadline_unix": now + 168 * 3600}
    write(root / "approval.json", approval)
    write(root / "start_permit.json", start)
    return approval, start


def test_exact_capacity_scope_and_per_seed_baseline_reuse(contract):
    assert campaign.validate(contract) is contract
    assert len(contract["jobs"]) == 18 and len(contract["reuse"]) == 9
    assert len([j for j in contract["jobs"] if j["host"] == "5080"]) == 12
    assert len([j for j in contract["jobs"] if j["host"] == "5090"]) == 6
    assert {j["seed"] for j in contract["jobs"]} == {42, 52, 62}
    assert "insta_market_basket" not in {j["dataset"] for j in contract["jobs"]}
    assert contract["reference_baseline"] == "titantpp_history_mlp_width16"
    for ds in {j["dataset"] for j in contract["jobs"]}:
        for seed in campaign.SEEDS:
            assert len({j["host"] for j in contract["jobs"] if j["dataset"] == ds and j["seed"] == seed}) == 1
    assert preparation.ARMS == campaign.ARMS and preparation.SEEDS == campaign.SEEDS
    assert preparation.ASSIGNMENTS == campaign.ASSIGNMENTS


@pytest.mark.parametrize("mutation", ["missing_fit", "extra_fit", "wrong_host", "seed", "baseline_fit",
                                      "retry", "warm_start", "workers", "test", "time_monitor",
                                      "reference", "lost_reuse", "missing_baseline_seed", "source", "dataset"])
def test_scope_source_training_and_split_drift_fail_closed(contract, mutation):
    c = deepcopy(contract)
    if mutation == "missing_fit": c["jobs"].pop()
    elif mutation == "extra_fit": c["jobs"].append(deepcopy(c["jobs"][0]))
    elif mutation == "wrong_host": c["jobs"][0]["host"] = "5090"
    elif mutation == "seed": c["jobs"][0]["seed"] = 72
    elif mutation == "baseline_fit": c["jobs"][0]["arm"] = "titantpp_history_mlp_width16"
    elif mutation == "retry": c["limits"]["automatic_retry"] = True
    elif mutation == "warm_start": c["training"]["warm_start"] = True
    elif mutation == "workers": c["limits"]["workers_per_host"] = 2
    elif mutation == "test": c["held_out_test_evaluated"] = True
    elif mutation == "time_monitor": c["training"]["monitor"] = "validation_time_nll"
    elif mutation == "reference": c["reference_baseline"] = "titantpp_history_mlp"
    elif mutation == "lost_reuse": c["reuse"].pop()
    elif mutation == "missing_baseline_seed": c["baseline_replays"]["yellow_trip_hourly"].pop("62")
    elif mutation == "source": c["source"]["files"]["synthetic.py"] = "b" * 64
    else: c["datasets"][0]["extra"] = "drift"
    with pytest.raises(ValueError): campaign.validate(c)


def test_deferred_instacart_cannot_enter_admitted_population(contract):
    d = {"dataset_id": "insta_market_basket"}
    contract["datasets"].append(d)
    contract["dataset_sha256"][d["dataset_id"]] = campaign.common.sha_json(d)
    with pytest.raises(ValueError, match="ataset|cope"):
        campaign.validate(contract)


def test_approval_budget_and_source_revision_are_required(contract, tmp_path, monkeypatch):
    root = tmp_path / "authorization"
    _, start = authorize(contract, root)
    monkeypatch.setattr(campaign.time, "time", lambda: 1001.)
    assert campaign.authorization(contract, root) == start
    changed = {**start, "deadline_unix": start["deadline_unix"] + 1}
    write(root / "start_permit.json", changed)
    with pytest.raises(ValueError, match="budget"):
        campaign.authorization(contract, root)
    write(root / "start_permit.json", start)
    monkeypatch.setattr(campaign.time, "time", lambda: start["deadline_unix"])
    with pytest.raises(ValueError, match="budget"):
        campaign.authorization(contract, root)
    monkeypatch.setenv("SOURCE_REVISION", "b" * 40)
    with pytest.raises(ValueError, match="Source revision"):
        campaign.validate(contract)


@pytest.mark.parametrize("host", ["5080", "5090"])
def test_training_permit_bound_to_own_host_and_qualification(contract, host, monkeypatch):
    root = Path(contract["hosts"][host]["root"])
    _, start = authorize(contract, root)
    q = {"status": "passed", "host": host, "contract_sha256": campaign.common.sha_json(contract),
         "source_files_sha256": contract["source"]["files_sha256"], "runtime": {"environment": {"PYTHONHASHSEED": "42"}}}
    permit = {"schema": campaign.TRAINING_PERMIT_SCHEMA, "host": host,
              "contract_sha256": campaign.common.sha_json(contract), "start_permit_sha256": campaign.common.sha_json(start),
              "qualifications": {host: q}}
    write(root / "qualification/receipt.json", q)
    write(root / "training_permit.json", permit)
    assert campaign.verify_training_permit(contract, root) == permit
    other = "5090" if host == "5080" else "5080"
    write(root / "training_permit.json", {**permit, "host": other})
    with pytest.raises(ValueError, match="host"):
        campaign.verify_training_permit(contract, root)
    write(root / "training_permit.json", {**permit, "qualifications": {host: q, other: q}})
    with pytest.raises(ValueError, match="own native"):
        campaign.verify_training_permit(contract, root)
    write(root / "training_permit.json", permit)
    write(root / "qualification/receipt.json", {**q, "status": "failed"})
    with pytest.raises(ValueError, match="changed"):
        campaign.verify_training_permit(contract, root)


def model_data():
    return {"model": {"hidden_dim": 64, "quantity_variant": width.VARIANT,
                      "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 7.,
                      "time_initial_location": .2, "time_initial_scale": .8,
                      "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None}},
            "statistics": {"train_log_mean": 1.2, "train_log_std": .8}, "loader": {"max_seq_len": 256}}


@pytest.mark.parametrize("seed", campaign.SEEDS)
def test_campaign_initialization_matches_width_contract_and_preserves_rng(seed):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        before = torch.get_rng_state().clone()
        actual = campaign.initial_states(model_data(), seed)
        assert torch.equal(before, torch.get_rng_state())
        expected = width.initial_states(model_data(), seed, arms=(width.BASELINE, *campaign.ARMS))
        assert actual == expected and len(set(actual.values())) == 3
    finally:
        torch.set_num_threads(previous)


class ReachedBudget(Exception):
    pass


def runtime(seed):
    return {"torch": "synthetic", "gpu": {"uuid": "owned_gpu"},
            "environment": {"PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}


@pytest.mark.parametrize("seed", campaign.SEEDS)
def test_fit_runtime_accepts_only_approved_per_seed_hash_change(contract, monkeypatch, seed):
    job = next(j for j in contract["jobs"] if j["seed"] == seed)
    root = Path(contract["hosts"][job["host"]]["root"])
    q = {"status": "passed", "runtime": runtime(42)}
    monkeypatch.setattr(campaign, "authorization", lambda *_: {"deadline_unix": 9999999999.})
    monkeypatch.setattr(campaign, "read", lambda *_: deepcopy(q))
    monkeypatch.setattr(campaign, "verify_training_permit", lambda *_: None)
    monkeypatch.setattr(campaign.common, "runtime_check", lambda *_: runtime(seed))
    monkeypatch.setenv("PYTHONHASHSEED", str(seed))
    def stop(*_, **__):
        def before_fit(): raise ReachedBudget()
        return before_fit
    monkeypatch.setattr(campaign, "PulseBudget", stop)
    with pytest.raises(ReachedBudget):
        campaign.run_fit(contract, job["host"], root, job, 9999999999.)
    changed = runtime(seed)
    changed["gpu"]["uuid"] = "other_gpu"
    monkeypatch.setattr(campaign.common, "runtime_check", lambda *_: changed)
    with pytest.raises(ValueError, match="runtime"):
        campaign.run_fit(contract, job["host"], root, job, 9999999999.)


@pytest.mark.parametrize("mutation", ["GPU", "CUBLAS", "hash", "torch", "qualification_hash", "qualification_status", "seed"])
def test_qualified_runtime_only_normalizes_hash_for_an_approved_seed(contract, mutation):
    job = next(j for j in contract["jobs"] if j["seed"] == 52)
    q = {"status": "passed", "runtime": runtime(42)}
    actual = runtime(52)
    assert campaign.verify_fit_runtime(contract, job["host"], q, job, actual)
    assert q["runtime"]["environment"]["PYTHONHASHSEED"] == "42"
    if mutation == "GPU": actual["gpu"]["uuid"] = "other_gpu"
    elif mutation == "CUBLAS": actual["environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    elif mutation == "hash": actual["environment"]["PYTHONHASHSEED"] = "62"
    elif mutation == "torch": actual["torch"] = "drift"
    elif mutation == "qualification_hash": q["runtime"]["environment"]["PYTHONHASHSEED"] = "52"
    elif mutation == "qualification_status": q["status"] = "failed"
    else: job = {**job, "seed": 72}
    with pytest.raises(ValueError):
        campaign.verify_fit_runtime(contract, job["host"], q, job, actual)


def test_budget_obeys_absolute_and_monotonic_deadlines_and_lease(tmp_path, monkeypatch):
    now = [1000.]
    monotonic = [500.]
    monkeypatch.setattr(campaign.time, "time", lambda: now[0])
    monkeypatch.setattr(campaign.time, "monotonic", lambda: monotonic[0])
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda *_: SimpleNamespace(free=6 * 1024**3))
    budget = campaign.PulseBudget(tmp_path, "digest", "job", 1100., lease=True)
    write(tmp_path / "server_lease.json", {"contract_sha256": "digest", "job": "job", "expires_unix": 1090.})
    budget()
    assert json.loads((tmp_path / "progress.json").read_text())["job"] == "job"
    monotonic[0] = 600.
    with pytest.raises(ValueError, match="budget"):
        budget()
    monotonic[0] = 501.; now[0] = 1003.
    write(tmp_path / "server_lease.json", {"contract_sha256": "digest", "job": "foreign", "expires_unix": 1090.})
    with pytest.raises(ValueError, match="lease"):
        budget()


def test_existing_run_cannot_be_resumed_or_retried(contract, monkeypatch):
    host = "5080"
    root = Path(contract["hosts"][host]["root"])
    root.mkdir(parents=True)
    write(root / "qualification/receipt.json", {"status": "passed"})
    first = next(j for j in contract["jobs"] if j["host"] == host)
    (root / "run" / first["id"]).mkdir(parents=True)
    monkeypatch.setattr(campaign, "authorization", lambda *_: {"deadline_unix": 9999999999.})
    monkeypatch.setattr(campaign, "verify_training_permit", lambda *_: {"qualifications": {host: {"status": "passed"}}})
    monkeypatch.setattr(campaign.common, "gpu_pids", lambda *_: set())
    monkeypatch.setattr(campaign.shutil, "which", lambda *_: "/synthetic/timeout")
    def forbidden(*_, **__): pytest.fail("An existing run must not create a child")
    monkeypatch.setattr(campaign.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="no implicit restart"):
        campaign.dispatch(contract, host, root)
    failure = json.loads((root / "failure.json").read_text())
    assert failure["automatic_retry"] is False
    assert not (root / "claims").exists()
