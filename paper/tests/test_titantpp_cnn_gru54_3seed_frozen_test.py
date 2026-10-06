"""Offline gate tests: no remote, GPU, or research population inference."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper/scripts/evaluate_titantpp_cnn_gru54_3seed_frozen_test.py"
spec = importlib.util.spec_from_file_location("cnn_gru_frozen_eval_tested", SCRIPT)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def metrics(count, bounds):
    def cell(n):
        return {"count": n, "represented_n": float(n), "qty_sse": float(n),
            "qty_absolute_error_sum": float(n), "qty_signed_error_sum": float(n),
            "time_nll_sum": float(n), "qty_rmse": 1. if n else None,
            "qty_mae": 1. if n else None, "qty_bias": 1. if n else None, "time_nll": 1. if n else None}
    cells = [{"bin": i, **cell(n)} for i, n in enumerate((count - 1, 0, 0, 0, 1))]
    return {**cell(count), "quantity_boundaries": bounds, "quantity_cells": cells,
        "tail": cell(1), "tail_definition": "raw_quantity > quantity_boundaries[-1]"}


@pytest.fixture
def campaign(tmp_path):
    parent = e.read(ROOT / "search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1/execution_contract.json")
    current = e.read(ROOT / "search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1/execution_contract.json")
    assert e.canonical(parent) == e.PARENT_SHA and e.canonical(current) == e.THREESEED_SHA
    dump(tmp_path / "parent_execution_contract.json", parent)
    dump(tmp_path / "original_three_seed_contract.json", current)
    bundles = {name: {"source_root": "source117" if name == "cnn_gru_117" else "source123",
        "source_files": deepcopy(original["source"]["files"]),
        "source_closure_sha256": original["source"]["files_sha256"], "datasets": deepcopy(original["datasets"])}
        for name, original in (("cnn_gru_117", parent), ("cnn_gru_123", current))}
    specs = {d["dataset_id"]: d for d in parent["datasets"]}
    oldjobs = {(j["dataset"], j["arm"], j["seed"]): j for j in parent["jobs"]}
    newjobs = {(j["dataset"], j["arm"], j["seed"]): j for j in current["jobs"]}
    manifest, refs, rows = {"datasets": []}, {}, []
    tests = dict(zip(e.DATASETS, (8327, 88019, 5226)))
    for dataset in e.DATASETS:
        d = specs[dataset]
        origin = d.get("parent_data_identity") or d["inherited_data_identity"]
        n = d["inherited_data_identity"]["populations"]["validation"]["target_count"]
        data = {"dataset": dataset, "path": "/fake/" + dataset + ".parquet", "sha256": origin["data"]["sha256"],
            "split_manifest": {"path": "/fake/split.json", "sha256": origin["split_manifest"]["sha256"]},
            "loader": d["loader"], "populations": {"validation": {"target_count": n}, "test": {"target_count": tests[dataset]}}}
        manifest["datasets"].append(data)
        refs[dataset] = {split: {"target_identity_sha256": "synthetic_targets_" + split,
            "truth_sha256": "synthetic_truth_" + split, "loader": d["loader"],
            "data_file_sha256": data["sha256"], "expected_target_count": count}
            for split, count in (("validation", n), ("test", tests[dataset]))}
        for seed in e.SEEDS:
            arm = e.ARMS[0]
            training = parent if seed == 42 else current
            job = (oldjobs if seed == 42 else newjobs)[dataset, arm, seed]
            rows.append({"dataset": dataset, "model": arm, "seed": seed, "job": job,
                "endpoint": "selected", "evaluator_source_bundle": "cnn_gru_117" if seed == 42 else "cnn_gru_123",
                "checkpoint_path": "original/" + job["id"] + "/best_val_qty_rmse_model.pt",
                "checkpoint_file_sha256": "binary", "state_tensor_sha256": "tensor", "selected_epoch": 2,
                "selected_metric_value": 1., "initial_state_sha256": "initial", "validation_count": n,
                "source_revision": training["source"]["base_git_revision"],
                "original_binding": {"training_contract_sha256": e.PARENT_SHA if seed == 42 else e.THREESEED_SHA,
                    "terminal_manifest_sha256": "terminal"}})
    registry = {"rows": rows, "bundles": bundles}
    dump(tmp_path / "evaluation_registry.json", registry)
    dump(tmp_path / "dataset_manifest.json", manifest)
    comparison_receipt = {"status": "complete", "candidate_conditions": 9,
        "scope": "Validation_only_original_quantity_selected_endpoints", "frozen_contract_sha256": e.THREESEED_SHA,
        "held_out_performance_read": False, "mixed_performance_file_read": False, "raw_prediction_read": False}
    dump(tmp_path / "final_validation_comparison_receipt.json", comparison_receipt)
    gate = {"passed": True, "conditions": 9, "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "comparison_receipt_path": "final_validation_comparison_receipt.json",
        "comparison_receipt_sha256": e.sha(tmp_path / "final_validation_comparison_receipt.json"),
        "scientific_adoption_gate": False,
        "rows": [{k:r[k] for k in ("dataset", "model", "seed", "checkpoint_file_sha256", "state_tensor_sha256", "selected_epoch")} for r in rows]}
    dump(tmp_path / "comparison_gate.json", gate)
    lineage = {"Test_previously_accessed": True, "independent_untouched_Test": False,
        "seed42_checkpoint_Test_previously_evaluated": True, "seed52_62_new_checkpoint_Test_inference": True,
        "fresh_replay_splits": ["validation", "test"], "head_refits_Test_evaluated": False,
        "prior_registry_sha256": "prior", "original_reuse_registry_sha256": "original"}
    dump(tmp_path / "evaluation_lineage.json", lineage)
    c = {"schema_version": 1, "campaign_kind": e.KIND, "parent_training_contract_sha256": e.PARENT_SHA,
        "original_3seed_training_contract_sha256": e.THREESEED_SHA,
        "source_closure_sha256": e.SOURCE_SHA, "automatic_retry": False, "raw_predictions_written": False,
        "selection_changed": False, "new_training": False,
        "approval": {"test_inference_authorized": True, "retraining_authorized": False,
            "allowed_splits": ["validation", "test"], "datasets": list(e.DATASETS), "models": list(e.ARMS), "seeds": list(e.SEEDS)},
        "registry_sha256": e.sha(tmp_path / "evaluation_registry.json"),
        "dataset_manifest_sha256": e.sha(tmp_path / "dataset_manifest.json"),
        "dataset_sha256": {d["dataset"]: d["sha256"] for d in manifest["datasets"]},
        "evaluator_sha256": e.PRIOR_SHA, "prior_evaluator": {"path": "prior.py", "sha256": e.PRIOR_SHA},
        "accepted_training_contracts": [{"path": "parent_execution_contract.json", "canonical_sha256": e.PARENT_SHA},
            {"path": "original_three_seed_contract.json", "canonical_sha256": e.THREESEED_SHA}],
        "resources": {"root": str(tmp_path), "python": "/native/python", "gpu_uuid": "native-gpu",
            "condition_timeout_seconds": 900, "campaign_timeout_seconds": 7200, "concurrent_workers": 1,
            "threads": 4, "rss_limit_gib": 12, "gpu_memory_limit_gib": 12, "max_device_memory_fraction": .8},
        "runtime_expected": {"numpy": "native"}, "environment": {},
        "population_references": refs, "validation_references": {e.identity(r): metrics(r["validation_count"], specs[r["dataset"]]["quantity_boundaries_all_train_rows"]) for r in rows},
        "validation_gate": {"conditions": 9, "all_before_test": True, "rel_tol": 1e-5, "abs_tol": 1e-5},
        "final_validation_comparison_gate": {"path": "comparison_gate.json", "sha256": e.sha(tmp_path / "comparison_gate.json")},
        "evaluation_lineage_sha256": e.sha(tmp_path / "evaluation_lineage.json"), "test_exposure_lineage": lineage}
    dump(tmp_path / "execution_contract.json", c)
    return tmp_path, c, registry, manifest


def receipt(campaign, c, registry, row, split):
    bundle = registry["bundles"][row["evaluator_source_bundle"]]
    spec = next(d for d in bundle["datasets"] if d["dataset_id"] == row["dataset"])
    p = c["population_references"][row["dataset"]][split]
    return {"status": "complete", "full_population": True, "max_batches": None,
        "retrained": False, "raw_predictions_written": False, "parts": [], "persisted_row_fields": [],
        "split": split, "heldout_read": split == "test", "contract_sha256": e.sha(campaign / "execution_contract.json"),
        "source_closure_sha256": bundle["source_closure_sha256"], "evaluator_source_bundle": row["evaluator_source_bundle"],
        "qualification": {k: True for k in e.QUALIFICATION_KEYS},
        **{k: row[k] for k in ("dataset", "model", "seed", "selected_epoch", "checkpoint_file_sha256", "state_tensor_sha256")},
        "runner_sha256": c["evaluator_sha256"], "registry_sha256": c["registry_sha256"], "dataset_manifest_sha256": c["dataset_manifest_sha256"],
        **p, "prediction_rows": p["expected_target_count"],
        "metrics": metrics(p["expected_target_count"], spec["quantity_boundaries_all_train_rows"]),
        "runtime": {"cpu_peak_rss_bytes": 1024, "cuda_peak_reserved_bytes": 0}}


def make_gate(campaign, c, registry):
    entries = []
    for row in registry["rows"]:
        path = campaign / "runs/validation" / e.identity(row) / "receipt.json"
        dump(path, receipt(campaign, c, registry, row, "validation"))
        entries.append(e.check_receipt(path, row, "validation", c, campaign))
    gate = {"passed": True, "conditions": 9, "contract_sha256": e.sha(campaign / "execution_contract.json"), "rows": entries}
    dump(campaign / "validation_gate.json", gate)
    return gate


def test_scientific_contract_and_path_only_mapping(campaign):
    folder, _, _, _ = campaign
    _, c, registry = e.load_campaign(folder, verify_bytes=False)
    assert len(registry["rows"]) == 9 and c["automatic_retry"] is False
    specs = registry["bundles"]["cnn_gru_117"]["datasets"]
    changed = deepcopy(specs)
    changed[0]["inherited_data_identity"]["data"]["path"] = "/5080/relocated.parquet"
    assert e.dataset_science(changed) == e.dataset_science(specs)
    changed[0]["inherited_data_identity"]["data"]["sha256"] = "changed"
    assert e.dataset_science(changed) != e.dataset_science(specs)


@pytest.mark.parametrize("change", ["missing", "duplicate", "seed", "model", "source", "loss", "tail", "split_SHA", "count", "approval", "retry", "tolerance", "workers"])
def test_changed_scope_or_science_stops_before_evaluation(campaign, change):
    folder, c, registry, manifest = campaign
    if change == "missing": registry["rows"].pop()
    if change == "duplicate": registry["rows"][-1] = registry["rows"][0]
    if change == "seed": registry["rows"][0]["seed"] = 52
    if change == "model": registry["rows"][0]["model"] = "titantpp_history_mlp_width16"
    if change == "source": registry["bundles"]["cnn_gru_117"]["source_files"]["data_loader/event_seq_data_module.py"] = "changed"
    if change == "loss": registry["bundles"]["cnn_gru_117"]["datasets"][0]["model"]["lambda_log_qty"] = 2
    if change == "tail": registry["bundles"]["cnn_gru_117"]["datasets"][0]["quantity_boundaries_all_train_rows"][-1] += 1
    if change == "split_SHA": manifest["datasets"][0]["split_manifest"]["sha256"] = "changed"
    if change == "count": c["population_references"][e.DATASETS[0]]["test"]["expected_target_count"] += 1
    if change == "approval": c["approval"]["test_inference_authorized"] = False
    if change == "retry": c["automatic_retry"] = True
    if change == "tolerance": c["validation_gate"]["rel_tol"] = .1
    if change == "workers": c["resources"]["concurrent_workers"] = 2
    dump(folder / "evaluation_registry.json", registry)
    dump(folder / "dataset_manifest.json", manifest)
    c["registry_sha256"], c["dataset_manifest_sha256"] = e.sha(folder / "evaluation_registry.json"), e.sha(folder / "dataset_manifest.json")
    dump(folder / "execution_contract.json", c)
    with pytest.raises(ValueError): e.load_campaign(folder, verify_bytes=False)


@pytest.mark.parametrize("change", ["partial", "duplicate", "receipt_SHA", "receipt_metrics", "checkpoint", "split", "population", "memory", "tail", "predictions"])
def test_global_gate_rejects_incomplete_or_corrupt_evidence(campaign, change):
    folder, c, registry, _ = campaign
    gate = make_gate(folder, c, registry)
    if change == "partial": gate["rows"].pop()
    elif change == "duplicate": gate["rows"][-1] = gate["rows"][0]
    elif change == "receipt_SHA": gate["rows"][0]["receipt_sha256"] = "changed"
    else:
        entry = gate["rows"][0]
        path = folder / entry["receipt_path"]
        value = e.read(path)
        if change == "receipt_metrics": value["metrics"]["time_nll"] += .1
        if change == "checkpoint": value["state_tensor_sha256"] = "foreign"
        if change == "split": value["split"] = "test"
        if change == "population": value["truth_sha256"] = "different"
        if change == "memory": value["runtime"]["cpu_peak_rss_bytes"] = 13 * 1024 ** 3
        if change == "tail": value["metrics"]["quantity_boundaries"][-1] += 1
        if change == "predictions": (path.parent / "predictions").mkdir()
        dump(path, value)
        entry["receipt_sha256"] = e.sha(path)
    dump(folder / "validation_gate.json", gate)
    with pytest.raises(ValueError): e.verify_validation_gate(folder, c, registry["rows"])


def test_validation_failure_never_starts_test_and_failure_is_preserved(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e, "load_campaign", lambda folder: (folder, c, registry))
    calls = []
    def fail(campaign, c, row, split, device, deadline):
        calls.append(split)
        raise ValueError("same-checkpoint Validation mismatch")
    with pytest.raises(ValueError, match="Validation mismatch"):
        e.pipeline(folder, "cpu", runner=fail)
    assert calls == ["validation"]
    assert not (folder / "validation_gate.json").exists()
    status = e.read(folder / "pipeline_status.json")
    assert status["status"] == "failed" and status["automatic_retry"] is False
    before = (folder / "pipeline_status.json").read_bytes()
    with pytest.raises(ValueError, match="Existing attempt"):
        e.pipeline(folder, "cpu", runner=fail)
    assert (folder / "pipeline_status.json").read_bytes() == before


def test_all_nine_validation_then_nine_test_exactly_once(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e, "load_campaign", lambda folder: (folder, c, registry))
    calls = []
    def run(campaign, c, row, split, device, deadline):
        if split == "test":
            assert len(calls) >= 9 and all(s == "validation" for s, _ in calls[:9])
            e.verify_validation_gate(campaign, c, registry["rows"])
        path = campaign / "runs" / split / e.identity(row) / "receipt.json"
        dump(path, receipt(campaign, c, registry, row, split))
        calls.append((split, e.identity(row)))
        return e.check_receipt(path, row, split, c, campaign)
    result = e.pipeline(folder, "cpu", runner=run)
    assert len(calls) == len(set(calls)) == 18
    assert [s for s, _ in calls] == ["validation"] * 9 + ["test"] * 9
    assert len(result["records"]) == 9 and result["full_population_splits"] == 18
    assert result["CPU_binary_audit_performed"] is False
    for record in result["records"]:
        assert record["validation"]["selected"]["qty_rmse"] == 1.
        assert record["test"]["tail"]["count"] == 1
        assert record["selected_epoch"] == 2 and record["checkpoint_file_sha256"] == "binary"
    assert not list(folder.rglob("*.parquet"))


def test_worker_test_gate_fails_before_loading_model_or_population(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e, "load_campaign", lambda folder: (folder, c, registry))
    monkeypatch.setattr(e, "verify_worker_launch", lambda *a: {"pipeline_pid": 1})
    monkeypatch.setattr(e, "prior_evaluator", lambda *a: pytest.fail("No scientific module or population may load"))
    with pytest.raises(FileNotFoundError):
        e.worker(folder, e.identity(registry["rows"][0]), "test", "cpu")
    assert not (folder / "claims").exists()


def test_exclusive_split_claim_rejects_second_worker(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e, "load_campaign", lambda folder: (folder, c, registry))
    monkeypatch.setattr(e, "verify_worker_launch", lambda *a: {"pipeline_pid": 1, "native_timeout_seconds": 60, "deadline_monotonic": e.time.monotonic() + 60})
    monkeypatch.setattr(e, "prior_evaluator", lambda *a: (_ for _ in ()).throw(ValueError("synthetic worker failure")))
    condition = e.identity(registry["rows"][0])
    with pytest.raises(ValueError, match="synthetic worker failure"):
        e.worker(folder, condition, "validation", "cpu")
    claim = folder / "claims" / ("validation__" + condition + ".json")
    before = claim.read_bytes()
    with pytest.raises(FileExistsError): e.worker(folder, condition, "validation", "cpu")
    assert claim.read_bytes() == before and e.read(claim)["automatic_retry"] is False


def test_first_minimum_tie_preserves_original_and_nonfinite_rejected():
    value = {"history": [{"epoch": i, "val_qty_rmse": rmse, "train_all_finite": True}
        for i, rmse in enumerate((4., 3., 3.), 1)]}
    before = deepcopy(value)
    assert e.first_minimum(value)["epoch"] == 2 and value == before
    value["history"][0]["val_qty_rmse"] = float("nan")
    with pytest.raises(ValueError, match="finite"): e.first_minimum(value)


def test_original_byte_bindings_earliest_epoch_and_cross_gpu_reference(campaign):
    folder, c, registry, _ = campaign
    row = deepcopy(registry["rows"][0])
    jobroot = folder / "original" / row["job"]["id"]
    jobroot.mkdir(parents=True)
    binary = jobroot / "best_val_qty_rmse_model.pt"
    binary.write_bytes(b"immutable synthetic checkpoint")
    history = {"history": [{"epoch": i, "val_qty_rmse": rmse, "train_all_finite": True}
        for i, rmse in enumerate((4., 1., 1.), 1)]}
    point = deepcopy(c["validation_references"][e.identity(row)])
    point.update(evaluation_scope="validation_only", held_out_test_evaluated=False, state_sha256="tensor")
    point["qty_rmse"] = 1.000004
    endpoint = {"status": "complete", "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "selected": point, "last": point, "completed_epochs": 3, "best_epoch": 2}
    dump(jobroot / "history.json", history)
    dump(jobroot / "endpoint_replays.json", endpoint)
    terminal = {"status": "complete", "scientific_success": True, "job": row["job"], "contract_sha256": e.PARENT_SHA,
        "files": {p.name: e.sha(p) for p in (binary, jobroot / "history.json", jobroot / "endpoint_replays.json")}}
    dump(jobroot / "terminal_manifest.json", terminal)
    row["checkpoint_path"], row["checkpoint_file_sha256"] = str(binary.relative_to(folder)), e.sha(binary)
    row["original_binding"] = {"training_contract_sha256": e.PARENT_SHA,
        **{key: str((jobroot / name).relative_to(folder)) for key, name in (
            ("terminal_manifest_path", "terminal_manifest.json"), ("history_path", "history.json"), ("endpoint_path", "endpoint_replays.json"))},
        **{key: e.sha(jobroot / name) for key, name in (
            ("terminal_manifest_sha256", "terminal_manifest.json"), ("history_sha256", "history.json"), ("endpoint_sha256", "endpoint_replays.json"))}}
    frozen = deepcopy(row)
    assert e.verify_original(folder, row, {e.PARENT_SHA: {}})["qty_rmse"] == 1.000004
    assert row == frozen
    later_tie = deepcopy(row); later_tie["selected_epoch"] = 3
    with pytest.raises(ValueError, match="Earliest minimum"):
        e.verify_original(folder, later_tie, {e.PARENT_SHA: {}})
    binary.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA changed"):
        e.verify_original(folder, row, {e.PARENT_SHA: {}})


def test_routing_proxy_changes_only_frozen_model_route(campaign):
    folder, c, registry, _ = campaign
    row = registry["rows"][0]
    calls = []
    engine, ordinary = object(), object()
    runner = SimpleNamespace(torch_load_checkpoint=lambda *a, **kw: {
        "selected_metric_value": row["selected_metric_value"], "model_state_sha256": row["state_tensor_sha256"]})
    def imported(name, package=None):
        calls.append(name)
        return runner if name == "simple_lab_test.search.common.runner" else engine if name == "paper.scripts.run_titantpp_cnn_gru" else ordinary
    proxy = e.FrozenImportProxy(SimpleNamespace(import_module=imported), c, folder, row)
    assert proxy.import_module("paper.scripts.run_titantpp_history_width") is engine
    assert proxy.import_module("unrelated.module") is ordinary
    assert calls == ["paper.scripts.run_titantpp_cnn_gru", "simple_lab_test.search.common.runner", "unrelated.module"]
    with pytest.raises(ValueError, match="Repeated"):
        proxy.import_module("paper.scripts.run_titantpp_history_width")


@pytest.mark.parametrize("remaining,expected", [(20., 20.), (2000., 900.)])
def test_native_timeout_is_bounded_by_campaign(campaign, monkeypatch, remaining, expected):
    folder, c, registry, _ = campaign
    launches = []
    monkeypatch.setattr(e.shutil, "which", lambda name: "/usr/bin/timeout")
    monkeypatch.setattr(e.time, "monotonic", lambda: 100.)
    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(e.subprocess, "Popen", launch)
    monkeypatch.setattr(e, "check_receipt", lambda *a: {"verified": True})
    e.run_condition(folder, c, registry["rows"][0], "validation", "cpu", 100. + remaining)
    command, kwargs = launches[0]
    assert command[:3] == ["/usr/bin/timeout", "--signal=TERM", "--kill-after=10s"]
    assert float(command[3]) == expected
    assert kwargs["start_new_session"] is True
    assert len(kwargs["env"]["TITANTPP_EVAL_LAUNCH_TOKEN"]) == 64
    launch = e.read(folder / "launches" / ("validation__" + e.identity(registry["rows"][0]) + ".json"))
    assert "token" not in launch and launch["native_timeout_seconds"] == expected


def test_missing_native_timeout_stops_before_scientific_worker(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e.shutil, "which", lambda name: None)
    monkeypatch.setattr(e.subprocess, "Popen", lambda *a, **kw: pytest.fail("No scientific worker may start"))
    with pytest.raises(ValueError, match="GNU timeout"):
        e.run_condition(folder, c, registry["rows"][0], "validation", "cpu", 999999.)


def test_contract_library_environment_is_present_before_python_exec(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    c["environment"] = {"LD_LIBRARY_PATH": "/native/nvidia/cu13/lib", "SOURCE_REVISION": registry["rows"][0]["source_revision"],
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    monkeypatch.setenv("LD_LIBRARY_PATH", "/parent/other-libraries")
    monkeypatch.setenv("PARENT_INHERITED_SETTING", "preserved")
    monkeypatch.setattr(e.shutil, "which", lambda name: "/usr/bin/timeout")
    monkeypatch.setattr(e.time, "monotonic", lambda: 100.)
    launched = []
    def launch(command, **kwargs):
        launched.append(kwargs["env"])
        return SimpleNamespace(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(e.subprocess, "Popen", launch)
    monkeypatch.setattr(e, "check_receipt", lambda *a: {"verified": True})
    e.run_condition(folder, c, registry["rows"][0], "validation", "cpu", 200.)
    child = launched[0]
    assert all(child[key] == value for key, value in c["environment"].items())
    assert child["PARENT_INHERITED_SETTING"] == "preserved"
    assert len(child["TITANTPP_EVAL_LAUNCH_TOKEN"]) == 64
    assert e.os.environ["LD_LIBRARY_PATH"] == "/parent/other-libraries"


def test_public_worker_cannot_bypass_pipeline_launch(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e, "load_campaign", lambda folder: (folder, c, registry))
    monkeypatch.delenv("TITANTPP_EVAL_LAUNCH_TOKEN", raising=False)
    monkeypatch.setattr(e, "prior_evaluator", lambda *a: pytest.fail("No model/population may load"))
    with pytest.raises(ValueError, match="ephemeral pipeline launch token"):
        e.worker(folder, e.identity(registry["rows"][0]), "validation", "cpu")
    assert not (folder / "claims").exists()


def test_launch_token_is_bound_to_live_native_timeout_and_pipeline(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    condition = e.identity(registry["rows"][0])
    token = "1" * 64
    native = ["/usr/bin/timeout", "60", "/native/python", str(SCRIPT), "--mode", "worker"]
    owner = ["/native/python", str(SCRIPT), "--campaign", str(folder), "--mode", "pipeline"]
    permit = {"condition": condition, "split": "validation", "pipeline_pid": 123,
        "contract_sha256": e.sha(folder / "execution_contract.json"), "native_command": native,
        "token_sha256": e.hashlib.sha256(token.encode()).hexdigest(), "native_timeout_seconds": 60,
        "deadline_monotonic": e.time.monotonic() + 60}
    path = folder / "launches" / f"validation__{condition}.json"
    dump(path, permit)
    commands = {e.os.getppid(): native, 123: owner}
    assert e.verify_worker_launch(folder, condition, "validation", c, token=token, process_reader=commands.__getitem__, process_parent_reader=lambda pid: 123) == permit
    with pytest.raises(ValueError, match="token binding"):
        e.verify_worker_launch(folder, condition, "validation", c, token="2" * 64, process_reader=commands.__getitem__, process_parent_reader=lambda pid: 123)
    commands[e.os.getppid()] = ["/native/python", str(SCRIPT), "--mode", "worker"]
    with pytest.raises(ValueError, match="native timeout"):
        e.verify_worker_launch(folder, condition, "validation", c, token=token, process_reader=commands.__getitem__, process_parent_reader=lambda pid: 123)
    commands[e.os.getppid()] = native
    commands[123] = ["/native/python", str(SCRIPT), "--campaign", str(folder), "--mode", "worker"]
    with pytest.raises(ValueError, match="pipeline command"):
        e.verify_worker_launch(folder, condition, "validation", c, token=token, process_reader=commands.__getitem__, process_parent_reader=lambda pid: 123)


def test_common_worker_lock_rejects_parallel_owned_worker(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e, "load_campaign", lambda folder: (folder, c, registry))
    monkeypatch.setattr(e, "verify_worker_launch", lambda *a: {"pipeline_pid": 1})
    monkeypatch.setattr(e, "prior_evaluator", lambda *a: pytest.fail("Second worker may not load models"))
    with (folder / "worker.lock").open("a") as lock:
        e.fcntl.flock(lock, e.fcntl.LOCK_EX | e.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            e.worker(folder, e.identity(registry["rows"][0]), "validation", "cpu")
    assert not (folder / "claims").exists()


def test_process_group_rss_includes_children_and_excludes_unowned(tmp_path):
    for pid, group, rss in ((100, 700, 8), (101, 700, 12), (102, 900, 100000)):
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "stat").write_text(f"{pid} (torch worker) R 1 {group} {group} 0")
        (folder / "status").write_text(f"Name: python\nVmRSS:\t{rss} kB\n")
    assert e.process_group_rss_bytes(700, proc_root=tmp_path) == 20 * 1024


def test_exceeded_live_rss_kills_only_owned_child_and_preserves_evidence(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e.shutil, "which", lambda name: "/usr/bin/timeout")
    monkeypatch.setattr(e.time, "monotonic", lambda: 100.)
    process = SimpleNamespace(pid=700, poll=lambda: None)
    monkeypatch.setattr(e.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(e, "process_group_rss_bytes", lambda pgid: 13 * 1024 ** 3 if pgid == 700 else pytest.fail("Unowned process inspected"))
    killed = []
    monkeypatch.setattr(e, "stop_child", lambda owned: killed.append(owned.pid))
    with pytest.raises(ValueError, match="RSS ceiling exceeded"):
        e.run_condition(folder, c, registry["rows"][0], "validation", "cpu", 200.)
    assert killed == [700]
    evidence = e.read(folder / "failures" / ("validation__" + e.identity(registry["rows"][0]) + ".json"))
    assert evidence["owned_process_group"] == 700 and evidence["automatic_retry"] is False
    assert evidence["peak_observed_process_group_rss_bytes"] == 13 * 1024 ** 3


def test_gpu_allocation_fraction_enforces_actual_gib_ceiling(campaign):
    folder, c, registry, _ = campaign
    configured = []
    native = SimpleNamespace(
        configure_runtime=lambda device, threads: configured.append((device, threads)),
        runtime_identity=lambda device: {"numpy": "native", "gpu": {"uuid": "native-gpu", "total_memory_bytes": 16 * 1024 ** 3}},
        torch=SimpleNamespace(cuda=SimpleNamespace(set_per_process_memory_fraction=lambda fraction, device: configured.append((fraction, device)))))
    proxy = e.FrozenImportProxy(SimpleNamespace(import_module=lambda name, package=None: native), c, folder, registry["rows"][0])
    proxy.import_module("paper.scripts.quantity_comparison_runtime").configure_runtime("cuda:0", 4)
    assert configured == [("cuda:0", 4), (.75, 0)]


def rewrite_registry(folder, c, registry):
    dump(folder / "evaluation_registry.json", registry)
    c["registry_sha256"] = e.sha(folder / "evaluation_registry.json")
    dump(folder / "execution_contract.json", c)


def test_dual_source_original_training_and_comparison_adoption_are_separate(campaign):
    folder, _, _, _ = campaign
    _, c, registry = e.load_campaign(folder, verify_bytes=False)
    rows = registry["rows"]
    assert {r["seed"] for r in rows} == {42, 52, 62}
    assert [e.row_source_closure(r) for r in rows].count(e.PARENT_SOURCE_SHA) == 3
    assert [e.row_source_closure(r) for r in rows].count(e.SOURCE_SHA) == 6
    assert e.read(folder / "comparison_gate.json")["scientific_adoption_gate"] is False
    assert c["source_closure_sha256"] == e.SOURCE_SHA


@pytest.mark.parametrize("change", ["bundle_swap", "new_bound_old_training", "old_bound_new_training",
    "new_revision", "original_job_host", "unknown_training", "source_roots_aliased", "new_source_closure"])
def test_wrong_original_lineage_rejected_before_imports(campaign, change):
    folder, c, registry, _ = campaign
    old = next(r for r in registry["rows"] if r["seed"] == 42)
    new = next(r for r in registry["rows"] if r["seed"] == 52)
    if change == "bundle_swap": new["evaluator_source_bundle"] = "cnn_gru_117"
    if change == "new_bound_old_training": new["original_binding"]["training_contract_sha256"] = e.PARENT_SHA
    if change == "old_bound_new_training": old["original_binding"]["training_contract_sha256"] = e.THREESEED_SHA
    if change == "new_revision": new["source_revision"] = old["source_revision"]
    if change == "original_job_host": new["job"] = {**new["job"], "host": "unapproved"}
    if change == "unknown_training": new["original_binding"]["training_contract_sha256"] = "0" * 64
    if change == "source_roots_aliased": registry["bundles"]["cnn_gru_123"]["source_root"] = "source117"
    if change == "new_source_closure": registry["bundles"]["cnn_gru_123"]["source_closure_sha256"] = e.PARENT_SOURCE_SHA
    rewrite_registry(folder, c, registry)
    with pytest.raises(ValueError): e.load_campaign(folder, verify_bytes=False)


@pytest.mark.parametrize("change", ["sha", "partial", "duplicate", "state", "checkpoint", "epoch",
    "scope", "heldout", "failed"])
def test_final_comparison_evidence_cannot_change_selected_checkpoint(campaign, change):
    folder, c, _, _ = campaign
    path = folder / "comparison_gate.json"
    gate = e.read(path)
    if change == "sha": c["final_validation_comparison_gate"]["sha256"] = "0" * 64
    if change == "partial": gate["rows"].pop()
    if change == "duplicate": gate["rows"][-1] = gate["rows"][0]
    if change == "state": gate["rows"][0]["state_tensor_sha256"] = "foreign"
    if change == "checkpoint": gate["rows"][0]["checkpoint_file_sha256"] = "foreign"
    if change == "epoch": gate["rows"][0]["selected_epoch"] += 1
    if change == "scope": gate["evaluation_scope"] = "test"
    if change == "heldout": gate["held_out_test_evaluated"] = True
    if change == "failed": gate["passed"] = False
    dump(path, gate)
    if change != "sha": c["final_validation_comparison_gate"]["sha256"] = e.sha(path)
    dump(folder / "execution_contract.json", c)
    with pytest.raises(ValueError): e.load_campaign(folder, verify_bytes=False)


@pytest.mark.parametrize("change", ["empty", "missing", "extra", "false", "integer", "closure", "bundle"])
def test_receipt_requires_exact_causal_proofs_and_row_source(campaign, change):
    folder, c, registry, _ = campaign
    row = next(r for r in registry["rows"] if r["seed"] == 52)
    value = receipt(folder, c, registry, row, "test")
    if change == "empty": value["qualification"] = {}
    if change == "missing": value["qualification"].pop("finite_outputs")
    if change == "extra": value["qualification"]["unknown"] = True
    if change == "false": value["qualification"]["parameters_unchanged"] = False
    if change == "integer": value["qualification"]["parameters_unchanged"] = 1
    if change == "closure": value["source_closure_sha256"] = e.PARENT_SOURCE_SHA
    if change == "bundle": value["evaluator_source_bundle"] = "cnn_gru_117"
    path = folder / "runs/test" / e.identity(row) / "receipt.json"
    dump(path, value)
    with pytest.raises(ValueError, match="qualification"): e.check_receipt(path, row, "test", c, folder)


@pytest.mark.parametrize("change", ["rmse", "mae", "nll", "bias", "count", "mass", "negative",
    "empty_nonzero", "cell_rmse", "bins", "boundary"])
def test_receipt_statistics_are_derived_from_full_population_sums(campaign, change):
    folder, c, registry, _ = campaign
    row = registry["rows"][0]
    value = receipt(folder, c, registry, row, "test")
    m = value["metrics"]
    if change == "rmse": m["qty_rmse"] += .1
    if change == "mae": m["qty_mae"] += .1
    if change == "nll": m["time_nll"] += .1
    if change == "bias": m["qty_bias"] += .1
    if change == "count": m["count"] = float(m["count"])
    if change == "mass": m["represented_n"] -= 1
    if change == "negative": m["time_nll_sum"] = -1
    if change == "empty_nonzero": m["quantity_cells"][1]["qty_sse"] = 1
    if change == "cell_rmse": m["quantity_cells"][0]["qty_rmse"] += .1
    if change == "bins": m["quantity_cells"][1]["bin"] = 0
    if change == "boundary": m["quantity_boundaries"] = [*m["quantity_boundaries"]]; m["quantity_boundaries"][0] += 1
    path = folder / "runs/test" / e.identity(row) / "receipt.json"
    dump(path, value)
    with pytest.raises(ValueError): e.check_receipt(path, row, "test", c, folder)


def test_worker_native_timeout_must_be_actual_child_of_pipeline(campaign):
    folder, c, registry, _ = campaign
    condition, token = e.identity(registry["rows"][0]), "1" * 64
    native = ["/usr/bin/timeout", "60", "/native/python", str(SCRIPT), "--mode", "worker"]
    owner = ["/native/python", str(SCRIPT), "--campaign", str(folder), "--mode", "pipeline"]
    dump(folder / "launches" / f"validation__{condition}.json", {
        "condition": condition, "split": "validation", "pipeline_pid": 123,
        "contract_sha256": e.sha(folder / "execution_contract.json"), "native_command": native,
        "token_sha256": e.hashlib.sha256(token.encode()).hexdigest(), "native_timeout_seconds": 60,
        "deadline_monotonic": e.time.monotonic() + 60})
    commands = {e.os.getppid(): native, 123: owner}
    with pytest.raises(ValueError, match="child of the admitted"):
        e.verify_worker_launch(folder, condition, "validation", c, token=token,
            process_reader=commands.__getitem__, process_parent_reader=lambda pid: 999)


def test_each_native_worker_inherits_its_original_source_revision(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    row = next(r for r in registry["rows"] if r["seed"] == 62)
    c["environment"] = {"SOURCE_REVISION": "wrong-parent-default"}
    monkeypatch.setattr(e.shutil, "which", lambda name: "/usr/bin/timeout")
    monkeypatch.setattr(e.time, "monotonic", lambda: 100.)
    captured = []
    monkeypatch.setattr(e.subprocess, "Popen", lambda command, **kw:
        captured.append(kw["env"]) or SimpleNamespace(returncode=0, poll=lambda: 0))
    monkeypatch.setattr(e, "check_receipt", lambda *a: {"verified": True})
    e.run_condition(folder, c, row, "validation", "cpu", 200.)
    assert captured[0]["SOURCE_REVISION"] == row["source_revision"]


def test_completion_records_preserve_per_row_source_and_exposure(campaign, monkeypatch):
    folder, c, registry, _ = campaign
    monkeypatch.setattr(e, "load_campaign", lambda _: (folder, c, registry))
    def run(campaign, c, row, split, device, deadline):
        path = folder / "runs" / split / e.identity(row) / "receipt.json"
        dump(path, receipt(folder, c, registry, row, split))
        return e.check_receipt(path, row, split, c, folder)
    completed = e.pipeline(folder, "cpu", runner=run)
    assert completed["test_exposure_lineage"] == c["test_exposure_lineage"]
    assert [r["source_closure_sha256"] for r in completed["records"]].count(e.PARENT_SOURCE_SHA) == 3
    assert [r["source_closure_sha256"] for r in completed["records"]].count(e.SOURCE_SHA) == 6


@pytest.mark.parametrize("change", ["sha", "scope", "incomplete", "wrong_contract", "wrong_conditions", "heldout", "mixed", "raw"])
def test_comparison_gate_is_bound_to_validation_only_original_receipt(campaign, change):
    folder, c, _, _ = campaign
    gate_path = folder / "comparison_gate.json"
    gate = e.read(gate_path)
    receipt_path = folder / gate["comparison_receipt_path"]
    receipt = e.read(receipt_path)
    if change == "sha": gate["comparison_receipt_sha256"] = "0" * 64
    if change == "scope": receipt["scope"] = "Validation_Test_mixed"
    if change == "incomplete": receipt["status"] = "pending"
    if change == "wrong_contract": receipt["frozen_contract_sha256"] = e.PARENT_SHA
    if change == "wrong_conditions": receipt["candidate_conditions"] = 6
    if change == "heldout": receipt["held_out_performance_read"] = True
    if change == "mixed": receipt["mixed_performance_file_read"] = True
    if change == "raw": receipt["raw_prediction_read"] = True
    dump(receipt_path, receipt)
    if change != "sha": gate["comparison_receipt_sha256"] = e.sha(receipt_path)
    dump(gate_path, gate)
    c["final_validation_comparison_gate"]["sha256"] = e.sha(gate_path)
    dump(folder / "execution_contract.json", c)
    with pytest.raises(ValueError, match="comparison receipt"): e.load_campaign(folder, verify_bytes=False)


@pytest.mark.parametrize("change", ["sha", "contract_copy", "untouched", "seed42", "head_refit", "replay_split"])
def test_prior_test_exposure_is_explicit_and_immutable(campaign, change):
    folder, c, _, _ = campaign
    path = folder / "evaluation_lineage.json"
    lineage = e.read(path)
    if change == "sha": c["evaluation_lineage_sha256"] = "0" * 64
    if change == "contract_copy": c["test_exposure_lineage"] = {}
    if change == "untouched": lineage["independent_untouched_Test"] = True
    if change == "seed42": lineage["seed42_checkpoint_Test_previously_evaluated"] = False
    if change == "head_refit": lineage["head_refits_Test_evaluated"] = True
    if change == "replay_split": lineage["fresh_replay_splits"] = ["test"]
    dump(path, lineage)
    if change not in ("sha", "contract_copy"):
        c["evaluation_lineage_sha256"], c["test_exposure_lineage"] = e.sha(path), lineage
    dump(folder / "execution_contract.json", c)
    with pytest.raises(ValueError): e.load_campaign(folder, verify_bytes=False)
