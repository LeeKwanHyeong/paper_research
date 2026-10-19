"""Execution packaging and preserved-condition checks without SSH or fitting."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from paper.scripts import manage_titantpp_shared_history_cap_execution as manager
from paper.scripts import prepare_titantpp_shared_history_cap as preparation


@pytest.fixture
def contract():
    prepared = json.loads((preparation.BUNDLE / "preparation_contract.json").read_text())
    parent = json.loads((preparation.PARENT / "execution_contract.json").read_text())
    observation = json.loads((manager.REPORT / "INITIAL_5080.json").read_text())
    source = deepcopy(prepared["source"])
    source["base_git_revision"] = source["git_revision"]
    return manager.build_contract(prepared, parent, observation, source)


def test_matrix_and_frozen_dataset_head_budget(contract):
    prepared = json.loads((preparation.BUNDLE / "preparation_contract.json").read_text())
    assert len(manager.runtime.validate_contract(contract, "5080")) == 18
    assert set(contract["hosts"]) == {"5080"}
    assert contract["datasets"] == prepared["datasets"]
    assert contract["training"] == prepared["training"]
    assert contract["baseline_validation_references"] == prepared["baseline_validation_references"]
    assert contract["common_condition"] == preparation.CONDITION
    assert contract["historical_baselines_retrained_or_reevaluated"] is False
    assert contract["held_out_test_authorized"] is False
    assert {j["arm"] for j in contract["jobs"]} == set(preparation.ARMS)


def test_single_gpu_queue_order_and_limits(contract):
    assert contract["maximum_parallel_workers"] == 3
    assert [j["seed"] for j in contract["jobs"][:3]] == [42, 52, 62]
    assert len({(j["dataset"], j["arm"]) for j in contract["jobs"][:3]}) == 1
    assert contract["hosts"]["5080"]["absolute_host_deadline_unix"] == 1792888885.911408
    assert contract["limits"]["fit_wall_seconds"] == 129600


def test_failed_predecessor_never_becomes_idle(contract):
    prepared = json.loads((preparation.BUNDLE / "preparation_contract.json").read_text())
    parent = json.loads((preparation.PARENT / "execution_contract.json").read_text())
    observation = json.loads((manager.REPORT / "INITIAL_5080.json").read_text())
    observation["observation"]["predecessor_terminal"]["dispatcher_exit.json"] = json.dumps({"actual_os_returncode": 1, "pid": 464187})
    with pytest.raises(ValueError): manager.build_contract(prepared, parent, observation, contract["source"])


def test_archive_contains_empty_root_sentinel_and_only_six_inputs(tmp_path, contract):
    source = tmp_path / "repo"; source.mkdir()
    bundle = tmp_path / "bundle"; bundle.mkdir()
    (source / "a.py").write_text("pass\n")
    (bundle / "execution_contract.json").write_text("{}")
    for name in contract["input_files"]:
        p = bundle / "input_snapshot" / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"fabricated")
    with patch.object(manager._impl, "ROOT", source), patch.object(manager._impl, "BUNDLE", bundle):
        output = manager._impl.package_archive(contract, {"files": {"a.py": hashlib.sha256(b"pass\n").hexdigest()}})
    import tarfile
    with tarfile.open(output) as archive:
        members = archive.getmembers(); names = {p.name for p in members}
        assert "source/sample_data" in names
        assert next(p for p in members if p.name == "source/sample_data").isdir()
        assert {n for n in names if n.startswith("source/inputs/")} == {"source/" + n for n in contract["input_files"]}
        assert not any("test" in n.lower() for n in names)


def test_collection_uses_only_new_campaign_runtime(contract):
    text = manager.collection_program(contract, "5080")
    compile(text, "<collection>", "exec")
    assert "import titantpp_shared_history_cap_execution_runtime as r" in text
    assert "import titantpp_quantity_loss_mlp18_runtime as r" not in text
    assert "actual_os_returncode" in text and "owned_pids_absent" in text
    assert "archive.add" in text


def test_no_deploy_if_local_source_verification_fails():
    with patch.object(manager, "verify_local", side_effect=ValueError("source changed")), patch.object(manager._impl, "ssh") as remote:
        with pytest.raises(ValueError): manager.deploy()
        remote.assert_not_called()
