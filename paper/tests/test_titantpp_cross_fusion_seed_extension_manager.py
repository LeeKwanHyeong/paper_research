"""Local synthetic contracts and archives only; never contact either GPU host."""
import tarfile

import pytest

from paper.scripts import manage_titantpp_cross_fusion_seed_extension as manager


@pytest.fixture
def prepared_fixture(tmp_path, monkeypatch):
    root = tmp_path / "project"
    parent = root / "search_artifacts/parent"
    parent_report = root / "reports/parent"
    bundle = root / "search_artifacts/extension"
    report = root / "reports/extension"
    for name, value in (("ROOT", root), ("PARENT", parent), ("PARENT_REPORT", parent_report),
                        ("BUNDLE", bundle), ("REPORT", report)):
        monkeypatch.setattr(manager, name, value)
    root.mkdir()
    blobs = {"models/parent_model.py": b"# inherited scientific model\n",
             "inputs/data/raw/train_validation.parquet": b"synthetic raw input bytes"}
    for relative, blob in blobs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    for name in manager.NEW_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic new operation module\n")
    source = {"models/parent_model.py": manager.sha(root / "models/parent_model.py")}
    dataset_ids = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
    assignments = (("5080", "5080"), ("5090", "5090"), ("5080", "5090"))
    jobs = []
    for dataset, hosts in zip(dataset_ids, assignments):
        for arm, host in zip(manager.ARMS, hosts):
            jobs.append({"dataset": dataset, "arm": arm, "host": host, "seed": 42,
                         "id": f"{dataset}__{arm}__seed42"})
    contract = {
        "schema": "synthetic_parent", "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "datasets": [{"dataset_id": d} for d in dataset_ids],
        "source": {"base_git_revision": "parent-commit", "files": source, "files_sha256": manager.canonical(source)},
        "input_files": {"inputs/data/raw/train_validation.parquet": manager.sha(root / "inputs/data/raw/train_validation.parquet")},
        "architecture": {"arms": list(manager.ARMS)},
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
                     "batch_size": 128, "monitor": "validation_raw_quantity_rmse"},
        "hosts": {h: {"root": f"/synthetic/parent_{h}", "source_root": f"/synthetic/parent_{h}/source",
                      "python": "/synthetic/python", "environment": {}, "gpu_uuid": f"GPU-{h}",
                      "tmux_binary": "/synthetic/tmux", "runtime_expected": {}}
                  for h in ("5080", "5090")},
        "jobs": jobs,
    }
    manager.write(parent / "execution_contract.json", contract)
    monkeypatch.setattr(manager, "PARENT_SHA", manager.sha(parent / "execution_contract.json"))
    with tarfile.open(parent / "deployment.tar.gz", "w:gz") as archive:
        for relative in blobs:
            archive.add(root / relative, arcname="source/" + relative)
        archive.add(parent / "execution_contract.json", arcname="execution_contract.json")
    manager.write(parent_report / "PREPARED.json", {"archive_sha256": manager.sha(parent / "deployment.tar.gz")})
    for host in ("5080", "5090"):
        manager.write(parent_report / "observations" / f"20260101_{host}.json", {
            "qualification.json": {"status": "passed", "host": host, "contract_sha256": manager.canonical(contract)},
            "start.json": {"qualification_sha256": f"qualification-{host}"},
        })
    refs = []
    for dataset in dataset_ids:
        for seed in (42, 52, 62):
            endpoint = root / "references" / f"{dataset}_{seed}.json"
            manager.write(endpoint, {"validation_rmse": 1.0, "validation_mae": .5})
            refs.append({"dataset": dataset, "seed": seed, "endpoint_path": str(endpoint),
                         "endpoint_sha256": manager.sha(endpoint)})
    refs_path = root / "search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1/width16_validation_refs.json"
    manager.write(refs_path, {"rows": refs})
    monkeypatch.setattr(manager, "MLP_REFS_SHA", manager.sha(refs_path))
    def git_only(command, **kwargs):
        if command == ["git", "status", "--porcelain"]:
            return b""
        if command == ["git", "rev-parse", "HEAD"]:
            return "extension-commit\n"
        raise AssertionError(f"Unexpected subprocess: {command}")
    monkeypatch.setattr(manager.subprocess, "check_output", git_only)
    monkeypatch.setattr(manager, "ssh", lambda *args, **kwargs: pytest.fail("Unexpected remote call"))
    return {"root": root, "parent": parent, "bundle": bundle, "report": report,
            "parent_contract": contract, "blobs": blobs}


def test_prepare_twelve_jobs_preserves_assignments_raw_inputs_and_baselines(prepared_fixture):
    f = prepared_fixture
    receipt = manager.prepare()
    contract = manager.verify_local()
    assert len(contract["jobs"]) == 12
    assert {j["seed"] for j in contract["jobs"]} == {52, 62}
    assert all(sum(j["host"] == h for j in contract["jobs"]) == 6 for h in ("5080", "5090"))
    original = {(j["dataset"], j["arm"]): j["host"] for j in f["parent_contract"]["jobs"]}
    assert all(original[j["dataset"], j["arm"]] == j["host"] for j in contract["jobs"])
    assert contract["input_files"] == f["parent_contract"]["input_files"]
    assert all("Cap" not in name for name in contract["input_files"])
    assert contract["architecture"] == f["parent_contract"]["architecture"]
    assert contract["training"] == f["parent_contract"]["training"]
    assert receipt["historical_baseline_retrained"] is False
    assert receipt["held_out_test_evaluated"] is False
    assert receipt["cap_data_used"] is False
    assert len(manager.read(f["bundle"] / "stored_mlp16_validation_references.json")) == 9


def test_archive_inherits_exact_bytes_and_adds_only_two_operation_files(prepared_fixture):
    f = prepared_fixture
    before = {path: path.read_bytes() for path in f["parent"].iterdir() if path.is_file()}
    manager.prepare()
    with tarfile.open(f["bundle"] / "deployment.tar.gz") as archive:
        names = archive.getnames()
        assert set(names) == {"execution_contract.json", *["source/" + name for name in f["blobs"]],
                              *["source/" + name for name in manager.NEW_FILES]}
        assert len(names) == len(set(names))
        for name, blob in f["blobs"].items():
            assert archive.extractfile("source/" + name).read() == blob
        for name in manager.NEW_FILES:
            assert archive.extractfile("source/" + name).read() == (f["root"] / name).read_bytes()
    assert all(path.read_bytes() == blob for path, blob in before.items())


def test_prepare_rejects_changed_parent_before_creating_bundle(prepared_fixture):
    path = prepared_fixture["parent"] / "execution_contract.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Seed42 contract changed"):
        manager.prepare()
    assert not prepared_fixture["bundle"].exists()


def test_prepare_refuses_existing_bundle(prepared_fixture):
    folder = prepared_fixture["bundle"]
    folder.mkdir()
    marker = folder / "preserve.txt"
    marker.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        manager.prepare()
    assert marker.read_text() == "existing evidence"


@pytest.mark.parametrize("target", ["contract", "operation"])
def test_verify_local_rejects_post_prepare_mutation(prepared_fixture, target):
    manager.prepare()
    path = (prepared_fixture["bundle"] / "execution_contract.json" if target == "contract"
            else prepared_fixture["root"] / manager.NEW_FILES[0])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        manager.verify_local()


def test_deploy_duplicate_attempt_stops_before_remote_call(prepared_fixture):
    manager.prepare()
    marker = prepared_fixture["bundle"] / "attempts/5080.json"
    manager.write(marker, {"previous": "preserve"})
    before = marker.read_bytes()
    with pytest.raises(FileExistsError):
        manager.deploy_start("5080")
    assert marker.read_bytes() == before


def test_deploy_rejects_changed_archive_before_attempt_or_remote(prepared_fixture):
    manager.prepare()
    archive = prepared_fixture["bundle"] / "deployment.tar.gz"
    with archive.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="archive changed"):
        manager.deploy_start("5090")
    assert not (prepared_fixture["bundle"] / "attempts").exists()
