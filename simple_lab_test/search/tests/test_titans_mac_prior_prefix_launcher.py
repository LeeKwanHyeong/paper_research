"""Operational contracts; these tests do not allocate CUDA or train models."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "paper/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_script("run_titans_mac_prior_prefix_5090")
packager = load_script("package_titans_mac_prior_prefix")
REVISION = "a" * 40
CONTRACT_HASH = "b" * 64
MANIFEST_HASH = "c" * 64


@pytest.fixture
def contract():
    return launcher.read(ROOT / "paper/contracts/titans_mac_prior_prefix_v1.json")


def argument(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize("phase,count,seeds", [("cuda", 0, set()), ("e1", 6, {42}),
                                               ("screening", 6, {42}), ("confirm", 12, {52, 62})])
def test_fresh_pair_commands_keep_shared_training_contract(contract, phase, count, seeds):
    args = argparse.Namespace(phase=phase, data_project=Path("/data"),
                              output_root=Path("/fresh"), source_revision=REVISION)
    plan = launcher.run_plan(args, contract)
    assert len(plan) == count
    assert {row[2] for row in plan} == seeds
    assert list(dict.fromkeys(row[0] for row in plan)) == (list(launcher.DATASETS) if count else [])
    for dataset, backbone, seed, output, command in plan:
        assert output == Path("/fresh") / dataset / backbone / f"seed_{seed}"
        expected = {"--device": "cuda", "--time-head-mode": "legacy_clamped_rmtpp",
                    "--time-scale": "3", "--time-w-max": str(10 / 3),
                    "--time-intercept-limit": "300", "--grad-clip": "1",
                    "--titans-memory-gradient-clip": "1", "--titans-mac-execution-backend": "optimized",
                    "--quantity-variants": "log_mse", "--lambda-tail": "0", "--lambda-log-qty": "1",
                    "--batch-size": "128", "--lr": "0.001", "--source-revision": REVISION,
                    "--epochs": "1" if phase == "e1" else "300",
                    "--min-epochs": "1" if phase == "e1" else "40",
                    "--early-stopping-patience": "1" if phase == "e1" else "40"}
        assert {name: argument(command, name) for name in expected} == expected
        assert argument(command, "--model-role") == (
            "experimental" if backbone == launcher.BASELINE else "titans_mac_prior_prefix_screening")
        assert not ({"--resume", "--force-rerun", "--max-train-batches", "--max-val-batches", "--max-series"}
                    & set(command))
    for index in range(0, len(plan), 2):
        baseline, candidate = plan[index:index + 2]
        assert baseline[0] == candidate[0] and baseline[2] == candidate[2]
        for key in ("--data", "--split-manifest", "--max-seq-len", "--lookback-weeks"):
            assert argument(baseline[-1], key) == argument(candidate[-1], key)


@pytest.mark.parametrize("key,value", [("time_intercept_limit", 30.), ("execution_backend", "reference"),
                                      ("batch_size", 2), ("lambda_tail", 1.), ("resume", True)])
def test_contract_drift_rejected(contract, key, value):
    launcher.validate_contract(contract)
    changed = copy.deepcopy(contract)
    changed["training"][key] = value
    with pytest.raises(ValueError, match="Frozen training"):
        launcher.validate_contract(changed)


def test_dry_run_does_not_allocate_gpu_or_create_output(contract, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not inspect CUDA, source artifacts or data")
    monkeypatch.setattr(launcher, "gpu_preflight", forbidden)
    monkeypatch.setattr(launcher, "verify_source", forbidden)
    monkeypatch.setattr(launcher, "verify_data", forbidden)
    args = argparse.Namespace(phase="screening", dry_run=True,
                              contract=ROOT / "paper/contracts/titans_mac_prior_prefix_v1.json",
                              output_root=tmp_path / "absent", data_project=Path("/absent"),
                              source_revision=REVISION)
    result = launcher.execute(args)
    assert result["training_started"] is False and result["run_count"] == 6
    assert not args.output_root.exists()


@pytest.mark.parametrize("compute,gdm,names", [("99", "inactive", ""), ("", "active", ""),
                                            ("", "inactive", "gnome-shell"), ("", "inactive", "Xwayland")])
def test_gpu_guard_rejects_occupied_runtime(compute, gdm, names):
    with pytest.raises(ValueError):
        launcher.check_gpu("NVIDIA GeForce RTX 5090, 30000", compute, gdm, names)


def test_gpu_guard_requires_single_5090_and_capacity():
    launcher.check_gpu("NVIDIA GeForce RTX 5090, 12000", "", "inactive", "python\nsshd")
    for value in ("NVIDIA GeForce RTX 5080, 30000", "NVIDIA GeForce RTX 5090, 11999",
                  "NVIDIA GeForce RTX 5090, 30000\nNVIDIA GeForce RTX 5090, 30000"):
        with pytest.raises(ValueError):
            launcher.check_gpu(value, "", "inactive", "")


def proof_fixture(tmp_path, phase):
    evidence = tmp_path / "audit.json"
    evidence.write_text("{}\n")
    proof = {"status": "PASS", "phase": phase, "source_revision": REVISION,
             "contract_sha256": CONTRACT_HASH, "source_manifest_sha256": MANIFEST_HASH,
             "held_out_test_evaluated": False, "run_root": str(tmp_path.resolve()),
             "evidence_files": {str(evidence): launcher.digest(evidence)},
             "tests": {"tests": 35, "failures": 0, "errors": 0, "skipped": 0},
             "audit": {"status": "PASS", "device": "cuda", "cuda_available": True,
                       "cost_gate_pass": True, "screening_pass": True},
             "runs": [{"dataset": d, "backbone": b, "seed": 42, "status": "PASS"}
                      for d in launcher.DATASETS for b in launcher.BACKBONES]}
    path = tmp_path / "status.json"
    launcher.save(path, proof)
    return path, proof, evidence


@pytest.mark.parametrize("phase", ["cuda", "e1", "screening"])
def test_phase_proof_requires_exact_source_contract_and_unchanged_evidence(tmp_path, phase):
    path, proof, evidence = proof_fixture(tmp_path, phase)
    assert launcher.verify_proof(path, phase, REVISION, CONTRACT_HASH, MANIFEST_HASH) == proof
    for key, bad in (("source_revision", "d" * 40), ("contract_sha256", "e" * 64),
                     ("source_manifest_sha256", "f" * 64), ("held_out_test_evaluated", True),
                     ("status", "RUNNING"), ("run_root", "/different")):
        launcher.save(path, {**proof, key: bad})
        with pytest.raises(ValueError):
            launcher.verify_proof(path, phase, REVISION, CONTRACT_HASH, MANIFEST_HASH)
    launcher.save(path, proof)
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="evidence changed"):
        launcher.verify_proof(path, phase, REVISION, CONTRACT_HASH, MANIFEST_HASH)


@pytest.mark.parametrize("phase", ["e1", "screening"])
def test_phase_gate_rejects_incomplete_or_duplicate_pair_grid(tmp_path, phase):
    path, proof, _ = proof_fixture(tmp_path, phase)
    proof["runs"][-1] = dict(proof["runs"][0])
    launcher.save(path, proof)
    with pytest.raises(ValueError):
        launcher.verify_proof(path, phase, REVISION, CONTRACT_HASH, MANIFEST_HASH)


def test_cuda_proof_rejects_skips_and_cost_failure(tmp_path):
    path, proof, _ = proof_fixture(tmp_path, "cuda")
    for field in ("skipped", "cost"):
        bad = copy.deepcopy(proof)
        if field == "skipped":
            bad["tests"]["skipped"] = 1
        else:
            bad["audit"]["cost_gate_pass"] = False
        launcher.save(path, bad)
        with pytest.raises(ValueError):
            launcher.verify_proof(path, "cuda", REVISION, CONTRACT_HASH, MANIFEST_HASH)


@pytest.mark.parametrize("extra,count", [("", 2), ("<skipped/>", 2), ("<failure/>", 2), ("", 0)])
def test_junit_requires_actual_non_skipped_cuda_contract_tests(tmp_path, extra, count):
    path = tmp_path / "result.xml"
    path.write_text(f'<testsuites><testsuite tests="{count}" failures="0" errors="0" skipped="0">'
                    '<testcase classname="simple_lab_test.search.tests.test_titans_mac_prior_prefix_contract" '
                    f'name="test_cuda">{extra}</testcase>'
                    '<testcase classname="simple_lab_test.search.tests.test_titans_mac_train_validation_entrypoint" '
                    'name="test_actual_main"/></testsuite></testsuites>')
    if extra or count != 2:
        with pytest.raises(ValueError):
            launcher.audit_xml(path)
    else:
        audit = launcher.audit_xml(path)
        assert audit["tests"] == 2
        assert audit["modules"] == {Path(test).stem: 1 for test in launcher.TESTS}


@pytest.mark.parametrize("remaining", launcher.TESTS)
def test_junit_rejects_missing_either_required_module(tmp_path, remaining):
    path = tmp_path / "incomplete.xml"
    path.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
                    f'<testcase classname="simple_lab_test.search.tests.{Path(remaining).stem}" '
                    'name="test_present"/></testsuite></testsuites>')
    with pytest.raises(ValueError, match="Both mandatory CUDA test modules"):
        launcher.audit_xml(path)


def test_committed_archive_excludes_dirty_untracked_and_root_scripts(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c",
                                        "user.email=fixture@example.invalid", *args], cwd=project, text=True).strip()
    git("init", "-q")
    for name in (*packager.REQUIRED, "models/example.py", "scripts/do_not_package.py"):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("committed\n")
    git("add", ".")
    git("commit", "-q", "-m", "isolated test fixture")
    revision = git("rev-parse", "HEAD")
    (project / "models/example.py").write_text("uncommitted edit\n")
    (project / "models/untracked.py").write_text("untracked\n")
    first, second = tmp_path / "one.tar", tmp_path / "two.tar"
    report = packager.package(project, revision, first)
    repeated = packager.package(project, revision, second)
    assert report["package_sha256"] == repeated["package_sha256"]
    with tarfile.open(first) as archive:
        assert archive.getmember("sample_data").isdir()
        assert archive.extractfile("models/example.py").read() == b"committed\n"
        assert "models/untracked.py" not in archive.getnames()
        assert not any(name.startswith("scripts/") for name in archive.getnames())
        manifest = json.loads(archive.extractfile("source_manifest.json").read())
        assert manifest["source_revision"] == revision
        for name, expected in manifest["files"].items():
            assert packager.digest_bytes(archive.extractfile(name).read()) == expected
    with pytest.raises(FileExistsError):
        packager.package(project, revision, first)


def test_package_rejects_unsafe_paths_and_uncommitted_revision():
    assert packager.selected_source("models/model.py")
    for path in ("/models/evil.py", "models/../evil.py", "scripts/train.py", "models/__pycache__/x.pyc",
                 "paper/results/unrelated.json", "models/cache.pyc"):
        assert not packager.selected_source(path)
    with pytest.raises(ValueError, match="full committed"):
        packager.committed_files(ROOT, "HEAD")


def test_source_manifest_binds_runtime_files_contract_and_frozen_references(tmp_path):
    (tmp_path / "sample_data").mkdir()
    required = (*launcher.TESTS, launcher.CUDA_VALIDATOR, launcher.COMPARATOR, launcher.TRAINER,
                launcher.POLICY, launcher.DIAGNOSTIC, "models/TPPs/CountAwareFactory.py",
                "models/Titan/common/titans_mac.py", "paper/scripts/run_titans_mac_prior_prefix_5090.py",
                "paper/contracts/titans_mac_prior_prefix_v1.json",
                "paper/results/titans_mac_prior_prefix_20260905/frozen_references.json")
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    contract_path = tmp_path / "paper/contracts/titans_mac_prior_prefix_v1.json"
    contract_path.write_text(json.dumps({"frozen_references": {"path": required[-1],
                                           "sha256": launcher.digest(tmp_path / required[-1])}}))
    manifest = {"contract_id": launcher.CONTRACT_ID, "source_revision": REVISION,
                "origin": "git_archive_committed_tree", "root_scripts_excluded": True,
                "scaffold_directories": ["sample_data"],
                "files": {name: launcher.digest(tmp_path / name) for name in required}}
    manifest_path = tmp_path / "source_manifest.json"
    launcher.save(manifest_path, manifest)
    assert launcher.verify_source(manifest_path, REVISION, contract_path, root=tmp_path)
    (tmp_path / launcher.TRAINER).write_text("changed")
    with pytest.raises(ValueError, match="Source file changed"):
        launcher.verify_source(manifest_path, REVISION, contract_path, root=tmp_path)
