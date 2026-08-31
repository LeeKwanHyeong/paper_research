"""Fail-closed tests for the authorized 5090-only stable MAC stage."""

import argparse
import copy
import json
from pathlib import Path
import sys

import pytest

from paper.scripts import run_titantpp_mac_stable_5090 as launcher
from paper.scripts.validate_titantpp_mac_stability_preflight import check_header, CONTEXTS

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "paper/contracts/count_aware_titantpp_mac_stable_seed52_5090_v1.json"


def contract():
    return json.loads(CONTRACT.read_text())


def test_only_5090_seed52_and_complete_context_grid():
    c = contract()
    launcher.validate_contract(c)
    for key, value in [("execution_server", "5080"), ("seed", 62),
                       ("other_hosts_authorized", True), ("other_seeds_authorized", True)]:
        bad = copy.deepcopy(c)
        bad["authorization"][key] = value
        with pytest.raises(ValueError):
            launcher.validate_contract(bad)
    bad = copy.deepcopy(c)
    bad["context_preflight_order"].pop()
    with pytest.raises(ValueError):
        launcher.validate_contract(bad)


@pytest.mark.parametrize("key,value", [("inner_gradient_clip", None), ("epochs", 40),
                                      ("time_scale", 1.), ("lambda_tail", .1)])
def test_contract_rejects_training_drift(key, value):
    c = contract()
    c["training"][key] = value
    with pytest.raises(ValueError):
        launcher.validate_contract(c)


def test_source_checksums_are_still_the_qualified_training_source():
    assert launcher.verify_source(ROOT, contract())["status"] == "complete"


@pytest.mark.parametrize("gpu,compute,gdm,names", [
    ("RTX 5090, 100", "", "inactive", ""),
    ("RTX 5080, 30000", "", "inactive", ""),
    ("RTX 5090, 15800", "123", "inactive", ""),
    ("RTX 5090, 15800", "", "active", ""),
    ("RTX 5090, 15800", "", "inactive", "python\nXwayland"),
])
def test_gpu_rejects_busy_wrong_or_desktop_runtime(gpu, compute, gdm, names):
    with pytest.raises(ValueError):
        launcher.check_gpu(gpu, compute, gdm, names)


def test_empty_5090_is_accepted_and_old_artifacts_are_not_overwritten(tmp_path):
    launcher.check_gpu("NVIDIA GeForce RTX 5090, 15801", "", "inactive", "python")
    old = tmp_path / "artifact"
    launcher.ensure_fresh(old)
    (old / "status.json").write_text('{"state":"failed"}')
    with pytest.raises(FileExistsError):
        launcher.ensure_fresh(old)
    assert json.loads((old / "status.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("dataset", launcher.VALIDATION_ORDER)
@pytest.mark.parametrize("phase", ["context_e1", "validation_e300"])
def test_commands_are_fresh_full_data_with_explicit_clip(dataset, phase, tmp_path):
    _, cmd = launcher.training_command(ROOT, tmp_path, contract(), dataset, phase, sys.executable)
    assert cmd[cmd.index("--titans-memory-gradient-clip") + 1] == "1"
    assert cmd[cmd.index("--seeds") + 1] == "52"
    assert cmd[cmd.index("--source-revision") + 1] == launcher.REVISION
    assert cmd[cmd.index("--epochs") + 1] == ("1" if phase == "context_e1" else "300")
    assert cmd[cmd.index("--max-seq-len") + 1] == str(CONTEXTS[dataset][1])
    assert cmd[cmd.index("--recompile-limit") + 1] == "64"
    assert not {"--resume", "--force-rerun", "--max-train-batches", "--max-val-batches"} & set(cmd)


def test_all_context_headers_and_e300_minimum_epochs():
    from simple_lab_test.search.tests.test_titans_stability_preflight_gate import evidence
    for dataset, (lookback, length) in CONTEXTS.items():
        c, s, h = evidence()
        c.update(dataset=dataset, lookback_weeks=lookback, max_seq_len=length)
        check_header(c, s, h, seed=42, revision="a"*40, dataset=dataset)
        c["epochs"] = 300
        with pytest.raises(ValueError, match="full epochs"):
            check_header(c, s, h, seed=42, revision="a"*40,
                         dataset=dataset, epochs=300, min_epochs=40)
        s["completed_epochs"] = 40
        history = [{**h[0], "epoch": e} for e in range(1, 41)]
        check_header(c, s, history, seed=42, revision="a"*40,
                     dataset=dataset, epochs=300, min_epochs=40)
        history[-1]["train_all_finite"] = False
        with pytest.raises(ValueError):
            check_header(c, s, history, seed=42, revision="a"*40,
                         dataset=dataset, epochs=300, min_epochs=40)


def test_child_failure_records_failed_and_never_starts_next_run(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    c = contract()
    cp = project / "contract.json"
    cp.write_text(json.dumps(c))
    for path in (launcher.VALIDATOR,
                 "paper/scripts/validate_count_aware_titantpp_mac_three_seed_validation.py",
                 c["existing_instacart_gate"]):
        p = project / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    calls = []
    monkeypatch.setattr(launcher, "verify_inputs", lambda *_: None)
    monkeypatch.setattr(launcher, "gpu_preflight", lambda: {"gdm": "inactive"})
    def command(project, output, contract, dataset, phase, python):
        calls.append((dataset, phase))
        return output / phase / dataset / "seed_52", [python, "-c", "raise SystemExit(3)"]
    monkeypatch.setattr(launcher, "training_command", command)
    args = argparse.Namespace(project_root=project, output_root=tmp_path / "output",
                              contract=cp, verify_only=False,
                              orchestration_revision="a"*40, python=sys.executable)
    with pytest.raises(RuntimeError, match="Stage exited 3"):
        launcher.execute(args)
    status = json.loads((args.output_root / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["child_pid"] is None and status["completed_run_count"] == 0
    assert calls == [("raf_spare_parts", "context_e1")]


def test_server_split_preserves_all_training_arguments_and_source():
    from paper.scripts import run_titantpp_mac_stable_5080 as original
    old = json.loads((ROOT / "paper/contracts/count_aware_titantpp_mac_stable_seed62_5080_v1.json").read_text())
    new = contract()
    for key in ("training", "datasets", "frozen_training_file_sha256", "runtime",
                "training_source_revision", "context_preflight_order", "validation_order"):
        assert old[key] == new[key]
    for dataset in launcher.VALIDATION_ORDER:
        for phase in ("context_e1", "validation_e300"):
            _, a = original.training_command(ROOT, Path("/tmp/split"), old, dataset, phase, sys.executable)
            _, b = launcher.training_command(ROOT, Path("/tmp/split"), new, dataset, phase, sys.executable)
            normalized = [arg.replace("stable_5080_", "stable_5090_").replace("seed62", "seed52")
                          .replace("seed_62", "seed_52") for arg in a]
            normalized[normalized.index("--seeds") + 1] = "52"
            assert normalized == b


def test_instacart_gate_cannot_reuse_the_other_seed(tmp_path, monkeypatch):
    c = contract()
    proof = json.loads((ROOT / c["existing_instacart_gate"]).read_text())
    monkeypatch.setattr(launcher, "verify_source", lambda *_: {"status": "complete"})
    hashes = {}
    for ds in c["datasets"].values():
        hashes[str(tmp_path / ds["data_path"])] = ds["data_sha256"]
        hashes[str(tmp_path / ds["split_manifest_path"])] = ds["split_manifest_sha256"]
    monkeypatch.setattr(launcher, "sha256_file", lambda p: hashes[str(p)])
    monkeypatch.setattr(launcher, "load_json", lambda _: proof)
    launcher.verify_inputs(tmp_path, c)
    proof["seed"] = 62
    with pytest.raises(ValueError, match="gate is missing"):
        launcher.verify_inputs(tmp_path, c)
