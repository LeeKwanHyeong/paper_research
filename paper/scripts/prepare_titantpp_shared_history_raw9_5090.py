"""Move the unstarted raw9 campaign to the explicitly approved idle5090."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import tarfile

from paper.scripts import prepare_titantpp_shared_history_raw9 as previous

ROOT, NAME = previous.ROOT, previous.NAME
BUNDLE, REPORT = previous.BUNDLE / "execution_5090_v1", previous.REPORT / "execution_5090_v1"
HOST_PARENT = ROOT / "search_artifacts/titantpp_quantity_loss_mlp18_20261019_v2/execution_contract.json"
CANCELLATION = previous.REPORT / "queue_v2/CANCELLED_FOR_5090.json"
NEW_SOURCE = (
    "paper/scripts/prepare_titantpp_shared_history_raw9_5090.py",
    "paper/scripts/manage_titantpp_shared_history_raw9_5090.py",
    "paper/scripts/titantpp_shared_history_raw9_5090_runtime.py",
    "paper/scripts/run_titantpp_shared_history_raw9_5090.py",
    "paper/scripts/compare_titantpp_shared_history_raw9_5090.py",
)
read, sha, canonical, write, require = previous.read, previous.sha, previous.canonical, previous.write, previous.require
baseline_document = previous.baseline_document


def migration_gate():
    cancellation = read(CANCELLATION)
    require(cancellation["status"] == "cancelled_before_any_scientific_or_remote_attempt"
        and cancellation["observed_pid_absence"] is True and cancellation["raw9_remote_attempt_count"] == 0
        and cancellation["raw9_fit_count"] == 0 and cancellation["replacement_host"] == "5090"
        and cancellation["Cap_training_modified"] is False, "5080 successor was not safely cancelled")
    require(not (previous.BUNDLE / "attempts/5080.json").exists()
        and not (previous.BUNDLE / "retrieval/5080").exists()
        and not (previous.REPORT / "queue_v2/LAUNCHED_5080.json").exists(), "A5080 scientific attempt exists")
    initial = read(REPORT / "INITIAL_5090.json")
    expected = read(HOST_PARENT)["hosts"]["5090"]["runtime_expected"]
    require(initial["host"] == "5090" and initial["gpu"]["actual_exit"] == 0
        and initial["gpu_processes"]["actual_exit"] == 0 and not initial["gpu_processes"]["stdout"].strip()
        and initial["new_root_exists"] is False and all(initial["runtime"][k] == v for k, v in expected.items()),
        "Initial5090 GPU/Runtime did not pass")
    require(all(not present for present in initial["old_paused_process_paths_present"].values()),
        "Historical5090 processes must not be resumed or silently adopted")
    return cancellation, initial


def prepare():
    require(not BUNDLE.exists(), "Preserve existing5090 bundle; no overwrite/retry")
    original = previous.verify(archive=True)
    cancellation, initial = migration_gate()
    tracked = set(subprocess.check_output(["git", "ls-files", "--", *NEW_SOURCE], cwd=ROOT, text=True).splitlines())
    require(tracked == set(NEW_SOURCE) and not subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *NEW_SOURCE], cwd=ROOT, text=True).strip(), "Commit new5090 adapters before sealing")
    c = deepcopy(original)
    files = {**original["source"]["files"], **{name: sha(ROOT / name) for name in NEW_SOURCE}}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, digest in files.items():
        require(sha(ROOT / name) == digest, "Preserved scientific source changed: " + name)
    host_parent = read(HOST_PARENT)
    h = deepcopy(host_parent["hosts"]["5090"])
    old_source = h["source_root"]
    remote = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{NAME}_execution_5090_v1"
    for key in ("completed_predecessor", "wait_predecessor", "wait_for_parent", "paused_receipt_sha256", "paused_completeness_sha256"):
        h.pop(key, None)
    h.update(root=remote, source_root=remote + "/source", tmux="shared_history_raw9_5090_20261020_v1",
        preflight_source_root=old_source, paused_processes=[])
    h["environment"].update(SOURCE_REVISION=revision, PYTHONPATH=h["source_root"],
        MPLCONFIGDIR=remote + "/cache/matplotlib", XDG_CACHE_HOME=remote + "/cache")
    c.update(schema="titantpp_shared_history_raw9_5090_v1", campaign=NAME + "/execution_5090_v1",
        created_at=datetime.now(timezone.utc).isoformat(), hosts={"5090": h},
        source={"files": files, "files_sha256": canonical(files), "base_git_revision": revision,
            "preserved_parent_file_count": len(original["source"]["files"])},
        migration={"former_contract_path": str(previous.BUNDLE / "execution_contract.json"),
            "former_contract_raw_sha256": sha(previous.BUNDLE / "execution_contract.json"),
            "cancellation_path": str(CANCELLATION), "cancellation_sha256": sha(CANCELLATION),
            "initial5090_path": str(REPORT / "INITIAL_5090.json"), "initial5090_sha256": sha(REPORT / "INITIAL_5090.json"),
            "host_parent_path": str(HOST_PARENT), "host_parent_sha256": sha(HOST_PARENT),
            "5080_Cap_dependency_superseded_by_user": True, "science_conditions_changed": False},
        approval="2026-10-20 user explicitly requested starting the authorized original sharedMLP9 on idle5090 now; cancel only5080pending successor, keep5080Cap running; same model/rawinputs/alpha1/head/budget/seeds/selection; max3parallel")
    for job in c["jobs"]:
        job["host"] = "5090"
    spec = deepcopy(read(previous.SPEC))
    spec.update(training_host="5090", predecessor_required="No wait for5080 Cap; cancelled5080pending successor and fresh idle5090 admission required",
        host_migration_only=True, existing_Cap5080_training_modified=False)
    write(REPORT / "SCIENCE_SPEC.json", spec)
    c["science_spec_sha256"] = sha(REPORT / "SCIENCE_SPEC.json")
    from paper.scripts import titantpp_shared_history_raw9_5090_runtime as runtime
    runtime.validate_contract(c, "5090")
    BUNDLE.mkdir(parents=True)
    for name, digest in c["input_files"].items():
        dest = BUNDLE / "input_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous.BUNDLE / "input_snapshot" / name, dest)
        require(sha(dest) == digest, "Original uncapped input snapshot changed")
    write(BUNDLE / "execution_contract.json", c)
    with (BUNDLE / "deployment.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(BUNDLE / "execution_contract.json", arcname="execution_contract.json")
        for name in files:
            archive.add(ROOT / name, arcname="source/" + name)
        for name in c["input_files"]:
            archive.add(BUNDLE / "input_snapshot" / name, arcname="source/" + name)
        marker = tarfile.TarInfo("source/sample_data")
        marker.type, marker.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(marker)
    receipt = {"status": "prepared_for_idle5090", "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": canonical(c), "archive_sha256": sha(BUNDLE / "deployment.tar.gz"),
        "new_fit_count": 9, "baseline_reference_count": 9, "source_file_count": len(files),
        "host": "5090", "max_parallel": 3, "held_out_test_evaluated": False}
    write(REPORT / "PREPARED.json", receipt)
    return receipt


def verify(*, archive=False, require_predecessor_complete=False):
    original = previous.verify(archive=False)
    migration_gate()
    receipt = read(REPORT / "PREPARED.json")
    path = BUNDLE / "execution_contract.json"
    require(sha(path) == receipt["contract_sha256"], "5090 contract bytes changed")
    c = read(path)
    require(canonical(c) == receipt["canonical_contract_sha256"], "5090 canonical contract changed")
    require(sha(REPORT / "SCIENCE_SPEC.json") == c["science_spec_sha256"], "5090 specification changed")
    for key in ("datasets", "input_files", "training", "architecture", "common_condition", "baseline_validation_references", "limits"):
        require(c[key] == original[key], "Scientific condition changed during host migration: " + key)
    require(c["migration"]["cancellation_sha256"] == sha(CANCELLATION)
        and c["migration"]["initial5090_sha256"] == sha(REPORT / "INITIAL_5090.json")
        and c["migration"]["host_parent_sha256"] == sha(HOST_PARENT), "5090 migration evidence changed")
    for name, digest in c["source"]["files"].items():
        require(sha(ROOT / name) == digest, "5090 source changed: " + name)
    for name, digest in c["input_files"].items():
        require(sha(BUNDLE / "input_snapshot" / name) == digest, "5090 raw input changed")
    if archive:
        require(sha(BUNDLE / "deployment.tar.gz") == receipt["archive_sha256"], "5090 deployment archive changed")
    from paper.scripts import titantpp_shared_history_raw9_5090_runtime as runtime
    runtime.validate_contract(c, "5090")
    return c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "verify"), required=True)
    a = parser.parse_args(argv)
    result = prepare() if a.mode == "prepare" else verify(archive=True)
    print(__import__("json").dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
