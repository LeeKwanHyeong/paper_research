"""Prepare/seal metadata-only execution v2 without modifying frozen science.

No preparation or remote action occurs on import. All writes are exclusive in
new versioned directories, and the original absolute host deadlines remain fixed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts import manage_shared_history_mlp_control as manager  # noqa: E402

ROOT, OLD_BUNDLE = manager.ROOT, manager.BUNDLE
BUNDLE, REPORT = OLD_BUNDLE / "execution_v2", manager.REPORT / "execution_v2"
OLD_SOURCE_REVISION = "747e9e255a6ebe2dd180fcafd61af7a494bce25c"
SELF = (
    "reports/shared_history_mlp_control_20261022_v1/execution_v2/prepare_execution.py"
)
INITIAL = (
    "reports/shared_history_mlp_control_20261022_v1/execution_v2/initial_execution.py"
)
require, read, sha, write, canonical = (
    manager.require,
    manager.read,
    manager.sha,
    manager.write,
    manager.canonical,
)


def pinned(path, digest):
    path = Path(path)
    require(
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(ROOT.resolve())
        and sha(path) == digest,
        "Pinned execution-v2 file changed: " + str(path),
    )


def original_helper():
    spec = importlib.util.spec_from_file_location(
        "frozen_initial_metadata_v2", manager.REPORT / "initial_execution.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def version_contract(old, parent_raw_sha):
    """Change only run metadata, isolated paths and a new implementation gate."""
    c = deepcopy(old)
    c["execution_version"] = 2
    c["design_sha256"] = old["science_spec_sha256"]
    c["execution_parent_contract_file_sha256"] = parent_raw_sha
    c["implementation_qualification"] = {"status": "pending", "training_allowed": False}
    for host, h in c["hosts"].items():
        root = old["hosts"][host]["root"]
        require(root.endswith("_" + host), "Original host root suffix differs")
        root = root[: -len(host) - 1] + "_execution_v2_" + host
        h.update(
            root=root,
            source_root=root + "/source",
            tmux=manager.NAME + "_execution_v2_" + host,
        )
        h["environment"].update(
            PYTHONPATH=root + "/source",
            MPLCONFIGDIR=root + "/cache/matplotlib",
            XDG_CACHE_HOME=root + "/cache",
            TORCHINDUCTOR_CACHE_DIR=root + "/cache/torchinductor",
        )
    return c


def validate_version(c, old, parent_raw_sha, *, sealed=False):
    expected = version_contract(old, parent_raw_sha)
    gate = c.get("implementation_qualification", {})
    if sealed:
        require(
            gate.get("status") == "passed"
            and gate.get("training_allowed") is True
            and all(
                isinstance(gate.get(key), str) and len(gate[key]) == 64
                for key in ("independent_qa_sha256", "tests_receipt_sha256")
            ),
            "Execution-v2 implementation gate is not sealed",
        )
        expected["implementation_qualification"] = gate
    require(
        c == expected,
        "Execution v2 must differ only in design metadata, paths and sealed implementation gate",
    )
    manager.validate(c)
    require(
        len(c["source"]["files"]) == 991
        and len(c["input_files"]) == 9
        and len(c["legacy_files"]) == 199,
        "Frozen inventory cardinality differs",
    )
    require(
        c["source"]["base_git_revision"] == OLD_SOURCE_REVISION,
        "Frozen scientific source revision differs",
    )


def verify_admission(admission_sha):
    path = REPORT / "ADMISSION.json"
    pinned(path, admission_sha)
    value = read(path)
    require(
        value.get("status") == "approved_within_existing_setup_scope"
        and value.get("execution_version") == 2
        and value.get("campaign") == manager.NAME
        and value.get("parent_contract_file_sha256")
        == sha(OLD_BUNDLE / "execution_contract.json"),
        "Metadata-only execution-v2 admission differs",
    )
    old = read(OLD_BUNDLE / "execution_contract.json")
    manager.validate(old)
    require(
        "design_sha256" not in old, "Original missing-key contract unexpectedly changed"
    )
    for name, digest in old["source"]["files"].items():
        pinned(ROOT / name, digest)
    for name, digest in {**old["input_files"], **old["legacy_files"]}.items():
        pinned(manager.below(OLD_BUNDLE / "input_snapshot", name), digest)
    pinned(manager.REPORT / "SCIENCE_SPEC.json", old["science_spec_sha256"])
    pinned(manager.REPORT / "AUTHORIZATION.json", old["authorization_sha256"])
    original_qa = manager.REPORT / "INDEPENDENT_QA.json"
    pinned(original_qa, old["implementation_qualification"]["independent_qa_sha256"])
    qa = read(original_qa)
    require(
        qa.get("status") == "PASS" and qa.get("independent_reviewer") is True,
        "Original scientific QA changed",
    )
    for name, digest in qa["files_reviewed_sha256"].items():
        pinned(ROOT / name, digest)
    manager.verify_parent_preservation()
    preservation = value.get("preserved_files")
    require(
        isinstance(preservation, dict) and bool(preservation),
        "Preserved failure/source inventory missing",
    )
    for name, item in preservation.items():
        pinned(ROOT / name, item["sha256"])
        require(
            (ROOT / name).stat().st_size == item["bytes"], "Preserved file size changed"
        )
    require(
        set(qa["files_reviewed_sha256"]) <= set(preservation),
        "Original reviewed files must be included in preserved evidence",
    )
    return value, old


def prepare(admission_sha):
    _, old = verify_admission(admission_sha)
    require(
        not BUNDLE.exists(),
        "Execution v2 already exists; preserve partial or complete files",
    )
    c = version_contract(old, sha(OLD_BUNDLE / "execution_contract.json"))
    validate_version(c, old, sha(OLD_BUNDLE / "execution_contract.json"))
    BUNDLE.mkdir(parents=True, exist_ok=False)
    write(
        BUNDLE / "PREPARE_ATTEMPT.json",
        {
            "admission_sha256": admission_sha,
            "parent_contract_file_sha256": sha(OLD_BUNDLE / "execution_contract.json"),
        },
    )
    for name, digest in {**c["input_files"], **c["legacy_files"]}.items():
        source = manager.below(OLD_BUNDLE / "input_snapshot", name)
        target = manager.below(BUNDLE / "input_snapshot", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        require(sha(target) == digest, "Versioned byte copy changed")
    write(BUNDLE / "prepared_contract.json", c)
    return {
        "status": "prepared_not_trainable",
        "execution_version": 2,
        "source_unchanged": True,
        "new_fit": 9,
        "reuse": 18,
        "deadline_extended": False,
    }


def verify_qa_tests(c, qa_sha, tests_receipt):
    pinned(REPORT / "INDEPENDENT_QA.json", qa_sha)
    qa = read(REPORT / "INDEPENDENT_QA.json")
    required = {SELF, INITIAL}
    require(
        qa.get("status") == "PASS"
        and qa.get("independent_reviewer") is True
        and required <= set(qa.get("files_reviewed_sha256", {})),
        "Independent execution-v2 QA missing",
    )
    for name, digest in qa["files_reviewed_sha256"].items():
        pinned(ROOT / name, digest)
    tests_path = Path(tests_receipt)
    tests = read(tests_path)
    require(
        tests.get("actual_os_returncode") == 0
        and tests.get("source_unchanged_during_tests") is True
        and tests.get("actual_cpu_training_e2e_passed") is True,
        "Actual successful CPU training E2E is required before seal",
    )
    require(
        all(
            tests.get("files_checked_sha256", {}).get(name) == sha(ROOT / name)
            for name in required
        ),
        "Execution-v2 tests are not bound to source",
    )
    require(
        all(
            sha(ROOT / name) == digest for name, digest in c["source"]["files"].items()
        ),
        "Frozen scientific source changed during local verification",
    )
    return qa, tests


def seal(admission_sha, qa_sha, tests_receipt):
    _, old = verify_admission(admission_sha)
    c = read(BUNDLE / "prepared_contract.json")
    validate_version(c, old, sha(OLD_BUNDLE / "execution_contract.json"))
    verify_qa_tests(c, qa_sha, tests_receipt)
    c["implementation_qualification"] = {
        "status": "passed",
        "training_allowed": True,
        "independent_qa_sha256": qa_sha,
        "tests_receipt_sha256": sha(tests_receipt),
    }
    validate_version(c, old, sha(OLD_BUNDLE / "execution_contract.json"), sealed=True)
    write(BUNDLE / "execution_contract.json", c)
    with (
        (BUNDLE / "deployment.tar.gz").open("xb") as stream,
        tarfile.open(fileobj=stream, mode="w:gz") as archive,
    ):
        archive.add(
            BUNDLE / "execution_contract.json", arcname="execution_contract.json"
        )
        for name, digest in c["source"]["files"].items():
            pinned(ROOT / name, digest)
            archive.add(ROOT / name, arcname="source/" + name)
        for name, digest in {**c["input_files"], **c["legacy_files"]}.items():
            source = manager.below(BUNDLE / "input_snapshot", name)
            pinned(source, digest)
            archive.add(source, arcname="source/" + name)
        marker = tarfile.TarInfo("source/sample_data")
        marker.type = tarfile.DIRTYPE
        archive.addfile(marker)
    result = {
        "status": "sealed",
        "execution_version": 2,
        "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": canonical(c),
        "archive_sha256": sha(BUNDLE / "deployment.tar.gz"),
        "source_files_sha256": c["source"]["files_sha256"],
        "admission_sha256": admission_sha,
        "independent_qa_sha256": qa_sha,
        "tests_receipt_sha256": sha(tests_receipt),
        "tests_receipt_path": str(Path(tests_receipt).resolve()),
        "parent_contract_file_sha256": sha(OLD_BUNDLE / "execution_contract.json"),
        "deadline_extended": False,
        "source_unchanged": True,
    }
    write(BUNDLE / "integrity.json", result)
    return result


def verify_sealed(admission_sha, integrity_sha, qa_sha):
    _, old = verify_admission(admission_sha)
    pinned(BUNDLE / "integrity.json", integrity_sha)
    integrity = read(BUNDLE / "integrity.json")
    c = read(BUNDLE / "execution_contract.json")
    validate_version(c, old, sha(OLD_BUNDLE / "execution_contract.json"), sealed=True)
    require(
        integrity.get("status") == "sealed"
        and integrity.get("execution_version") == 2
        and integrity.get("admission_sha256") == admission_sha
        and integrity.get("independent_qa_sha256") == qa_sha
        and integrity.get("contract_sha256") == sha(BUNDLE / "execution_contract.json")
        and integrity.get("canonical_contract_sha256") == canonical(c)
        and integrity.get("archive_sha256") == sha(BUNDLE / "deployment.tar.gz")
        and integrity.get("source_files_sha256") == c["source"]["files_sha256"],
        "Sealed v2 inventory changed",
    )
    tests_path = Path(integrity["tests_receipt_path"])
    pinned(tests_path, integrity["tests_receipt_sha256"])
    verify_qa_tests(c, qa_sha, tests_path)
    require(
        c["implementation_qualification"]["independent_qa_sha256"] == qa_sha
        and c["implementation_qualification"]["tests_receipt_sha256"]
        == integrity["tests_receipt_sha256"],
        "Sealed gate binding differs",
    )
    for name, digest in {**c["input_files"], **c["legacy_files"]}.items():
        pinned(manager.below(BUNDLE / "input_snapshot", name), digest)
    original_helper().check_archive(
        (BUNDLE / "deployment.tar.gz").read_bytes(),
        integrity["contract_sha256"],
        canonical(c),
    )
    return c, integrity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "seal", "check"), required=True)
    parser.add_argument("--admission-sha", required=True)
    parser.add_argument("--qa-sha")
    parser.add_argument("--tests-receipt", type=Path)
    parser.add_argument("--integrity-sha")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        result = prepare(args.admission_sha)
    elif args.mode == "seal":
        result = seal(args.admission_sha, args.qa_sha, args.tests_receipt)
    else:
        c, _ = verify_sealed(args.admission_sha, args.integrity_sha, args.qa_sha)
        result = {
            "status": "sealed_verified",
            "execution_version": 2,
            "canonical_contract_sha256": canonical(c),
            "ssh_calls": 0,
        }
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
