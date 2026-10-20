"""Package only the approved extra seeds; preserve scientific source983 and historical fits."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from paper.scripts import titantpp_shared_external_extra42_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_shared_external_extra42_20261020_v1"
BUNDLE, REPORT = ROOT / "search_artifacts" / NAME, ROOT / "reports" / NAME
PARENT = ROOT / "search_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1"
PARENT_SHA = "4ed35f0054995f51dadb9b1d6227fc9c09194f9414a9f832bb723dc1d1c058f1"
SPEC, CONDITIONS = REPORT / "SCIENCE_SPEC.json", REPORT / "FROZEN_CONDITIONS.json"
NEW_SOURCE = ("paper/scripts/prepare_titantpp_shared_external_extra42.py", "paper/scripts/manage_titantpp_shared_external_extra42.py",
    "paper/scripts/titantpp_shared_external_extra42_runtime.py", "paper/scripts/run_titantpp_shared_external_extra42.py")
read, sha, canonical, require = runtime.read, runtime.sha_file, runtime.sha_json, runtime.require


def write(path, value): runtime.write(path, value, exclusive=True)


def population(value):
    return {k: value[k] for k in ("target_count", "target_identity_sha256", "target_quantity_sha256")}


def build_jobs():
    return [{"id": f"{d}__{a}__seed{s}", "dataset": d, "arm": a, "seed": s, "host": "5080",
        "time_alpha": 1.0, "time_input": "log1p", "loss_mode": "log_mse"}
        for d in runtime.DATASETS for a in runtime.ARMS for s in runtime.SEEDS]


def _copy(source, relative, digest):
    source = Path(source); require(sha(source) == digest, "Original source/input changed: " + str(source))
    target = runtime.local_path(BUNDLE, relative); target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst: shutil.copyfileobj(src, dst)
    require(sha(target) == digest, "Snapshot copy changed"); return target


def prepare():
    require(not BUNDLE.exists(), "Existing extra42 bundle; preserve attempt and originals")
    require(sha(PARENT / "execution_contract.json") == PARENT_SHA, "Original raw9 contract changed")
    original, spec, conditions = read(PARENT / "execution_contract.json"), read(SPEC), read(CONDITIONS)
    require(len(original["source"]["files"]) == 983 and set(conditions["models"]) == set(runtime.ARMS), "Frozen source983/model7 missing")
    for name, digest in original["source"]["files"].items(): require(sha(ROOT / name) == digest, "Preserved source983 changed")
    for name, digest in conditions.get("source_bindings", {}).items(): require(sha(ROOT / name) == digest, "Historical architecture evidence changed")
    c = deepcopy(original); c.pop("migration", None); c.pop("predecessor", None)
    remote = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + NAME + "_5080"
    native = deepcopy(spec["native"]); native.update(root=remote, source_root=remote + "/source", tmux="shared_external_extra42_5080_20261020")
    native["environment"].update(PYTHONPATH=native["source_root"], MPLCONFIGDIR=remote + "/cache/matplotlib", XDG_CACHE_HOME=remote + "/cache")
    source_files = {**original["source"]["files"], **{name: sha(ROOT / name) for name in NEW_SOURCE}}
    c.update(schema=runtime.SCHEMA, campaign=NAME, scope="execution", created_at=runtime.now(), hosts={"5080": native},
        source={**deepcopy(original["source"]), "files": source_files, "files_sha256": canonical(source_files), "preserved_parent_file_count": 983,
            "preserved_parent_files_sha256": original["source"]["files_sha256"]},
        jobs=build_jobs(), execution_authorization={"real_data_training": True, "real_data_inference": True, "remote_execution": True, "test_evaluation": True},
        maximum_parallel_workers=3, held_out_test_authorized=True, evaluation_scope="validation_selection_then_frozen_test",
        historical_models_retrained=False, historical_cnn_retrained=False, historical_cnn_reevaluated=False, checkpoint_warmstart=False,
        frozen_conditions=deepcopy(conditions), science_spec_sha256=sha(SPEC), frozen_conditions_sha256=sha(CONDITIONS),
        parent_original_contract={"path": str(PARENT / "execution_contract.json"), "sha256": PARENT_SHA},
        prior_owned_pids=spec["prior_owned_pids"], prior_completion_bindings=spec["prior_completion_bindings"],
        approval=spec["user_instruction"], recurring_automation_created=False)
    c["architecture"] = {**c["architecture"], "arms": list(runtime.ARMS), "native_paper_heads_used": False}
    c["design_sha256"] = canonical(c["architecture"])
    for definition in c["datasets"]:
        d = definition["dataset_id"]; full = spec["datasets"][d]["full_data"]
        relative = f"inputs/full/{d}.parquet"; c["input_files"][relative] = full["sha256"]
        definition["full_data_identity"] = {"data": {"path": relative, "sha256": full["sha256"]},
            "populations": {"test": population(spec["datasets"][d]["populations"]["test"])}}
    runtime.validate_contract(c, "5080")
    BUNDLE.mkdir(parents=True, exist_ok=False)
    for name, digest in original["input_files"].items(): _copy(PARENT / "input_snapshot" / name, "input_snapshot/" + name, digest)
    for definition in c["datasets"]:
        d = definition["dataset_id"]; full = spec["datasets"][d]["full_data"]; path = Path(full["path"])
        _copy(path if path.is_absolute() else ROOT / path, "input_snapshot/inputs/full/" + d + ".parquet", full["sha256"])
    write(BUNDLE / "execution_contract.json", c)
    result = {"status": "prepared_not_sealed", "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": canonical(c), "new_fit_count": 42, "extra_seeds": list(runtime.SEEDS),
        "source_file_count": len(source_files), "actual_scientific_fit_count": 0}
    write(REPORT / "PREPARED.json", result); return result


def seal(tests_receipt):
    require(not (BUNDLE / "integrity.json").exists() and not (BUNDLE / "deployment.tar.gz").exists(), "Existing seal; do not overwrite")
    c = read(BUNDLE / "execution_contract.json"); runtime.validate_contract(c, "5080")
    tests = read(tests_receipt); require(tests.get("status") in ("PASS", "passed"), "Operating/scientific tests did not pass")
    qa = read(REPORT / "INDEPENDENT_QA.json"); reviewed = qa.get("files_reviewed_sha256", {})
    required = {*NEW_SOURCE, "tests/test_titantpp_shared_external_extra42_ops.py", "tests/test_titantpp_shared_external_extra42_worker.py",
        str(SPEC.relative_to(ROOT)), str(CONDITIONS.relative_to(ROOT))}
    require(qa.get("status") == "PASS" and required <= set(reviewed), "Independent implementation review inventory incomplete")
    for name, digest in reviewed.items(): require(sha(runtime.local_path(ROOT, name)) == digest, "Version changed after independent review")
    tracked = set(subprocess.check_output(["git", "ls-files", "--", *NEW_SOURCE], cwd=ROOT, text=True).splitlines())
    require(tracked == set(NEW_SOURCE) and not subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *NEW_SOURCE], cwd=ROOT, text=True).strip(), "Commit isolated operation files before sealing")
    files = {"execution_contract.json": sha(BUNDLE / "execution_contract.json")}
    for name, digest in c["source"]["files"].items(): require(sha(ROOT / name) == digest, "Prepared source changed before seal")
    with (BUNDLE / "deployment.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(BUNDLE / "execution_contract.json", arcname="execution_contract.json")
        for name in c["source"]["files"]: archive.add(ROOT / name, arcname="source/" + name)
        for name, digest in c["input_files"].items():
            p = BUNDLE / "input_snapshot" / name; require(sha(p) == digest, "Prepared input changed")
            archive.add(p, arcname="source/" + name); files["input_snapshot/" + name] = digest
        marker = tarfile.TarInfo("source/sample_data"); marker.type, marker.mode = tarfile.DIRTYPE, 0o755; archive.addfile(marker)
    files["deployment.tar.gz"] = sha(BUNDLE / "deployment.tar.gz")
    result = {"schema": "titantpp_shared_external_extra42_integrity_v1", "status": "sealed", "files": files,
        "source_files": c["source"]["files"], "canonical_contract_sha256": canonical(c), "science_spec_sha256": sha(SPEC),
        "frozen_conditions_sha256": sha(CONDITIONS), "tests_receipt": {"path": str(tests_receipt), "sha256": sha(tests_receipt)},
        "independent_qa_sha256": sha(REPORT / "INDEPENDENT_QA.json"), "operation_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    write(BUNDLE / "integrity.json", result)
    return {"status": "sealed", "integrity_sha256": sha(BUNDLE / "integrity.json"), "contract_sha256": files["execution_contract.json"], "new_fit_count": 42}


def verify(integrity_sha, *, archive=False):
    require(sha(BUNDLE / "integrity.json") == integrity_sha, "Integrity raw SHA differs")
    sealed = read(BUNDLE / "integrity.json"); c = read(BUNDLE / "execution_contract.json")
    require(sealed["status"] == "sealed" and canonical(c) == sealed["canonical_contract_sha256"] and sha(SPEC) == sealed["science_spec_sha256"]
        and sha(CONDITIONS) == sealed["frozen_conditions_sha256"] and sha(REPORT / "INDEPENDENT_QA.json") == sealed["independent_qa_sha256"], "Science/implementation seal changed")
    for name, digest in sealed["files"].items():
        if name == "deployment.tar.gz" and not archive: continue
        require(sha(runtime.local_path(BUNDLE, name)) == digest, "Sealed input/archive/contract changed")
    for name, digest in sealed["source_files"].items(): require(sha(runtime.local_path(ROOT, name)) == digest, "Sealed scientific/operation source changed")
    for name, digest in c["prior_completion_bindings"].items(): require(sha(name) == digest, "Prior completion source changed")
    runtime.validate_contract(c, "5080"); return c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", choices=("prepare", "seal", "verify"), required=True)
    parser.add_argument("--tests-receipt", type=Path); parser.add_argument("--integrity-sha"); a = parser.parse_args(argv)
    result = prepare() if a.mode == "prepare" else seal(a.tests_receipt) if a.mode == "seal" else verify(a.integrity_sha, archive=True)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
