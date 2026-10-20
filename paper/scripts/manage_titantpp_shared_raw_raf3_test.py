"""One finite RAF3 frozen-checkpoint Validation/Test evaluation after raw9 completes.

This module never trains a model, resumes an old child or changes a training
artifact. Existing SSH/archive primitives are reused; only new evaluation
children belong to this supervisor's timeout cleanup.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import (
    gpu_snapshot, local_path, now, read, require, sha_file as sha, sha_json as canonical,
    terminate_owned_child, write as _write)

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_shared_raw_raf3_test_20261020_v1"
BUNDLE, REPORT = ROOT / "search_artifacts" / NAME, ROOT / "reports" / NAME
SPEC, REFERENCES = REPORT / "SCIENCE_SPEC.json", REPORT / "REFERENCES.json"
TRAIN_BUNDLE = ROOT / "search_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1"
TRAIN_REPORT = ROOT / "reports/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1"
TRAIN_SHA = "4ed35f0054995f51dadb9b1d6227fc9c09194f9414a9f832bb723dc1d1c058f1"
REMOTE = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + NAME + "_5090"
MODULE = "paper.scripts.manage_titantpp_shared_raw_raf3_test"
WORKER = "paper.scripts.evaluate_titantpp_shared_raw_raf3_frozen"
GPU_UUID = "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"
EPOCHS = {42: 9, 52: 5, 62: 19}
_OWNED_CHILDREN = []
NEW_SOURCE = ("paper/scripts/manage_titantpp_shared_raw_raf3_test.py",
    "paper/scripts/evaluate_titantpp_shared_raw_raf3_frozen.py",
    "paper/scripts/compare_titantpp_shared_raw_raf3_test.py")
REVIEW_FILES = (*NEW_SOURCE, "tests/test_titantpp_shared_raw_raf3_test_ops.py",
    "tests/test_titantpp_shared_raw_raf3_frozen.py", "tests/test_titantpp_shared_raw_raf3_test_compare.py",
    "reports/" + NAME + "/SCIENCE_SPEC.json")
_BASE = Path(__file__).with_name("manage_titantpp_quantity_loss_mlp18.py")
_BASE_SHA = "331aa17a98532217bb4cdda8865cf09530cc54b5c31f4968456655e93be27e8d"
require(sha(_BASE) == _BASE_SHA, "Preserved bounded transport/archive primitives changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_transport", _BASE)
_transport = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transport)


def write(path, value):
    _write(path, value, exclusive=True)


def verify_reviewed_implementation():
    """Admit exactly the source, tests and science version independently reviewed."""
    path = REPORT / "INDEPENDENT_QA.json"
    report = read(path)
    require(report.get("status") == "PASS", "Independent evaluation review did not pass")
    reviewed = report.get("files_reviewed_sha256")
    require(isinstance(reviewed, dict) and set(REVIEW_FILES) <= set(reviewed), "Independent review inventory is incomplete")
    for name, digest in reviewed.items():
        require(sha(local_path(ROOT, name)) == digest, "Evaluation implementation changed after independent review: " + name)
    return {"path": str(path), "sha256": sha(path)}


def _pid_present(pid):
    result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "pid="], capture_output=True, text=True, timeout=20)
    return result.returncode == 0 and result.stdout.strip() == str(pid)


def training_complete():
    """Recheck original raw9 completion and all original SHA bindings locally."""
    from paper.scripts import manage_titantpp_shared_history_raw9_5090 as parent
    require(sha(TRAIN_BUNDLE / "execution_contract.json") == TRAIN_SHA, "Frozen raw9 contract changed")
    contract = parent.verify_local()
    final = read(TRAIN_BUNDLE / "retrieval/final_receipt.json")
    require(final["status"] == "complete" and final["new_fit_count"] == final["validation_replay_count"] == 9
        and final["actual_scientific_worker_exit0_count"] == 18 and final["actual_dispatcher_exit0_count"] == 1
        and final["all_training_terminal_verified"] is True and final["held_out_test_evaluated"] is False,
        "Original raw9 completion is not fully verified")
    retrieval = TRAIN_BUNDLE / "retrieval/5090"
    parent.verify_collected(contract, "5090", retrieval / "originals")
    collection, manifest = read(retrieval / "receipt.json"), read(retrieval / "manifest.json")["files"]
    require(collection["status"] == "complete" and manifest and len(manifest) == collection["file_count"]
        and sha(retrieval / "original.tar.gz") == collection["archive_sha256"]
        and sha(retrieval / "manifest.json") == collection["manifest_sha256"], "Original raw9 archive/manifest changed")
    for name, digest in manifest.items():
        require(sha(local_path(retrieval / "originals", name)) == digest, "Original raw9 artifact changed")
    for field in ("source_bindings", "archive_bindings", "report_files"):
        require(final.get(field), "Raw9 completion binding missing")
        for name, digest in final[field].items():
            require(sha(name) == digest, "Raw9 completion source/result changed")
    finisher = read(TRAIN_REPORT / "FINISHER_STARTED.json")
    require(not _pid_present(finisher["pid"]), "Existing raw9 finisher has not exited")
    return contract, retrieval / "originals", final


def _copy(source, relative, expected):
    require(sha(source) == expected, "Input/checkpoint/metadata SHA changed: " + str(source))
    target = local_path(BUNDLE, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "Preserve existing evaluation original")
    shutil.copyfile(source, target)
    require(sha(target) == expected, "Copied evaluation original differs")
    return {"path": relative, "sha256": expected}


def prepare():
    verify_reviewed_implementation()
    require(not BUNDLE.exists(), "Existing evaluation bundle; no overwrite/retry")
    contract, originals, completion = training_complete()
    spec, references = read(SPEC), read(REFERENCES)
    require(len(references.get("references", [])) == 21, "Expected21 fixed Test references")
    tracked = set(subprocess.check_output(["git", "ls-files", "--", *NEW_SOURCE], cwd=ROOT, text=True).splitlines())
    require(tracked == set(NEW_SOURCE) and not subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *NEW_SOURCE], cwd=ROOT, text=True).strip(), "Commit isolated evaluation operation files before prepare")
    BUNDLE.mkdir(parents=True, exist_ok=False)
    training_contract = _copy(TRAIN_BUNDLE / "execution_contract.json", "training/execution_contract.json", TRAIN_SHA)
    training_completion = _copy(TRAIN_BUNDLE / "retrieval/final_receipt.json", "training/final_receipt.json", sha(TRAIN_BUNDLE / "retrieval/final_receipt.json"))
    references_binding = _copy(REFERENCES, "references/REFERENCES.json", sha(REFERENCES))
    for name, digest in contract["source"]["files"].items():
        _copy(ROOT / name, "source/" + name, digest)
    operation_files = {}
    for name in NEW_SOURCE:
        digest = sha(ROOT / name)
        _copy(ROOT / name, "source/" + name, digest)
        operation_files[name] = digest
    jobs = []
    for job in contract["jobs"]:
        if job["dataset"] != "raf_spare_parts":
            continue
        job_root = originals / "run" / job["id"]
        train, validation = read(job_root / "train_receipt.json"), read(job_root / "validation_receipt.json")
        require(train["selected_epoch"] == validation["selected_epoch"] == EPOCHS[job["seed"]], "Frozen RAF selected epoch changed")
        frozen = next(value for value in spec["selected_checkpoints"] if value["seed"] == job["seed"])
        require(train["checkpoint_sha256"] == frozen["checkpoint_sha256"] and train["state_sha256"] == frozen["state_sha256"], "Approved selected binary/tensor changed")
        remote_selected = Path(train["selected_checkpoint"])
        selected_rel = str(remote_selected.relative_to(contract["hosts"]["5090"]["root"]))
        selected = local_path(originals, selected_rel)
        history, metadata = selected.parent / "history.json", selected.parent / "summary.json"
        copied = {}
        for name, digest in train["files"].items():
            copied[name] = _copy(local_path(originals, name), "training/" + name, digest)
        for name, path in (("train_receipt", job_root / "train_receipt.json"), ("validation_receipt", job_root / "validation_receipt.json")):
            relative = "training/" + str(path.relative_to(originals))
            if not (BUNDLE / relative).exists(): _copy(path, relative, sha(path))
        entry = {"id": job["id"], "dataset": job["dataset"], "seed": job["seed"], "arm": job["arm"], "selected_epoch": EPOCHS[job["seed"]]}
        paths = {"train_receipt": job_root / "train_receipt.json", "validation_receipt": job_root / "validation_receipt.json",
            "checkpoint": selected, "selection_history": history, "checkpoint_metadata": metadata,
            "validation_predictions": Path(validation["prediction_path"])}
        paths["validation_predictions"] = local_path(originals, str(paths["validation_predictions"].relative_to(contract["hosts"]["5090"]["root"])))
        prediction_relative = "training/" + str(paths["validation_predictions"].relative_to(originals))
        if not (BUNDLE / prediction_relative).exists(): _copy(paths["validation_predictions"], prediction_relative, sha(paths["validation_predictions"]))
        for label, path in paths.items():
            relative = "training/" + str(path.relative_to(originals))
            require((BUNDLE / relative).is_file(), "Frozen job original missing: " + label)
            entry[label] = {"path": relative, "sha256": sha(path)}
        jobs.append(entry)
    require(len(jobs) == 3, "RAF fixed jobs3 missing")
    dataset = next(d for d in contract["datasets"] if d["dataset_id"] == "raf_spare_parts")
    relative = "inputs/data/raf_spare_parts/train_validation.parquet"
    training = _copy(TRAIN_BUNDLE / "input_snapshot" / relative, "source/" + relative, contract["input_files"][relative])
    manifest = "inputs/data/raf_spare_parts/split_manifest.json"
    _copy(TRAIN_BUNDLE / "input_snapshot" / manifest, "source/" + manifest, contract["input_files"][manifest])
    full = spec["full_data"]
    full_binding = _copy(ROOT / full["path"], "inputs/raf_original_full.parquet", full["sha256"])
    h = contract["hosts"]["5090"]
    c = {"schema": "titantpp_shared_raw_raf3_frozen_v1", "host": "5090", "dataset": "raf_spare_parts",
        "evaluation_authorized": True, "source_root": "source", "training_contract": training_contract,
        "training_completion": training_completion, "jobs": jobs, "data": {"training": training, "full": full_binding,
            "populations": spec["populations"]}, "operation_files": operation_files, "references": references_binding,
        "remote_root": REMOTE, "native": {"python": h["python"], "gpu_uuid": h["gpu_uuid"], "runtime_expected": h["runtime_expected"],
            "environment": {**h["environment"], "PYTHONPATH": REMOTE + "/source", "MPLCONFIGDIR": REMOTE + "/cache/matplotlib", "XDG_CACHE_HOME": REMOTE + "/cache"}},
        "absolute_host_deadline_unix": h["absolute_host_deadline_unix"], "child_timeout_seconds": 7200,
        "references_sha256": sha(REFERENCES), "science_spec_sha256": sha(SPEC), "held_out_test_authorized": True,
        "new_training_count": 0, "checkpoint_reselection": False, "reference_reexecution_count": 0,
        "recurring_automation_created": False}
    write(BUNDLE / "evaluation_contract.json", c)
    with (BUNDLE / "deployment.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file() and path.name != "deployment.tar.gz": archive.add(path, arcname=str(path.relative_to(BUNDLE)))
        marker = tarfile.TarInfo("source/sample_data"); marker.type, marker.mode = tarfile.DIRTYPE, 0o755; archive.addfile(marker)
    receipt = {"status": "prepared", "contract_sha256": sha(BUNDLE / "evaluation_contract.json"),
        "canonical_contract_sha256": canonical(c), "archive_sha256": sha(BUNDLE / "deployment.tar.gz"),
        "candidate_count": 3, "baseline_reference_count": 21, "actual_native_evaluation_count": 0}
    write(REPORT / "PREPARED.json", receipt)
    return receipt


def verify_local(*, archive=False):
    receipt = read(REPORT / "PREPARED.json")
    path = BUNDLE / "evaluation_contract.json"
    require(sha(path) == receipt["contract_sha256"], "Evaluation contract bytes changed")
    c = read(path)
    require(canonical(c) == receipt["canonical_contract_sha256"] and sha(SPEC) == c["science_spec_sha256"]
        and sha(REFERENCES) == c["references_sha256"], "Evaluation specification/references changed")
    for name, digest in c["operation_files"].items(): require(sha(ROOT / name) == digest, "Prepared evaluation operation changed")
    if archive: require(sha(BUNDLE / "deployment.tar.gz") == receipt["archive_sha256"], "Deployment archive changed")
    return c


def _receipt_path(root, job, stage):
    return root / "qualification" / (job["id"] + ".json") if stage == "qualification" else root / "results" / job["id"] / f"{stage}_receipt.json"


def _owned_pid_absent(pid):
    return not (Path("/proc") / str(pid)).exists()


def _run_remote_stage(root, contract, stage):
    """Fresh child per seed, at most3; drain current children after a failure."""
    require(stage in ("qualification", "validation", "test"), "Unsupported evaluation stage")
    logs = root / "logs"; logs.mkdir(exist_ok=True)
    pending, active, exits, failure = list(contract["jobs"]), {}, {}, None
    native, source = contract["native"], local_path(root, contract["source_root"])
    while active or (pending and failure is None):
        for job_id, record in list(active.items()):
            child = record["process"]
            timed_out = child.poll() is None and (time.monotonic() - record["started"] > contract["child_timeout_seconds"]
                or time.time() >= contract["absolute_host_deadline_unix"])
            if timed_out: terminate_owned_child(child)
            code = child.poll()
            if code is None: continue
            result = {"job_id": job_id, "split": stage, "contract_sha256": canonical(contract), "pid": child.pid,
                "actual_os_returncode": code, "timeout": timed_out, "actual_child_exit_observed": True,
                "stdout_sha256": sha(record["out"]), "stderr_sha256": sha(record["err"]), "observed_at": now()}
            write(root / "results" / job_id / f"{stage}_exit.json", result)
            exits[job_id] = result
            del active[job_id]
            if code != 0 or timed_out or not _receipt_path(root, record["job"], stage).exists():
                failure = failure or {"job_id": job_id, "stage": stage, "exit": result}
            else:
                receipt = read(_receipt_path(root, record["job"], stage))
                valid = receipt.get("status") == ("passed" if stage == "qualification" else "complete")
                valid = valid and receipt.get("job_id") == job_id and receipt.get("contract_sha256") == canonical(contract)
                if stage == "qualification": valid = valid and receipt.get("real_data_loaded") is False and receipt.get("checks") and all(receipt["checks"].values())
                if not valid: failure = failure or {"job_id": job_id, "stage": stage, "error": "Native stage receipt invalid"}
        if failure is None:
            try:
                require(time.time() < contract["absolute_host_deadline_unix"], "Inherited evaluation deadline expired")
                gpu = gpu_snapshot(native["gpu_uuid"])
                require(set(gpu["compute_pids"]) <= {r["process"].pid for r in active.values()}, "Unknown GPU process during evaluation")
                for _ in range(min(3 - len(active), len(pending))):
                    job = pending.pop(0)
                    write(root / "results" / job["id"] / f"{stage}_launch.intent.json", {"at": now(), "stage": stage, "job_id": job["id"]})
                    command = [native["python"], "-u", "-m", WORKER, "--root", str(root), "--job", job["id"]]
                    command += ["--mode", "synthetic"] if stage == "qualification" else ["--split", stage]
                    out, err = logs / f"{job['id']}.{stage}.stdout", logs / f"{job['id']}.{stage}.stderr"
                    with out.open("xb") as stdout, err.open("xb") as stderr:
                        child = subprocess.Popen(command, cwd=source, env={**os.environ, **native["environment"], "PYTHONDONTWRITEBYTECODE": "1"},
                            stdout=stdout, stderr=stderr, start_new_session=True)
                    _OWNED_CHILDREN.append(child)
                    active[job["id"]] = {"process": child, "job": job, "started": time.monotonic(), "out": out, "err": err}
            except Exception:
                failure = failure or {"stage": stage, "traceback": traceback.format_exc()}
        if failure is not None and not (root / "pipeline_failure.json").exists(): write(root / "pipeline_failure.json", failure)
        if active or (pending and failure is None): time.sleep(.5)
    require(not failure and len(exits) == 3, "Evaluation stage failed; raw outputs preserved")
    return exits


def run_remote_stage(root, contract, stage):
    try:
        return _run_remote_stage(root, contract, stage)
    finally:
        for child in _OWNED_CHILDREN:
            if child.poll() is None: terminate_owned_child(child)


def remote_run(root):
    root = Path(root).resolve(); c = read(root / "evaluation_contract.json")
    require(c["remote_root"] == str(root) and c["schema"] == "titantpp_shared_raw_raf3_frozen_v1", "Evaluation root/schema mismatch")
    require(not any((root / name).exists() for name in ("pipeline.intent.json", "pipeline_receipt.json", "pipeline_failure.json", "validation_gate.json", "results", "qualification")), "Existing evaluation attempt; no retry")
    write(root / "pipeline.intent.json", {"pid": os.getpid(), "at": now(), "contract_sha256": canonical(c)})
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Owned evaluation supervisor interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    require(not gpu_snapshot(c["native"]["gpu_uuid"])["compute_pids"], "5090 is not fresh idle")
    qualification = run_remote_stage(root, c, "qualification")
    validation = run_remote_stage(root, c, "validation")
    gate = {"schema": "titantpp_shared_raw_raf3_validation_gate_v1", "status": "PASS", "contract_sha256": canonical(c), "job_count": 3,
        "validation_receipts": {job["id"]: sha(_receipt_path(root, job, "validation")) for job in c["jobs"]},
        "validation_worker_exits": {job["id"]: sha(root / "results" / job["id"] / "validation_exit.json") for job in c["jobs"]}}
    write(root / "validation_gate.json", gate)
    importlib.import_module(WORKER).verify_validation_gate(root, c)
    tested = run_remote_stage(root, c, "test")
    pids = [record["pid"] for records in (qualification, validation, tested) for record in records.values()]
    require(all(_owned_pid_absent(pid) for pid in pids) and not gpu_snapshot(c["native"]["gpu_uuid"])["compute_pids"], "Evaluation owner/GPU remains")
    files = {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()
        and path.relative_to(root).parts[0] not in ("source", "cache") and path.name != "pipeline.log"}
    result = {"status": "complete", "contract_sha256": canonical(c), "qualification": qualification, "validation": validation, "test": tested,
        "qualification_exit0_count": 3, "actual_data_worker_exit0_count": 6, "actual_native_worker_exit0_count": 9,
        "owned_pids_absent": True, "gpu_compute_pids": [], "files": files}
    write(root / "remote_final_receipt.json", result)
    write(root / "pipeline_receipt.json", result)
    return result


def advance():
    training_complete()
    c = verify_local(archive=True)
    write(BUNDLE / "remote.intent.json", {"at": now(), "automatic_retry": False})
    h = c["native"]
    fresh = ("import pathlib,subprocess\n" + f"assert not pathlib.Path({REMOTE!r}).exists()\n"
        "x=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)\n"
        "assert not x.stdout.strip(),'GPU busy'\n")
    _transport.save_transport("5090", "fresh_idle", _transport.ssh("5090", shlex.join([h["python"], "-c", fresh])))
    _transport.save_transport("5090", "install", _transport.ssh("5090", "mkdir " + shlex.quote(REMOTE) + " && tar -xzf - -C " + shlex.quote(REMOTE),
        stdin=(BUNDLE / "deployment.tar.gz").read_bytes(), timeout=300))
    command = [h["python"], "-u", "-m", MODULE, "--mode", "remote-run", "--root", REMOTE]
    supervisor = ("import os,subprocess,pathlib,json,sys,time,signal\n" + f"p=subprocess.Popen({command!r},cwd={REMOTE + '/source'!r},env={{**os.environ,**{h['environment']!r}}},start_new_session=True)\n"
        f"limit=max(.01,min(21600.,{c['absolute_host_deadline_unix']!r}-time.time()));timed_out=False\n"
        "try:code=p.wait(timeout=limit)\n"
        "except subprocess.TimeoutExpired:\n"
        " timed_out=True;os.killpg(p.pid,signal.SIGTERM)\n"
        " try:code=p.wait(timeout=60)\n"
        " except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()\n"
        f"path=pathlib.Path({REMOTE!r})/'pipeline_exit.json'\n"
        "path.open('x').write(json.dumps({'pid':p.pid,'actual_os_returncode':code,'timeout':timed_out,'actual_child_exit_observed':True,'supervisor_own_exit_directly_observed':False})+chr(10))\n"
        "sys.exit(code)\n")
    _transport.save_transport("5090", "evaluation", _transport.ssh("5090", shlex.join([h["python"], "-u", "-c", supervisor]),
        timeout=max(1., min(21660., c["absolute_host_deadline_unix"] - time.time() + 60))))
    return collect_and_compare(c)


def collect_and_compare(c):
    destination = BUNDLE / "retrieval"; destination.mkdir(exist_ok=False)
    program = ("import pathlib,json,tarfile,sys,subprocess\n" + f"root=pathlib.Path({REMOTE!r})\n"
        "f=json.loads((root/'pipeline_receipt.json').read_text());x=json.loads((root/'pipeline_exit.json').read_text())\n"
        "assert f['status']=='complete' and f['owned_pids_absent'] and f['actual_native_worker_exit0_count']==9 and f['actual_data_worker_exit0_count']==6 and x['actual_os_returncode']==0 and not x['timeout']\n"
        "pids=[x['pid']]+[e['pid'] for stage in ('qualification','validation','test') for e in f[stage].values()]\n"
        "assert all(not (pathlib.Path('/proc')/str(pid)).exists() for pid in pids)\n"
        "gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20);assert not gpu.stdout.strip()\n"
        "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:\n"
        " for path in sorted(root.rglob('*')):\n"
        "  rel=path.relative_to(root)\n"
        "  if rel.parts[0] in ('source','cache'):continue\n"
        "  assert not path.is_symlink()\n"
        "  if path.is_file():archive.add(path,arcname=str(rel),recursive=False)\n")
    result = _transport.ssh("5090", shlex.join([c["native"]["python"], "-c", program]), timeout=900)
    archive = destination / "original.tar.gz"; archive.write_bytes(result.stdout)
    _transport.save_transport("5090", "collection", result)
    _transport.safe_extract(archive, destination / "originals")
    originals = destination / "originals"
    final, exit_record = read(originals / "pipeline_receipt.json"), read(originals / "pipeline_exit.json")
    require(final["status"] == "complete" and final["contract_sha256"] == canonical(c)
        and final["actual_native_worker_exit0_count"] == 9 and final["actual_data_worker_exit0_count"] == 6 and final["qualification_exit0_count"] == 3
        and final["owned_pids_absent"] is True and exit_record["actual_os_returncode"] == 0 and exit_record["timeout"] is False,
        "Incomplete collected evaluation")
    require(final["files"], "Empty evaluation original inventory")
    for name, digest in final["files"].items(): require(sha(local_path(originals, name)) == digest, "Evaluation original SHA changed")
    manifest = {str(path.relative_to(originals)): sha(path) for path in originals.rglob("*") if path.is_file()}
    write(destination / "manifest.json", {"files": manifest})
    write(destination / "receipt.json", {"status": "verified", "archive_sha256": sha(archive), "manifest_sha256": sha(destination / "manifest.json"),
        "file_count": len(manifest), "candidate_test_count": 3, "validation_replay_count": 3,
        "actual_data_worker_exit0_count": 6, "qualification_exit0_count": 3, "actual_native_worker_exit0_count": 9,
        "actual_pipeline_exit0_count": 1, "owner_absence_verified": True})
    importlib.import_module("paper.scripts.compare_titantpp_shared_raw_raf3_test").main(["--bundle", str(BUNDLE)])
    return {"status": "complete", "final_receipt": str(BUNDLE / "final_receipt.json")}


def wait_and_advance():
    write(REPORT / "FOLLOWUP_ACTIVE.json", {"pid": os.getpid(), "started_at": now(), "recurring_automation": False})
    parent = read(TRAIN_BUNDLE / "execution_contract.json")
    deadline = parent["hosts"]["5090"]["absolute_host_deadline_unix"]
    while True:
        require(time.time() < deadline, "Inherited deadline expired while waiting raw9")
        require(not (TRAIN_REPORT / "QUEUE_FAILURE.json").exists(), "Raw9 failed; no evaluation retry")
        final_path = TRAIN_BUNDLE / "retrieval/final_receipt.json"
        finisher = read(TRAIN_REPORT / "FINISHER_STARTED.json")
        present = _pid_present(finisher["pid"])
        if final_path.exists() and not present: break
        require(present, "Raw9 finisher exited without complete verified original collection")
        time.sleep(min(30., max(.01, deadline - time.time())))
    prepare()
    return advance()


def start_followup():
    verify_reviewed_implementation()
    require(not any((REPORT / name).exists() for name in ("FOLLOWUP_STARTED.json", "FOLLOWUP_ACTIVE.json", "FAILURE.json")), "Existing follow-up; no duplicate/retry")
    command = [sys.executable, "-u", "-m", MODULE, "--mode", "wait"]
    with (REPORT / "followup.stdout.log").open("xb") as out, (REPORT / "followup.stderr.log").open("xb") as err:
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out, stderr=err, start_new_session=True)
    result = {"pid": process.pid, "command": command, "started_at": now(), "recurring_automation": False,
        "own_os_exit_directly_observed": False, "purpose": "One raw9 completion wait then frozen RAF3 Validation/Test"}
    write(REPORT / "FOLLOWUP_STARTED.json", result)
    return result


_transport.ROOT, _transport.BUNDLE, _transport.REPORT = ROOT, BUNDLE, REPORT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "verify", "advance", "wait", "start-followup", "remote-run"), required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    functions = {"prepare": prepare, "verify": lambda: verify_local(archive=True), "advance": advance,
        "wait": wait_and_advance, "start-followup": start_followup, "remote-run": lambda: remote_run(args.root)}
    try: result = functions[args.mode]()
    except Exception as error:
        target = args.root if args.mode == "remote-run" else REPORT
        if target is not None and not (Path(target) / "FAILURE.json").exists():
            write(Path(target) / "FAILURE.json", {"status": "failed", "at": now(), "error": str(error), "automatic_retry": False})
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
