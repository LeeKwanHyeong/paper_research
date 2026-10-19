"""Approved shared-history18 on an idle5080; preserve every prior campaign.

The bounded transport/archive primitives are inherited byte-for-byte. Scientific
preparation stays sealed; this separate execution contract grants only the
latest approved training and selected-checkpoint Validation, never Test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from paper.scripts import prepare_titantpp_shared_history_cap as preparation
from paper.scripts import titantpp_shared_history_cap_execution_runtime as runtime

_BASE = Path(__file__).with_name("manage_titantpp_quantity_loss_mlp18.py")
_BASE_SHA = "331aa17a98532217bb4cdda8865cf09530cc54b5c31f4968456655e93be27e8d"
if hashlib.sha256(_BASE.read_bytes()).hexdigest() != _BASE_SHA:
    raise ValueError("Preserved manager changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_base", _BASE)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
ROOT = Path(__file__).resolve().parents[2]
NAME = preparation.NAME
BUNDLE = ROOT / "search_artifacts" / NAME / "execution_v1"
REPORT = ROOT / "reports" / NAME / "execution_v1"
MODULE = "paper.scripts.titantpp_shared_history_cap_execution_runtime"
NEW_SOURCE = (
    "paper/scripts/manage_titantpp_shared_history_cap_execution.py",
    "paper/scripts/titantpp_shared_history_cap_execution_runtime.py",
    "paper/scripts/run_titantpp_shared_history_cap_execution.py",
    "paper/scripts/compare_titantpp_shared_history_cap_execution.py",
)
AUTHORIZATION = {"real_data_training": True, "real_data_inference": True,
                 "remote_execution": True, "test_evaluation": False}
read, sha, canonical, require = runtime.read, runtime.sha_file, runtime.sha_json, runtime.require
write, now = _impl.write, _impl.now


def source_snapshot(prepared):
    required = set(NEW_SOURCE)
    tracked = set(subprocess.check_output(["git", "ls-files", "--", *NEW_SOURCE], cwd=ROOT, text=True).splitlines())
    require(tracked == required and not subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *NEW_SOURCE], cwd=ROOT, text=True).strip(),
        "Commit new execution files before preparing")
    files = {**prepared["source"]["files"], **{n: sha(ROOT / n) for n in NEW_SOURCE}}
    for n, h in files.items():
        require(sha(ROOT / n) == h, "Preserved source changed: " + n)
    return {"files": files, "files_sha256": canonical(files),
            "base_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "preserved_preparation_file_count": 966, "new_execution_source_count": 4}


def build_contract(prepared, parent, observation, source):
    require(observation["actual_transport_exit_code"] == 0, "Initial5080 query failed")
    snapshot = observation["observation"]
    require(snapshot["host"] == "5080" and snapshot["gpu"]["exit_code"] == 0
            and snapshot["gpu_processes"]["exit_code"] == 0, "GPU identity unavailable")
    h = deepcopy(parent["hosts"]["5080"])
    previous = deepcopy(h)
    root = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{NAME}_execution_v1_5080"
    xp = snapshot["predecessor_terminal"]["dispatcher_exit.json"]
    exit_record = json.loads(xp)
    require(exit_record["actual_os_returncode"] == 0, "5080 predecessor is not complete")
    h.update(root=root, source_root=root + "/source", tmux="shared_history18_5080_20261019_v1",
             completed_predecessor={"root": previous["root"],
                 "contract_raw_sha256": preparation.PARENT_SHA,
                 "contract_canonical_sha256": canonical(parent),
                 "source_files_sha256": parent["source"]["files_sha256"],
                 "dispatcher_identity": {"pid": exit_record["pid"]},
                 "dispatcher_exit_sha256": hashlib.sha256(xp.encode()).hexdigest()})
    h["environment"].update(SOURCE_REVISION=source["base_git_revision"], PYTHONPATH=h["source_root"],
                             MPLCONFIGDIR=root + "/cache/matplotlib", XDG_CACHE_HOME=root + "/cache")
    architecture = deepcopy(read(preparation.SPEC)["architecture"])
    architecture.update(arms=list(preparation.ARMS), time_alpha=0., quantity_loss_mode="log_mse",
        quantity_loss_modes=["log_mse"], quantity_loss_normalization="none",
        time_input_transform="log1p_nonnegative_float32", quantity_input_transform="log1p_nonnegative_float32",
        model_architecture_changed=True, checkpoint_warmstart=False)
    datasets = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
    jobs = [{"id": f"{d}__{a}__seed{s}__losslog_mse", "dataset": d, "arm": a,
             "seed": s, "host": "5080", **preparation.CONDITION}
            for d in datasets for a in preparation.ARMS for s in (42, 52, 62)]
    return {"schema": "titantpp_shared_history_cap_execution_v1", "scope": "execution",
        "campaign": NAME + "/execution_v1", "created_at": now(), "source": source,
        "science_spec_sha256": preparation.SPEC_SHA, "design_sha256": canonical(architecture),
        "architecture": architecture, "common_condition": deepcopy(preparation.CONDITION),
        "hosts": {"5080": h}, "jobs": jobs, "datasets": deepcopy(prepared["datasets"]),
        "training": deepcopy(prepared["training"]),
        "input_files": {n: v["sha256"] for n, v in prepared["input_files"].items()},
        "baseline_validation_references": deepcopy(prepared["baseline_validation_references"]),
        "limits": {"fit_wall_seconds": 129600, "queue_wall_seconds": 604800, "evaluation_wall_seconds": 7200},
        "execution_authorization": AUTHORIZATION, "evaluation_scope": "validation_only",
        "held_out_test_authorized": False, "preparation_contract_sha256": canonical(prepared),
        "original_preparation_contract_path": str(preparation.BUNDLE / "preparation_contract.json"),
        "initial5080_observation_sha256": sha(REPORT / "INITIAL_5080.json"),
        "approval": "2026-10-19 user approved two candidates x Cap3 x3seed=18, only idle5080 max3 parallel, keep current training/common conditions/results, compare stored MLP16 Validation RMSE/MAE and everyseed difference",
        "current_training_modified": False, "historical_baselines_retrained_or_reevaluated": False,
        "maximum_parallel_workers": 3, "recurring_automation_created": False}


def prepare():
    preparation.verify()
    require(not BUNDLE.exists(), "Execution bundle already exists; no overwrite or automatic restart")
    prepared = read(preparation.BUNDLE / "preparation_contract.json")
    parent, _, _ = preparation.verify_parent()
    source = source_snapshot(prepared)
    c = build_contract(prepared, parent, read(REPORT / "INITIAL_5080.json"), source)
    require(time.time() < c["hosts"]["5080"]["absolute_host_deadline_unix"], "Inherited host deadline expired")
    runtime.validate_contract(c, "5080")
    BUNDLE.mkdir(parents=True, exist_ok=False)
    for name, binding in prepared["input_files"].items():
        target = BUNDLE / "input_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(binding["path"], target)
        require(sha(target) == binding["sha256"], "Input copy changed")
    shutil.copyfile(preparation.SPEC, REPORT / "SCIENCE_SPEC.json")
    write(BUNDLE / "execution_contract.json", c)
    archive = _impl.package_archive(c, source)
    receipt = {"status": "prepared", "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": canonical(c), "archive_sha256": sha(archive),
        "new_fit_count": 18, "baseline_reference_count": 9, "source_file_count": len(source["files"]),
        "host": "5080", "max_parallel": 3, "held_out_test_authorized": False}
    write(REPORT / "PREPARED.json", receipt)
    return receipt


def verify_local(*, archive=False):
    preparation.verify()
    c = _base_verify_local(archive=archive)
    prepared = read(preparation.BUNDLE / "preparation_contract.json")
    require(c["datasets"] == prepared["datasets"] and c["training"] == prepared["training"]
            and c["baseline_validation_references"] == prepared["baseline_validation_references"],
            "Common dataset/head/budget/baseline changed")
    for name, h in c["source"]["files"].items(): require(sha(ROOT / name) == h, "Execution source changed: " + name)
    return c


def verify_collected(c, host, originals):
    final = _base_verify_collected(c, host, originals)
    for j in c["jobs"]:
        for stage in runtime.STAGES:
            value = read(Path(originals) / "run" / j["id"] / f"{stage}_receipt.json")
            require(value.get("candidate_name") == j["arm"]
                    and value.get("history_relation_parameter_count") == 9600
                    and value.get("time_input") == "log1p", "Collected shared-history identity changed")
    return final


def deploy(host="5080"):
    c = verify_local(archive=True); h = c["hosts"][host]
    write(BUNDLE / "attempts" / f"{host}.json", {"host": host, "at": now(), "automatic_retry": False})
    install = "mkdir " + shlex.quote(h["root"]) + " && tar -xzf - -C " + shlex.quote(h["root"])
    _impl.save_transport(host, "install", _impl.ssh(host, install, stdin=(BUNDLE / "deployment.tar.gz").read_bytes(), timeout=300))
    command = [h["python"], "-u", "-m", MODULE, "--contract", h["root"] + "/execution_contract.json",
               "--host", host, "--mode", "launch_dispatch"]
    supervisor = ("import os,subprocess,json,pathlib,datetime,sys\n"
        f"p=subprocess.Popen({command!r},cwd={h['source_root']!r},env={{**os.environ,**{h['environment']!r}}})\n"
        "code=p.wait()\n"
        f"with (pathlib.Path({h['root']!r})/'dispatcher_exit.json').open('x') as f:\n"
        " json.dump({'pid':p.pid,'actual_os_returncode':code,'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor_own_exit_directly_observed':False},f);f.write(chr(10))\n"
        "sys.exit(code)\n")
    shell = shlex.join([h["python"], "-u", "-c", supervisor]) + " > " + shlex.quote(h["root"] + "/dispatcher.log") + " 2>&1"
    _impl.save_transport(host, "launch", _impl.ssh(host, shlex.join([h["tmux_binary"], "new-session", "-d", "-s", h["tmux"], shell])))
    result = {"status": "launch_requested_not_yet_verified", "host": host, "at": now(), "planned_fit_count": 18}
    write(REPORT / "LAUNCHED_5080.json", result)
    return result


def finish():
    """A single finite remote wait, then one immutable collection and comparison."""
    c = verify_local(); h = c["hosts"]["5080"]
    program = ("import pathlib,time,json,sys\n"
        f"root=pathlib.Path({h['root']!r});deadline={h['absolute_host_deadline_unix']!r}\n"
        "while not (root/'dispatcher_exit.json').exists():\n"
        " if time.time()>=deadline:raise TimeoutError('Inherited deadline expired')\n"
        " time.sleep(min(30.,max(.01,deadline-time.time())))\n"
        "x=json.loads((root/'dispatcher_exit.json').read_text())\n"
        "print(json.dumps(x));sys.exit(0 if x['actual_os_returncode']==0 else 1)\n")
    result = _impl.ssh("5080", shlex.join([h["python"], "-c", program]), timeout=max(1., h["absolute_host_deadline_unix"] - time.time() + 60))
    _impl.save_transport("5080", "finite_wait_complete", result)
    collected = _impl.collect("5080")
    from paper.scripts.compare_titantpp_shared_history_cap_execution import main as compare
    compare()
    return {"status": "complete", "collection": collected, "final_receipt": str(BUNDLE / "retrieval/final_receipt.json")}


def start_finisher():
    verify_local()
    require(not (REPORT / "FINISHER_STARTED.json").exists(), "Finite completion already started")
    command = [sys.executable, "-u", "-m", "paper.scripts.manage_titantpp_shared_history_cap_execution", "--mode", "finish"]
    with (REPORT / "finish.stdout.log").open("xb") as out, (REPORT / "finish.stderr.log").open("xb") as err:
        p = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                             stdout=out, stderr=err, start_new_session=True)
    value = {"pid": p.pid, "command": command, "started_at": now(), "recurring_automation": False,
             "purpose": "One finite wait then collect18 and compare9stored baselines", "own_os_exit_directly_observed": False}
    write(REPORT / "FINISHER_STARTED.json", value)
    return value


_impl.ROOT, _impl.NAME, _impl.BUNDLE, _impl.REPORT, _impl.MODULE, _impl.runtime = ROOT, NAME, BUNDLE, REPORT, MODULE, runtime
_base_verify_local = _impl.verify_local
# Preserve the collector implementation, replacing only this campaign's receipt schema.
_verifier_source = __import__("inspect").getsource(_impl.verify_collected).replace(
    '"titantpp_quantity_loss_mlp18_stage_v1"', '"titantpp_shared_history_cap_execution_stage_v1"')
exec(compile(_verifier_source.replace("def verify_collected(", "def _shared_base_verify_collected(", 1), str(_BASE), "exec"), _impl.__dict__)
_base_verify_collected = _impl._shared_base_verify_collected
_old_collection_program = _impl.collection_program
def collection_program(c, host):
    return _old_collection_program(c, host).replace(
        "from paper.scripts import titantpp_quantity_loss_mlp18_runtime as r", "from paper.scripts import titantpp_shared_history_cap_execution_runtime as r")
_impl.verify_local, _impl.verify_collected, _impl.collection_program = verify_local, verify_collected, collection_program
status, collect, safe_extract = _impl.status, _impl.collect, _impl.safe_extract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "verify", "deploy", "status", "collect", "start-finisher", "finish"), required=True)
    a = parser.parse_args(argv)
    try:
        value = {"prepare": prepare, "verify": verify_local, "deploy": deploy,
                 "status": lambda: status("5080"), "collect": lambda: collect("5080"),
                 "start-finisher": start_finisher, "finish": finish}[a.mode]()
    except Exception as e:
        if a.mode == "finish" and not (REPORT / "FINISH_FAILURE.json").exists():
            write(REPORT / "FINISH_FAILURE.json", {"status": "failed", "at": now(), "error": str(e), "automatic_retry": False})
        raise
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
