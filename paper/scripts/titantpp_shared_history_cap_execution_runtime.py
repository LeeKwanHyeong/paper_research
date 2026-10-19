"""Isolated shared-history Cap18 queue, preserving every pre-existing process.

Only this campaign's new children are owned by the bounded, SHA-pinned queue.
The four stopped external-model processes are read and verified, never signalled.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
from pathlib import Path
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import require as _require, sha_file as _sha_file

_BASE_PATH = Path(__file__).with_name("titantpp_quantity_loss_mlp18_runtime.py")
_BASE_SHA = "03a4d8c07e7ee1af2164e2a7b1cfe088a86bc189dc1e35ba014a4e6b2f28e9ec"
_require(_sha_file(_BASE_PATH) == _BASE_SHA, "Preserved bounded queue changed")
_spec = importlib.util.spec_from_file_location("paper.scripts._shared_history_cap_execution_private", _BASE_PATH)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
_impl.SCHEMA = "titantpp_shared_history_cap_execution_v1"
_impl.MODULE = "paper.scripts.titantpp_shared_history_cap_execution_runtime"
_impl.WORKER_MODULE = "paper.scripts.run_titantpp_shared_history_cap_execution"
_impl.ARMS = ("titantpp_history_mlp_shared15_v1", "titantpp_history_mlp_prefix_capacity_v1")
_impl.LOSS_MODES = ("log_mse",)
_PARENT_RAW_SHA = "67bfcad872b68886bf1de09879e031e055c70063bafbf4aa0db42b4ae5ca6fc7"
_HOST_DEADLINE = 1792888885.911408
_AUTHORIZATION = {"real_data_training": True, "real_data_inference": True,
                  "remote_execution": True, "test_evaluation": False}
_MODE_BY_ARM = dict(zip(_impl.ARMS, ("shared", "independent")))
_PREFIX_COUNTS = [0, 2880, 3840, 4800, 5760, 6720, 7680, 8640, 9600]


def validate_contract(contract, host):
    """Admit exactly the approved two arms, Cap3, three seeds, one idle host."""
    r = _impl
    r.require(contract.get("schema") == r.SCHEMA and contract.get("scope") == "execution", "Wrong execution contract")
    r.require(contract.get("execution_authorization") == _AUTHORIZATION, "Execution approval differs")
    r.require(host == "5080" and set(contract["hosts"]) == {"5080"}, "Only 5080 is assigned")
    r.require(contract.get("common_condition") == {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.}, "Common condition changed")
    r.require(len(contract["datasets"]) == 3 and {d["dataset_id"] for d in contract["datasets"]} == set(r.DATASETS), "Cap dataset matrix changed")
    caps = {"yellow_trip_hourly": 65., "raf_spare_parts": 10., "intermittent_frozen_5000": 20.}
    r.require(all(d.get("cap_value") == caps[d["dataset_id"]] and d["loader"]["batch_size"] == 128
                  and d["model"]["hidden_dim"] == 64 and d["optimizer"] == {"grad_clip": 1., "lr": .001, "name": "AdamW", "weight_decay": .01}
                  for d in contract["datasets"]), "Cap/input/head/optimizer configuration changed")
    jobs = contract["jobs"]
    r.require(len(jobs) == len({j["id"] for j in jobs}) == 18, "Expected 18 unique fits")
    r.require({(j["dataset"], j["arm"], j["seed"]) for j in jobs}
              == {(d, a, s) for d in r.DATASETS for a in r.ARMS for s in (42, 52, 62)}, "Fit matrix changed")
    for job in jobs:
        r.alpha_label(job["time_alpha"])
        r.require(type(job["seed"]) is int and job["host"] == "5080" and job["loss_mode"] == "log_mse"
                  and job["time_input"] == "log1p" and job["id"] == f"{job['dataset']}__{job['arm']}__seed{job['seed']}__losslog_mse", "Job identity/condition changed")
    r.require(contract["training"] == {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
              "batch_size": 128, "monitor": "validation_raw_quantity_rmse"}, "Training budget/selection changed")
    r.require(contract["limits"] == {"fit_wall_seconds": 129600, "queue_wall_seconds": 604800, "evaluation_wall_seconds": 7200}, "Limits changed")
    r.require(contract["evaluation_scope"] == "validation_only" and contract["held_out_test_authorized"] is False, "Test forbidden")
    source = contract["source"]
    r.require(source["files"] and r.sha_json(source["files"]) == source["files_sha256"], "Source inventory changed")
    expected_inputs = {f"inputs/data/{d}/{n}" for d in r.DATASETS for n in ("train_validation.parquet", "split_manifest.json")}
    r.require(set(contract["input_files"]) == expected_inputs, "Only six Train/Validation input files are allowed")
    architecture = contract["architecture"]
    expected = {"arms": list(r.ARMS), "hidden_dim": 64, "correction_input_dim": 128, "branch_count": 8,
        "branch_thresholds": [1, 2, 4, 8, 16, 32, 64, 128], "correction_trainable_parameter_count": 9600,
        "residual_divisor": 8, "bias": False, "activation": "exact GELU", "encoder_blocks": 2,
        "time_alpha": 0., "quantity_loss_mode": "log_mse", "quantity_loss_modes": ["log_mse"],
        "quantity_loss_normalization": "none", "model_architecture_changed": True, "checkpoint_warmstart": False,
        "time_input_transform": "log1p_nonnegative_float32", "quantity_input_transform": "log1p_nonnegative_float32"}
    r.require(all(architecture.get(k) == v for k, v in expected.items()), "Shared-history architecture/condition changed")
    r.require(architecture["shared"]["arm"] == r.ARMS[0] and architecture["shared"]["input_width"] == 15
              and architecture["shared"]["input_transform_count"] == 1 and architecture["shared"]["output_widths"] == [15] * 8
              and architecture["independent"]["arm"] == r.ARMS[1] and architecture["independent"]["input_transform_count"] == 8
              and architecture["independent"]["branch_widths"] == [15] + [5] * 7, "Shared/prefix control differs")
    refs = contract["baseline_validation_references"]
    r.require(len(refs) == 9 and {(x["dataset"], x["seed"]) for x in refs} == {(d, s) for d in r.DATASETS for s in (42, 52, 62)}, "Stored MLP16 matrix changed")
    r.require(all(x["time_alpha"] == 0. and x["loss_mode"] == "log_mse" and x["time_input"] == "log1p"
                  and x["time_input_transform"] == "log1p_nonnegative_float32" and x["time_head_trained"] is False
                  and x["retrained"] is False and x["reevaluated"] is False and x["sha256"] for x in refs), "Stored MLP16 references changed")
    h = contract["hosts"][host]
    r.require(all(Path(h[k]).is_absolute() for k in ("root", "source_root", "python"))
              and Path(h["source_root"]) == Path(h["root"]) / "source", "Host paths invalid")
    deadline = h["absolute_host_deadline_unix"]
    r.require(type(deadline) in (int, float) and math.isfinite(deadline) and deadline == _HOST_DEADLINE, "Inherited host deadline changed")
    paused = h["paused_processes"]
    r.require(len(paused) == len({p["pid"] for p in paused}) == 4
              and all(p["state"] in ("T", "t") for p in paused), "Four stopped campaign identities required")
    parent = h["completed_predecessor"]
    r.require(Path(parent["root"]).is_absolute() and parent["root"] != h["root"]
              and parent["contract_raw_sha256"] == _PARENT_RAW_SHA and parent["contract_canonical_sha256"]
              and parent["source_files_sha256"] and parent["dispatcher_identity"]["pid"] > 0
              and parent["dispatcher_exit_sha256"], "Completed 5080 predecessor binding changed")
    return jobs


def predecessor_gate(contract, host, *, require_complete=False):
    """5080 completion only; never inspect or wait for the live 5090 recovery."""
    r = _impl
    validate_contract(contract, host)
    h = contract["hosts"][host]
    binding = h["completed_predecessor"]
    root = Path(binding["root"])
    cp, xp = root / "execution_contract.json", root / "dispatcher_exit.json"
    r.require(r.sha_file(cp) == binding["contract_raw_sha256"], "Predecessor contract bytes changed")
    parent = r.read(cp)
    r.require(r.sha_json(parent) == binding["contract_canonical_sha256"]
              and parent["source"]["files_sha256"] == binding["source_files_sha256"], "Predecessor contract/source binding changed")
    r.require(all(contract[k] == parent[k] for k in ("datasets", "training", "input_files", "common_condition")), "Common source-data/loss/budget differs from parent")
    old_host = parent["hosts"][host]
    r.require(all(old_host[k] == h[k] for k in ("python", "gpu_uuid", "runtime_expected", "paused_processes", "absolute_host_deadline_unix")), "Inherited host/Runtime/stopped identities changed")
    r.require(r.sha_file(xp) == binding["dispatcher_exit_sha256"], "Predecessor exit evidence changed")
    exit_record = r.read(xp)
    r.require(exit_record["actual_os_returncode"] == 0 and exit_record["pid"] == binding["dispatcher_identity"]["pid"]
              and exit_record.get("timeout", False) is False, "Predecessor dispatcher actual exit0 missing")
    r.require(r.proc_identity(exit_record["pid"]) is None, "Predecessor dispatcher still exists")
    # Reading /proc adds no signal or mutation. Check the isolated old source
    # and contract arguments, so an unrecorded old Python child also blocks work.
    old_owners = []
    for pid in r.process_parents():
        identity = r.proc_identity(pid)
        if identity is not None and identity.get("argv"):
            python = "python" in Path(identity["argv"][0]).name
            tied = identity.get("cwd") == old_host["source_root"] or str(cp) in identity["argv"]
            if python and tied:
                old_owners.append(identity)
    r.require(not old_owners, "A predecessor-owned Python process remains")
    gpu, paused = r.gpu_state(contract, host)
    return {"status": "passed", "host": host, "parent_root": str(root),
        "parent_dispatcher_exit_sha256": r.sha_file(xp), "parent_dispatcher_actual_exit0": True,
        "parent_owned_python_pids_absent": True, "current_gpu_allowlist_verified": True,
        "paused_processes": paused, "gpu": gpu, "other_host_inspected_or_changed": False}


def verify_receipt(contract, host, job, stage):
    """Bind the actual selected binary and branch identity before queue admission."""
    r = _impl
    r.require(stage in r.STAGES, "Test stage forbidden")
    root = Path(contract["hosts"][host]["root"])
    path = root / "run" / job["id"] / f"{stage}_receipt.json"
    receipt = r.read(path)
    r.require(receipt.get("schema") == "titantpp_shared_history_cap_execution_stage_v1"
              and receipt.get("status") == "complete" and receipt.get("job_id") == job["id"]
              and receipt.get("stage") == stage and receipt.get("contract_sha256") == r.sha_json(contract)
              and all(receipt.get(k) == job[k] for k in ("dataset", "arm", "seed", "time_alpha", "loss_mode"))
              and receipt.get("held_out_test_evaluated") is False, "Worker stage identity/completion differs")
    r.require(receipt.get("quantity_only") is True and receipt.get("time_head_trained") is False
              and receipt.get("quantity_loss_normalization") == "none", "Quantity-only log loss differs")
    r.verify_time_head_proof(receipt.get("time_head_no_training_verification"))
    r.require(receipt.get("candidate_name") == job["arm"]
              and receipt.get("history_relation_sharing") == _MODE_BY_ARM[job["arm"]]
              and receipt.get("history_relation_parameter_count") == 9600
              and receipt.get("architecture_metadata", {}).get("active_prefix_parameter_counts") == _PREFIX_COUNTS
              and receipt.get("time_input") == "log1p"
              and receipt.get("time_input_transform") == receipt.get("quantity_input_transform") == "log1p_nonnegative_float32", "Shared-history candidate/input proof missing")
    files = receipt.get("files")
    r.require(isinstance(files, dict) and files, "Worker artifact inventory empty")
    job_root = (root / "run" / job["id"]).resolve()
    for name, digest in files.items():
        artifact = r.local_path(root, name)
        r.require(artifact.is_relative_to(job_root) and artifact != path.resolve(), "Artifact escapes job or references itself")
        r.require(r.sha_file(artifact) == digest, "Worker artifact SHA differs: " + name)
    selected = Path(receipt["selected_checkpoint"])
    r.require(selected.is_absolute() and selected.resolve().is_relative_to(job_root), "Selected checkpoint escapes job")
    relative = str(selected.relative_to(root))
    r.require(files.get(relative) == receipt["checkpoint_sha256"] == r.sha_file(selected), "Selected binary missing/changed")
    r.require(isinstance(receipt.get("state_sha256"), str) and len(receipt["state_sha256"]) == 64
              and type(receipt.get("selected_epoch")) is int and 0 < receipt["selected_epoch"] <= 300, "Selected epoch/state missing")
    if stage == "validation":
        train = r.read(path.with_name("train_receipt.json"))
        keys = ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256", "target_identity", "loss_mode",
                "history_relation_sharing", "history_relation_parameter_count", "architecture_metadata")
        r.require(all(train.get(k) == receipt.get(k) for k in keys)
                  and receipt.get("validation_replay_passed") is True, "Validation replay/selection differs")
    return {"path": str(path.relative_to(root)), "sha256": r.sha_file(path)}


def qualification_identity(contract, host):
    r = _impl
    h = contract["hosts"][host]
    expected = [h["python"], "-u", "-m", r.MODULE, "--contract", str(Path(h["root"]) / "execution_contract.json"), "--host", host, "--mode", "qualify"]
    live = r.proc_identity(os.getpid())
    r.require(live is not None and live["pid"] == live["pgid"] == os.getpid()
              and live["cwd"] == str(Path(h["source_root"]).resolve()) and live["argv"] == expected, "Qualification owner identity differs")
    attempt = r.read(Path(h["root"]) / "qualification_attempt.json")
    r.require(attempt["host"] == host and attempt["contract_sha256"] == r.sha_json(contract), "Qualification attempt differs")
    return live


def qualify_three_concurrent(contract, host):
    """Three synthetic CUDA children per arm; only the new child groups are owned."""
    r = _impl
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    owner = qualification_identity(contract, host)
    r.write(root / "qualification_owner_identity.json", {"host": host, "contract_sha256": r.sha_json(contract),
            "qualification_owner_identity": owner, "registered_at": r.now()}, exclusive=True)
    results, all_pids = {}, []
    for arm in r.ARMS:
        children = []
        try:
            for index in range(3):
                prefix = root / f"admission_{arm}_{index}"
                command = [target["python"], "-u", "-c",
                    "import importlib,json,sys; w=importlib.import_module(sys.argv[1]); "
                    "c=json.load(open(sys.argv[2])); r=w.synthetic_condition(c,sys.argv[3],'log_mse',arm=sys.argv[4]); "
                    "print(json.dumps(r,allow_nan=False))", r.WORKER_MODULE, str(root / "execution_contract.json"), host, arm]
                with prefix.with_suffix(".stdout").open("xb") as out, prefix.with_suffix(".stderr").open("xb") as err:
                    process = r.subprocess.Popen(command, cwd=source,
                        env={**os.environ, **target["environment"], "PYTHONDONTWRITEBYTECODE": "1"}, stdout=out, stderr=err, start_new_session=True)
                children.append((process, prefix, command))
                all_pids.append(process.pid)
            rows, started = [], r.time.monotonic()
            for process, prefix, command in children:
                limit = min(contract["limits"]["evaluation_wall_seconds"] - (r.time.monotonic() - started), target["absolute_host_deadline_unix"] - r.time.time())
                timed_out = False
                try:
                    code = process.wait(timeout=max(.001, limit))
                except r.subprocess.TimeoutExpired:
                    timed_out = True
                    r.terminate_owned_child(process)
                    code = process.returncode
                record = _synthetic_exit_record(process, prefix, command, timed_out)
                r.write(prefix.with_suffix(".json"), record, exclusive=True)
                r.require(code == 0 and not timed_out, "Native three-worker admission failed; originals retained")
                result = r.json.loads(prefix.with_suffix(".stdout").read_text())
                r.require(result.get("status") == "passed" and result.get("real_data_loaded") is False
                          and result.get("loss_mode") == "log_mse" and result.get("candidate_name") == arm
                          and result.get("time_input") == "log1p", "Synthetic candidate differs")
                rows.append({**record, "result": result})
            results[arm] = rows
        finally:
            for process, prefix, command in children:
                cleaned = process.poll() is None
                if cleaned:
                    r.terminate_owned_child(process)
                if not prefix.with_suffix(".json").exists():
                    record = _synthetic_exit_record(process, prefix, command, False)
                    record["owned_synthetic_cleanup_after_failure"] = cleaned
                    r.write(prefix.with_suffix(".json"), record, exclusive=True)
    r.require(all(r.proc_identity(pid) is None for pid in all_pids), "Synthetic child remains")
    current = qualification_identity(contract, host)
    r.require(all(current[k] == owner[k] for k in ("pid", "start_ticks", "pgid", "argv", "cwd")), "Qualification owner changed")
    gpu, paused = r.gpu_state(contract, host, (owner["pid"],))
    r.require(not set(all_pids).intersection(gpu["compute_pids"]), "Synthetic CUDA child remains")
    return {"status": "passed", "maximum_simultaneous_workers": 3, "per_arm": results,
        "native_worker_exit0_count": 6, "owned_worker_pids_absent": True,
        "qualification_owner_identity": owner, "qualification_self_context_allowed": True,
        "qualification_self_context_scope": "only exact current qualification child", "gpu_after": gpu, "paused_after": paused}


def _synthetic_exit_record(process, prefix, command, timed_out):
    r = _impl
    return {"pid": process.pid, "command": command, "actual_os_returncode": process.returncode, "timeout": timed_out,
        "stdout_sha256": r.sha_file(prefix.with_suffix(".stdout")), "stderr_sha256": r.sha_file(prefix.with_suffix(".stderr")),
        "stdout_bytes": prefix.with_suffix(".stdout").stat().st_size, "stderr_bytes": prefix.with_suffix(".stderr").stat().st_size,
        "observed_exit_at": r.now()}


def qualify(contract, host):
    r = _impl
    validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    r.require(Path.cwd().resolve() == source.resolve(), "Qualification cwd differs")
    r.require(not any((root / p).exists() for p in ("qualification_attempt.json", "qualification.json", "start.json", "run", "dispatch_attempt.json")), "Existing qualification/start/run; no automatic retry")
    r.verify_files(contract, source)
    for name in contract.get("packaging", {}).get("required_empty_source_directories", []):
        r.require(r.local_path(source, name).is_dir(), "Required empty source directory missing")
    r.write(root / "qualification_attempt.json", {"host": host, "started_at": r.now(), "contract_sha256": r.sha_json(contract)}, exclusive=True)
    r.require(r.time.time() < target["absolute_host_deadline_unix"], "Inherited deadline expired")
    gate = r.predecessor_gate(contract, host, require_complete=True)
    gpu, paused = r.gpu_state(contract, host)
    free_before = r.free_gpu_bytes(target["gpu_uuid"])
    runtime = r.runtime_check(target)
    synthetic = importlib.import_module(r.WORKER_MODULE).synthetic_check(contract, host)
    r.require(synthetic.get("status") == "passed" and synthetic.get("real_data_loaded") is False, "Native synthetic qualification failed")
    peaks = synthetic["cuda_memory_peaks"]
    r.require(set(peaks) == set(r.ARMS) and all(type(p.get("max_memory_reserved_bytes")) is int and p["max_memory_reserved_bytes"] > 0 for p in peaks.values()), "Both-arm memory measurement missing")
    peak = max(p["max_memory_reserved_bytes"] for p in peaks.values())
    r.require(3 * peak < .85 * free_before, "Three-worker free-memory admission failed")
    concurrent = r.qualify_three_concurrent(contract, host)
    r.paused_processes(contract, host)
    result = {"status": "passed", "host": host, "verified_at": r.now(), "contract_sha256": r.sha_json(contract),
        "predecessor_5080_gate": gate, "gpu_before": gpu, "paused_processes": paused, "runtime": runtime, "synthetic": synthetic,
        "three_worker_native_admission": concurrent, "parallel_memory_admission": {"maximum_workers": 3,
            "maximum_reserved_bytes_per_worker": peak, "three_worker_reserved_estimate_bytes": 3 * peak,
            "free_memory_before_qualification_bytes": free_before, "fraction_limit": .85, "passed": True,
            "limitation": "Per-process full-shape peaks estimate memory; separate three-process synthetic checks cover both arms."}}
    r.require(r.time.time() < target["absolute_host_deadline_unix"], "Deadline exceeded during qualification")
    r.write(root / "qualification.json", result, exclusive=True)
    return result


_impl.validate_contract = validate_contract
_impl.verify_receipt = verify_receipt
_impl.predecessor_gate = predecessor_gate
_impl.qualify = qualify
_impl.qualify_three_concurrent = qualify_three_concurrent
for _name, _value in tuple(_impl.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    try:
        raise SystemExit(_impl.main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
