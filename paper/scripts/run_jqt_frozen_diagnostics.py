#!/usr/bin/env python3
"""Read-only J/Q/T checkpoint diagnosis. Default: metadata validation only.

Run this separate evaluator against the immutable CUDA source, never copy it
over the training runner. Execution requires an approved, hashed contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REVISION = "76426fd981e624144947fd582f794c2832e9fe05"
SOURCE_IDENTITY_SHA256 = "79214b78ace5bc802646db5ec36e654347d5529b2c2cea10db3e492be38b53c9"
DATASETS = ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
BOUNDARIES = ([2, 31, 46, 187], [7, 686, 1562, 3449], [8, 20, 25, 35])
HELPERS = ("run_jqt_frozen_diagnostics.py", "jqt_quantity_analysis.py", "jqt_time_analysis.py")
PACKAGES = {"paper", "models", "data_loader", "simple_lab_test", "utils"}
ENVIRONMENT_KEYS = {"CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


class BudgetedFile:
    """A seekable archive writer that rejects writes before exceeding its budget."""
    def __init__(self, handle, limit):
        self.handle, self.limit, self.high_water = handle, limit, 0

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def write(self, data):
        end = max(self.high_water, self.handle.tell() + len(data))
        require(end <= self.limit, "Output size limit reached")
        result = self.handle.write(data)
        self.high_water = end
        return result


def bounded_artifact(path, maximum, writer, *, reserve=64 * 1024, expected_size=None):
    """Reserve 64 KiB for terminal evidence and preserve partial files on failure."""
    path = Path(path)
    used = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    remaining = maximum - reserve - used
    require(remaining > 0, "Output size limit reached")
    require(expected_size is None or expected_size <= remaining, "Output size limit reached")
    with path.open("xb") as handle:
        writer(BudgetedFile(handle, remaining))


def write_budgeted_json(path, value, maximum, *, terminal=False):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    bounded_artifact(path, maximum, lambda handle: handle.write(raw),
                     reserve=0 if terminal else 64 * 1024, expected_size=len(raw))


def prepare_contract(evidence):
    """Only small, previously retrieved JSON is read; no data/torch imports."""
    evidence = Path(evidence).resolve()
    suite = read_json(evidence / "remote_receipts/suite_contract.json")
    terminal = evidence / "monitor/20260910T233614Z_terminal"
    status = read_json(terminal / "suite_run/status.json")
    require(suite["source"]["revision"] == REVISION and status["status"] == "complete",
            "Expected completed frozen suite")
    require(status["held_out_test_evaluated"] is False, "Wrong evaluation scope")
    remote = Path(suite["source"]["root"]).parent
    datasets = []
    for number, (name, boundaries) in enumerate(zip(DATASETS, BOUNDARIES), 1):
        prefix = Path(f"suite_run/{number:02d}_{name}")
        training = read_json(evidence / f"remote_receipts/contracts/{name}.json")
        wrapper = read_json(terminal / prefix / "wrapper_manifest.json")
        require(wrapper["contract"] == training, "Training contract mismatch")
        old = read_json(ROOT / f"paper/results/final_backbone_and_baselines_20260909/B_reference/{name}/launch_contract.json")
        require(old["data_sha256"] == training["data"]["sha256"] and
                old["quantity_contract"]["boundaries"] == boundaries, "Train boundaries drift")
        require(all(old["validation_target_population"][k] == v for k, v in
                    training["populations"]["validation"].items()), "Population drift")
        states = []
        for objective in ("joint", "quantity_only", "time_only"):
            arm = terminal / prefix / objective
            summary = read_json(arm / "summary.json")
            arm_contract = read_json(arm / "contract.json")
            require(summary["status"] == "complete" and summary["epochs_completed"] == 120,
                    "Incomplete arm")
            require(summary["contract_sha256"] == json_digest(arm_contract), "Arm identity drift")
            choices = [("quantity", "raw_quantity_rmse")] if objective != "time_only" else []
            if name == "yellow_trip_hourly" and objective != "quantity_only":
                choices += [("time", "legacy_time_loss"), ("time", "last")]
            for purpose, selector in choices:
                selected = summary["selectors"].get(selector)
                epoch = selected["best_epoch"] if selected else 120
                states.append({
                    "id": f"{objective}_{selector}", "objective": objective, "purpose": purpose,
                    "selector": selector, "epoch": epoch,
                    "state_sha256": selected["state_sha256"] if selected else summary["last_state_sha256"],
                    "path": str(remote / prefix / objective /
                                (f"best_{selector}_model.pt" if selected else "last_epoch_state.pt")),
                    "arm_contract": arm_contract, "expected": summary["history"][epoch - 1],
                })
        datasets.append({"dataset_id": name, "training_contract": training,
                         "training_contract_path": str(remote / f"contracts/{name}.json"),
                         "train_log_mean": wrapper["train_log_mean"],
                         "train_log_std": wrapper["train_log_std"],
                         "quantity_boundaries": boundaries,
                         "history_boundaries": [1, 3, 7, 15, 31] if number == 3 else [64, 128],
                         "states": states})
    result = {
        "schema": "jqt_frozen_diagnosis_v1", "authorization": "pending",
        "source": suite["source"], "server": "5090", "device": "cuda:0", "threads": 4,
        "python_executable": "/opt/miniconda3/envs/ai_env/bin/python3.12",
        "output_dir": str(remote.parent / "jqt_frozen_diagnosis_v1"),
        "max_wall_seconds": 7200, "max_output_bytes": 512 * 2**20,
        "max_concurrent_gpu_jobs": 1, "optimizer_updates": 0, "held_out": False,
        "quantity_split": "validation", "quantity_boundary_population": "all_train_rows",
        "quantity_quantiles": [0.5, 0.9, 0.95, 0.99], "quantile_interpolation": "nearest",
        "bin_boundary_semantics": "equality_in_lower_bin",
        "time_train_sample": {"count": 2048, "seed": 42, "order": "sorted_canonical_indices"},
        "replay": {"atol": 1e-5, "rtol": 1e-5, "max_pair_delta_distortion": 0.001},
        "evaluator_files": {name: digest(Path(__file__).parent / name) for name in HELPERS},
        "evidence": {"suite_contract_sha256": digest(evidence / "remote_receipts/suite_contract.json"),
                     "completion_status_sha256": digest(terminal / "suite_run/status.json")},
        "datasets": datasets,
    }
    validate_contract(result)
    return result


def validate_contract(c, *, execute=False):
    require(c["schema"] == "jqt_frozen_diagnosis_v1", "Unknown evaluation contract")
    require(c["authorization"] in {"pending", "approved"}, "Unknown authorization")
    require(not execute or c["authorization"] == "approved", "Execution authorization is pending")
    require(c["source"]["revision"] == REVISION, "Source revision changed")
    require(json_digest(c["source"]) == SOURCE_IDENTITY_SHA256, "Frozen source identity changed")
    require(c["server"] == "5090" and c["device"] == "cuda:0" and c["threads"] == 4,
            "Only the specified frozen 5090 runtime is allowed")
    require(c["python_executable"] == "/opt/miniconda3/envs/ai_env/bin/python3.12", "Interpreter path drift")
    require(c["optimizer_updates"] == 0 and c["held_out"] is False and
            c["quantity_split"] == "validation", "Evaluation scope changed")
    require(c["max_concurrent_gpu_jobs"] == 1 and type(c["max_wall_seconds"]) is int and
            0 < c["max_wall_seconds"] <= 7200 and type(c["max_output_bytes"]) is int and
            2**20 <= c["max_output_bytes"] <= 512 * 2**20, "Invalid execution limits")
    require(c["replay"] == {"atol": 1e-5, "rtol": 1e-5, "max_pair_delta_distortion": 0.001},
            "Replay rule changed")
    require(c["time_train_sample"] == {"count": 2048, "seed": 42, "order": "sorted_canonical_indices"},
            "Train diagnostic sample changed")
    require([d["dataset_id"] for d in c["datasets"]] == list(DATASETS), "Dataset scope changed")
    require(c["quantity_boundary_population"] == "all_train_rows" and
            c["quantity_quantiles"] == [0.5, 0.9, 0.95, 0.99] and
            c["quantile_interpolation"] == "nearest" and
            c["bin_boundary_semantics"] == "equality_in_lower_bin", "Binning rule changed")
    for i, d in enumerate(c["datasets"]):
        tc = d["training_contract"]
        require(tc["dataset_id"] == DATASETS[i] and tc["source"]["revision"] == REVISION and
                tc["source"]["files"] == c["source"]["files"], "Dataset source drift")
        require(tc["seed"] == 42 and tc["epochs"] == 120 and
                tc["loader"]["batch_size"] == 128 and tc["loader"]["num_workers"] == 0 and
                tc["loader"]["validation_shuffle"] is False and tc["loader"]["drop_last"] is False,
                "Frozen training/loader scope changed")
        require(tc["runtime"] == c["datasets"][0]["training_contract"]["runtime"] and
                set(tc["runtime"]["environment"]) == ENVIRONMENT_KEYS, "Runtime contract drift")
        remote = Path(c["source"]["root"]).parent
        prefix = remote / f"suite_run/{i + 1:02d}_{DATASETS[i]}"
        require(d["training_contract_path"] == str(remote / f"contracts/{DATASETS[i]}.json"),
                "Training contract path changed")
        require(d["quantity_boundaries"] == BOUNDARIES[i] and d["history_boundaries"] ==
                ([1, 3, 7, 15, 31] if i == 2 else [64, 128]), "Strata drift")
        expected = {("quantity", "joint", "raw_quantity_rmse"),
                    ("quantity", "quantity_only", "raw_quantity_rmse")}
        if i == 1:
            expected |= {("time", o, s) for o in ("joint", "time_only")
                         for s in ("legacy_time_loss", "last")}
        require(len(d["states"]) == len(expected) and
                {(s["purpose"], s["objective"], s["selector"]) for s in d["states"]} == expected,
                "Checkpoint scope changed")
        for s in d["states"]:
            ac = s["arm_contract"]
            require(ac["identity"]["contract"] == tc and ac["objective"] == s["objective"],
                    "Checkpoint training identity drift")
            filename = "last_epoch_state.pt" if s["selector"] == "last" else f"best_{s['selector']}_model.pt"
            require(s["id"] == f"{s['objective']}_{s['selector']}" and
                    s["path"] == str(prefix / s["objective"] / filename), "Checkpoint path/ID drift")
            require(isinstance(s["state_sha256"], str) and len(s["state_sha256"]) == 64 and
                    all(char in "0123456789abcdef" for char in s["state_sha256"]), "Invalid state digest")
            require(s["expected"]["epoch"] == s["epoch"] and s["expected"]["validation_count"] ==
                    tc["populations"]["validation"]["target_count"], "Expected metric scope drift")
            batches = math.ceil(tc["populations"]["train"]["target_count"] / 128)
            require(s["expected"]["global_step"] == s["epoch"] * batches and
                    s["expected"]["train_count"] == tc["populations"]["train"]["target_count"] and
                    s["expected"]["train_batches"] == batches and
                    s["expected"]["validation_batches"] == math.ceil(s["expected"]["validation_count"] / 128),
                    "Expected step/population drift")
            require(s["epoch"] == 120 if s["selector"] == "last" else 1 <= s["epoch"] <= 120,
                    "Invalid epoch")
    require(set(c["evaluator_files"]) == set(HELPERS), "Evaluator closure changed")
    for name, sha in c["evaluator_files"].items():
        require(digest(Path(__file__).parent / name) == sha, f"Evaluator changed: {name}")


def load_helper(name):
    spec = importlib.util.spec_from_file_location("_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_checkpoint_payload(payload, state, hash_state):
    require(payload["schema_version"] == "time_quantity_diagnostic_cuda_v2", "Checkpoint schema mismatch")
    require(payload["contract_sha256"] == json_digest(state["arm_contract"]), "Checkpoint identity mismatch")
    if state["selector"] == "last":
        require(payload["contract"] == state["arm_contract"] and payload["epoch"] == state["epoch"],
                "Last checkpoint scope mismatch")
        recorded = payload["model_state_sha256"]
    else:
        require(payload["selector"] == state["selector"] and payload["applicable"] is True and
                payload["best_epoch"] == state["epoch"] and
                payload["best_value"] == state["expected"][state["selector"]], "Selector mismatch")
        recorded = payload["state_sha256"]
    require(payload["global_step"] == state["expected"]["global_step"], "Checkpoint step mismatch")
    require(hash_state(payload["model_state_dict"]) == recorded == state["state_sha256"],
            "Checkpoint tensor state mismatch")
    return payload["model_state_dict"]


def load_checkpoint(path):
    """Allow only the concrete NumPy RNG constructors needed by frozen last states."""
    import numpy as np
    import torch
    safe = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype("uint32")), np._core.numeric._frombuffer]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


def collect_rows(model, loader, engine, timing, purpose, objective, device, limit):
    """Collect one immutable state, retaining the original per-target outputs."""
    import numpy as np
    import torch
    require(not model.training and purpose in {"quantity", "time"}, "Invalid evaluation mode")
    columns = {}
    with torch.no_grad():
        for batch in loader:
            limit()
            args = engine._batch_tensors(batch, device)
            if purpose == "quantity":
                out = engine.task_outputs(model, *args, objective)
                out = {k: out[k] for k in ("true_qty", "pred_qty", "log_qty_loss", "history_length")}
            else:
                out = timing.inspect_legacy_time_batch(model, *args, objective)
            for key, value in out.items():
                require(isinstance(value, torch.Tensor) and value.ndim == 1 and
                        value.numel() == len(args[0]), f"Unexpected batch field/shape: {key}")
                require(bool(torch.isfinite(value).all()), f"Nonfinite batch field: {key}")
                columns.setdefault(key, []).append(value.detach().cpu().numpy())
    require(bool(columns), "No evaluation batches")
    rows = {k: np.concatenate(v) for k, v in columns.items()}
    require(len(rows["history_length"]) == len(loader.dataset), "Evaluated target count drift")
    return rows


def quantity_metrics(rows):
    import numpy as np
    e = rows["pred_qty"].astype(np.float64) - rows["true_qty"].astype(np.float64)
    return {"raw_quantity_rmse": float(np.sqrt(np.mean(e * e))),
            "quantity_mae": float(np.mean(abs(e))),
            "quantity_train_loss": float(np.mean(rows["log_qty_loss"].astype(np.float64)))}


def verify_time_replay(rows, expected):
    import numpy as np
    actual = float(np.mean(rows["original_legacy_loss"].astype(np.float64)))
    require(math.isclose(actual, expected["legacy_time_loss"], abs_tol=1e-5, rel_tol=1e-5),
            "Time metric replay failed")
    return {"passed": True, "actual": actual, "expected": expected["legacy_time_loss"]}


def worker_environment(c):
    """Set numerical settings before interpreter/import initialization."""
    env = os.environ.copy()
    for key, value in c["datasets"][0]["training_contract"]["runtime"]["environment"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["JQT_DIAGNOSTIC_CONTRACT_SHA256"] = json_digest(c)
    return env


def record_failure(c, pid, error):
    """Append failure evidence only to the directory owned by this invocation."""
    output = Path(c["output_dir"])
    start = output / "started.json"
    if not start.is_file():
        return
    receipt = read_json(start)
    if receipt.get("pid") != pid or receipt.get("contract_sha256") != json_digest(c):
        return
    try:
        complete = read_json(output / "complete.json")
        completed = complete.get("status") == "complete" and complete.get("contract_sha256") == json_digest(c)
    except (OSError, ValueError):
        completed = False
    if not (output / "failed.json").exists() and not completed:
        write_budgeted_json(output / "failed.json", {"status": "failed", "error": str(error)[:4096],
                            "contract_sha256": json_digest(c), "automatic_retry": False},
                            c["max_output_bytes"], terminal=True)


class SupervisorTerminated(Exception):
    pass


def supervise(c, contract_path):
    """Own the worker's process group and enforce a deadline from process launch."""
    def terminated(signum, frame):
        raise SupervisorTerminated(f"Supervisor received signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminated)
    proc = None
    try:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--contract",
                                 str(Path(contract_path).resolve()), "--worker"], start_new_session=True,
                                 env=worker_environment(c))
        try:
            code = proc.wait(timeout=c["max_wall_seconds"])
        except (subprocess.TimeoutExpired, KeyboardInterrupt, SupervisorTerminated) as error:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            record_failure(c, proc.pid, error)
            raise
        if code != 0:
            message = f"Evaluator stopped with exit code {code}; no automatic retry"
            record_failure(c, proc.pid, message)
            raise ValueError(message)
    finally:
        signal.signal(signal.SIGTERM, previous)


def execute(c):
    """Called in an isolated supervised process only after authorization."""
    validate_contract(c, execute=True)
    require(os.environ.get("JQT_DIAGNOSTIC_CONTRACT_SHA256") == json_digest(c),
            "Execution requires the supervised --execute entry point")
    require(Path(sys.executable).resolve() == Path(c["python_executable"]).resolve(),
            "Interpreter executable differs")
    source = Path(c["source"]["root"]).resolve()
    require(subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            == REVISION, "Frozen checkout revision mismatch")
    for name, sha in c["source"]["files"].items():
        require(digest(source / name) == sha, f"Frozen source changed: {name}")
    require(not any(name.split(".")[0] in PACKAGES for name in sys.modules),
            "Run in a clean process without imported project modules")
    output = Path(c["output_dir"]).resolve()
    require(not output.exists() and not output.is_relative_to(source), "Refusing output overwrite/source write")
    output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output.parent / "jqt_frozen_diagnosis_gpu0.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    busy = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    require(not busy, "GPU has an existing compute process; no automatic retry")
    runtime = c["datasets"][0]["training_contract"]["runtime"]
    require(all(os.environ.get(key) == value for key, value in runtime["environment"].items()),
            "Worker numerical environment differs")
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from paper.scripts import run_time_quantity_diagnostic as frozen
    from paper.scripts import time_quantity_diagnostic as engine
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    frozen.configure_runtime(c["device"], c["threads"])
    require(frozen.current_runtime(c["device"]) == runtime, "Frozen numerical runtime differs")
    frozen.validate_import_origins()
    quantity = load_helper("jqt_quantity_analysis")
    timing = load_helper("jqt_time_analysis")
    output.mkdir()
    write_budgeted_json(output / "started.json", {"pid": os.getpid(), "contract_sha256": json_digest(c)},
                        c["max_output_bytes"])
    def save_json(path, value):
        write_budgeted_json(path, value, c["max_output_bytes"])
    save_json(output / "contract.json", c)
    started = time.monotonic()

    def limit():
        require(time.monotonic() - started < c["max_wall_seconds"], "Wall-time limit reached")
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= c["max_output_bytes"],
                "Output size limit reached")

    results = []
    for d in c["datasets"]:
        limit()
        frame, metadata = frozen.validate_inputs(Path(d["training_contract_path"]),
                                                 d["training_contract"], c["device"])
        require(metadata["train_log_mean"] == d["train_log_mean"] and
                metadata["train_log_std"] == d["train_log_std"], "Initialization metadata drift")
        paired, expected = {}, {}
        for state in d["states"]:
            limit()
            file_hash = digest(state["path"])
            payload = load_checkpoint(state["path"])
            weights = verify_checkpoint_payload(payload, state, canonical_state_dict_sha256)
            model, train, validation, _ = frozen.build_arm_inputs(d["training_contract"], frame, metadata)
            model.load_state_dict(weights, strict=True)
            model.to(c["device"]).eval()
            engine.configure_parameters(model, state["objective"])
            require(canonical_state_dict_sha256(model.state_dict()) == state["state_sha256"], "Loaded model differs")
            jobs = [("validation", validation)]
            if state["purpose"] == "time":
                indices = np.sort(np.random.default_rng(42).choice(len(train.dataset), 2048, replace=False))
                sampled = DataLoader(Subset(train.dataset, indices.tolist()), batch_size=128,
                                     shuffle=False, num_workers=0, collate_fn=train.collate_fn,
                                     generator=torch.Generator().manual_seed(42))
                jobs.append(("train_sample", sampled))
            for split, loader in jobs:
                rows = collect_rows(model, loader, engine, timing, state["purpose"],
                                    state["objective"], c["device"], limit)
                if state["purpose"] == "quantity":
                    ds = loader.dataset
                    ids = np.array([str(ds.parts[p]) for p, _ in ds.index])
                    positions = np.array([i + 1 for _, i in ds.index], dtype=np.int64)
                    seqs = np.array([int(ds.seq_lists[p][i + 1]) for p, i in ds.index], dtype=np.int64)
                    truth = np.array([ds.val_lists[p][i + 1] for p, i in ds.index], dtype=np.float32)
                    require(np.array_equal(truth, rows["true_qty"]), "Prediction target order differs")
                    rows.update(series_ids=ids, target_position=positions, target_seq=seqs)
                    paired[state["objective"]] = rows
                    expected[state["objective"]] = state["expected"]
                else:
                    report = timing.summarize_time_rows(rows)
                    if split == "validation":
                        report["replay"] = verify_time_replay(rows, state["expected"])
                    if split == "train_sample":
                        rows["canonical_target_index"] = indices
                        report["canonical_index_sha256"] = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
                    save_json(output / f"{d['dataset_id']}_{state['id']}_{split}.json", report)
                bounded_artifact(output / f"{d['dataset_id']}_{state['id']}_{split}.npz",
                                 c["max_output_bytes"], lambda handle: np.savez_compressed(handle, **rows))
                limit()
            require(canonical_state_dict_sha256(model.state_dict()) == state["state_sha256"] and
                    all(p.grad is None for p in model.parameters()), "Evaluation mutated model/gradients")
            require(digest(state["path"]) == file_hash, "Checkpoint file changed during evaluation")
            results.append({"dataset_id": d["dataset_id"], "state_id": state["id"],
                            "checkpoint_file_sha256": file_hash, "state_sha256": state["state_sha256"]})
            del model, payload, weights, train, validation
        j, q = paired["joint"], paired["quantity_only"]
        for key in ("series_ids", "target_position", "target_seq", "true_qty", "history_length"):
            require(np.array_equal(j[key], q[key]), f"Paired row identity mismatch: {key}")
        replay = quantity.verify_quantity_replay(quantity_metrics(j), quantity_metrics(q),
                                                 expected["joint"], expected["quantity_only"])
        analysis = quantity.analyze_quantity_pair(d["dataset_id"], j["true_qty"], j["pred_qty"], q["pred_qty"],
                    j["log_qty_loss"], q["log_qty_loss"], j["history_length"], j["series_ids"],
                    d["quantity_boundaries"], d["history_boundaries"])
        analysis["replay"] = replay
        save_json(output / f"{d['dataset_id']}_quantity.json", analysis)
    frozen.validate_import_origins()
    for name, sha in c["source"]["files"].items():
        require(digest(source / name) == sha, f"Source changed during evaluation: {name}")
    validate_contract(c)
    limit()
    write_budgeted_json(output / "complete.json", {"status": "complete", "contract_sha256": json_digest(c),
                   "optimizer_updates": 0, "held_out_evaluated": False,
                   "elapsed_seconds": time.monotonic() - started, "states": results},
                   c["max_output_bytes"], terminal=True)
    lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-contract", type=Path, help="Local retrieved evidence directory; JSON only")
    parser.add_argument("--write-contract", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(not (args.worker and args.execute), "Worker and supervisor modes are exclusive")
    if args.prepare_contract:
        require(not args.contract and not args.execute and not args.worker, "Preparation cannot execute")
        c = prepare_contract(args.prepare_contract)
        if args.write_contract:
            write_new_json(args.write_contract, c)
        print(json.dumps({"status": "prepared_pending_authorization", "states": 10,
                          "max_wall_seconds": c["max_wall_seconds"], "contract_sha256": json_digest(c)}))
        return
    require(args.contract is not None and not args.write_contract, "--contract is required")
    c = read_json(args.contract)
    validate_contract(c, execute=args.execute or args.worker)
    if args.worker:
        try:
            execute(c)
        except Exception as error:
            record_failure(c, os.getpid(), error)
            raise
    elif args.execute:
        supervise(c, args.contract)
    else:
        print(json.dumps({"status": "metadata_valid", "authorization": c["authorization"],
                          "model_loaded": False, "data_read": False, "gpu_initialized": False}))


if __name__ == "__main__":
    main()
