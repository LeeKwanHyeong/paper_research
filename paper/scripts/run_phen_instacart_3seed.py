"""Fresh PHEN-only Instacart fits; immutable, Validation-only execution.

This adapter reuses the frozen full-PHEN constructor and joint-training hooks.
It does not relax or enter the historical 36-condition A--D controller.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import gzip
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from paper.scripts import run_titantpp_mlp_cap36 as common

SCHEMA = "phen_instacart_3seed_v1"
SEEDS = (42, 52, 62)
HOST_BY_SEED = {"42": "5080", "52": "5090", "62": "5090"}
MONITOR = "validation_raw_quantity_rmse"
VARIANT = "count_only_log_regression"
require = common.require
read = common.read
sha = common.sha_file
canonical = common.sha_json
write = common.write_exclusive


def now():
    return datetime.now(timezone.utc).isoformat()


def below(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            "Expected a safe relative path")
    result = Path(root) / path
    require(not any(p.is_symlink() for p in (result, *result.parents)), "Symlink not admitted")
    return common.resolve_under(root, result)


def validate(contract):
    require(contract.get("schema") == SCHEMA, "Wrong experiment schema")
    require(contract.get("seeds") == list(SEEDS)
            and contract.get("host_by_seed") == HOST_BY_SEED, "Only the approved three seeds are admitted")
    require(contract.get("model_role") == "observed_time_joint_v1"
            and contract.get("evaluation_scope") == "validation_only", "Wrong execution scope")
    require(set(contract.get("hosts", {})) == {"5080", "5090"}, "Wrong hosts")
    datasets = contract.get("datasets", [])
    require(len(datasets) == 1 and datasets[0].get("dataset_id") == "insta_market_basket",
            "Only Instacart is admitted")
    source = contract["source"]
    require(bool(source.get("files")) and source.get("files_sha256") == canonical(source["files"]),
            "Source inventory missing or inconsistent")
    require(isinstance(source.get("base_git_revision"), str) and len(source["base_git_revision"]) == 40,
            "Source revision missing")
    require(bool(contract.get("input_files")), "Input inventory missing")
    for mapping in (source["files"], contract["input_files"]):
        for path, digest in mapping.items():
            require(not Path(path).is_absolute() and ".." not in Path(path).parts, "Unsafe inventory path")
            require(isinstance(digest, str) and len(digest) == 64
                    and set(digest) <= set("0123456789abcdef"), "Invalid file digest")
    for key, expected in {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
                          "batch_size": 128, "monitor": MONITOR}.items():
        require(contract["training"].get(key) == expected, "Training condition changed: " + key)
    require(not contract["training"].get("warm_start", False), "Warm start is forbidden")
    data = datasets[0]
    require(data["loader"]["batch_size"] == 128 and data["loader"]["max_seq_len"] == 64
            and data["loader"]["lookback_weeks"] == 52, "Instacart loader changed")
    require(data["model"]["hidden_dim"] == 64 and data["model"]["quantity_variant"] == VARIANT
            and data["model"]["lambda_log_qty"] == 1. and data["model"]["lambda_tail"] == 0.
            and data["model"]["time_head_mode"] == "heteroscedastic_lognormal_duration",
            "PHEN common heads/loss changed")
    require(set(data["inherited_data_identity"]["populations"]) == {"train", "validation"},
            "Test population is not admitted")
    for key in ("data", "split_manifest"):
        entry = data["inherited_data_identity"][key]
        require(contract["input_files"].get(entry["path"]) == entry["sha256"],
                "Input record is not sealed: " + key)
    for host, spec in contract["hosts"].items():
        require(Path(spec["root"]).is_absolute() and Path(spec["python"]).is_absolute(), "Absolute host paths required")
        require(spec.get("boot_id") and spec.get("gpu_uuid"), "Native host identity required")
        require(isinstance(spec.get("paused_processes"), list), "Paused process inventory required")
        if host == "5090":
            require(not spec["paused_processes"], "5090 must have no retained GPU predecessor")
    return data


def verify_files(contract, host):
    data = deepcopy(validate(contract))
    root = Path(contract["hosts"][host]["root"]).resolve()
    source = root / "source"
    require(Path(__file__).resolve().is_relative_to(source.resolve()), "Runner is not executing from its sealed source")
    for path, digest in {**contract["source"]["files"], **contract["input_files"]}.items():
        require(sha(below(source, path)) == digest, "Frozen source/input bytes changed: " + path)
    for key in ("data", "split_manifest"):
        data["inherited_data_identity"][key]["path"] = str(below(source, data["inherited_data_identity"][key]["path"]))
    return data, root


def process_identity(pid):
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "state": stat[0], "pgid": int(stat[2]), "start_ticks": int(stat[19]),
            "argv": (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
            "cwd": str((proc / "cwd").resolve(strict=True))}


def gpu_guard(contract, host):
    spec = contract["hosts"][host]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    require(boot == spec["boot_id"], "Host rebooted since admission")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], timeout=15, text=True).splitlines()
    require(gpu == [spec["gpu_uuid"]], "GPU identity changed")
    retained = {}
    for expected in spec["paused_processes"]:
        actual = process_identity(expected["pid"])
        require(actual["state"] == "T", "Protected process is not stopped")
        require(all(actual[key] == expected[key] for key in ("pid", "pgid", "start_ticks", "argv", "cwd")),
                "Protected process identity changed")
        retained[actual["pid"]] = actual
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                     timeout=15, text=True)
    active = {int(line.strip()) for line in output.splitlines() if line.strip()}
    require(active <= set(retained), "Unexpected GPU process; refusing overlap")
    return {"observed_at": now(), "boot_id": boot, "gpu_uuid": gpu[0],
            "retained_paused_processes": list(retained.values()), "gpu_processes": sorted(active)}


def runtime(device):
    import torch
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    configure_runtime(device, 4)
    require(device == "cpu" or (device == "cuda:0" and torch.cuda.is_available()), "Unsupported device")
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None}


def qualify(contract, host, mode, device):
    import torch
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts.count_aware_tpp_backbone import core
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    data, root = verify_files(contract, host)
    folder = root / "qualification"
    folder.mkdir(exist_ok=True)
    write(folder / (mode + "_attempt.json"), {"contract_sha256": canonical(contract), "started_at": now()})
    guard = gpu_guard(contract, host) if device.startswith("cuda") else None
    native = runtime(device)
    torch.manual_seed(42)
    model, encoder = shared._build_model(data, "full")
    model = model.to(device).train()
    tracker = shared.raw9.JointTimeHeadTracker(model)
    optimizer = tracker.bind_optimizer(torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01))
    shape = (128, 64) if mode == "capacity" else (3, 8)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    dt = torch.ones(shape, device=device)
    quantities = torch.arange(shape[1], device=device).float().add(1).expand(shape).clone()
    mask = torch.ones(shape, device=device, dtype=torch.bool)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = core.target_outputs(model, dt, mask, quantities, lambda_log_qty=1.)
        require(torch.equal(result["joint_loss"], result["time_loss"] + result["log_qty_loss"]), "Joint loss changed")
        tracker.joint_observations += 1
        loss = result["joint_loss"].mean()
        require(bool(torch.isfinite(loss)), "Nonfinite synthetic loss")
        loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    proof = tracker.proof()
    model.eval()
    state = canonical_state_dict_sha256(model.state_dict())
    with torch.no_grad():
        original = core.target_outputs(model, dt, mask, quantities, lambda_log_qty=1.)
        for changed_dt, changed_q in ((dt, quantities.clone()), (dt.clone(), quantities)):
            if changed_q is not quantities:
                changed_q[:, -1] += 100
            else:
                changed_dt[:, -1] += 1
            altered = core.target_outputs(model, changed_dt, mask, changed_q, lambda_log_qty=1.)
            torch.testing.assert_close(original["pred_qty"], altered["pred_qty"], rtol=1e-5, atol=1e-5)
        binary = io.BytesIO()
        torch.save(model.state_dict(), binary)
        binary.seek(0)
        restored, restored_encoder = shared._build_model(data, "full")
        require(encoder == restored_encoder, "Encoder identity changed")
        restored.load_state_dict(torch.load(binary, map_location="cpu", weights_only=True), strict=True)
        restored = restored.to(device).eval()
        replay = core.target_outputs(restored, dt, mask, quantities, lambda_log_qty=1.)
        for key in ("pred_qty", "time_loss"):
            torch.testing.assert_close(original[key], replay[key], rtol=1e-5, atol=1e-5)
    require(state == canonical_state_dict_sha256(model.state_dict()), "Evaluation changed state")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    result = {"status": "passed", "mode": mode, "host": host, "contract_sha256": canonical(contract),
              "source_files_sha256": contract["source"]["files_sha256"], "runtime": native, "gpu_guard": guard,
              "shape": list(shape), "real_data_loaded": False, "held_out_test_evaluated": False,
              "time_head_training_verification": proof, "parameter_count": sum(p.numel() for p in model.parameters()),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.startswith("cuda") else None,
              "checks": ["finite_forward_backward", "target_quantity_hidden", "target_time_hidden",
                         "checkpoint_roundtrip", "evaluation_state_unchanged", "joint_time_gradient"], "completed_at": now()}
    optimizer.step = tracker.original_step
    write(folder / (mode + ".json"), result)
    return result


def verify_qualification(contract, host, root):
    receipts = {}
    for mode, shape in (("synthetic", [3, 8]), ("capacity", [128, 64])):
        receipt = read(root / "qualification" / (mode + ".json"))
        require(receipt["status"] == "passed" and receipt["host"] == host
                and receipt["contract_sha256"] == canonical(contract)
                and receipt["source_files_sha256"] == contract["source"]["files_sha256"]
                and receipt["shape"] == shape and receipt["runtime"]["device"] == "cuda:0",
                "Missing native qualification: " + mode)
        receipts[mode] = receipt
    require(receipts["synthetic"]["runtime"] == receipts["capacity"]["runtime"], "Qualification runtime drift")
    return receipts


def load_selected(contract, data, seed, folder, summary, fit_proof, device):
    from paper.scripts import run_shared_component_ablation as shared
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    checkpoint = common.resolve_under(folder, summary["checkpoint_path"])
    payload = torch_load_checkpoint(checkpoint, map_location="cpu")
    require(payload.get("backbone") == "full" and payload.get("seed") == seed
            and payload.get("variant") == VARIANT and payload.get("checkpoint_monitor") == MONITOR
            and payload.get("best_epoch") == summary["best_epoch"]
            and payload.get("evaluation_scope") == "validation_only"
            and payload.get("held_out_test_evaluated") is False, "Selected checkpoint provenance mismatch")
    digest = canonical_state_dict_sha256(payload["model_state_dict"])
    require(digest == payload["model_state_sha256"] == summary["checkpoint_state_sha256"], "Selected tensor digest mismatch")
    require(payload["interface_meta"]["source_files_sha256"] == contract["source"]["files_sha256"], "Checkpoint source changed")
    history = read(checkpoint.parent / "history.json")["history"]
    require(shared.raw9.prepared.select_best_epoch(history) == summary["best_epoch"], "Not the first strict Validation minimum")
    shared.raw9.verify_time_head_proof(payload["time_head_training_verification"], payload["model_state_dict"])
    shared.raw9.verify_checkpoint_fit_proof(payload["time_head_training_verification"], fit_proof)
    model, encoder = shared._build_model(data, "full")
    require(payload["encoder_config"] == encoder, "Checkpoint encoder changed")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), checkpoint, digest


def evaluate_validation(data, frame, model, output, device):
    import numpy as np
    import torch
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    before = canonical_state_dict_sha256(model.state_dict())
    loader = make_loader(frame, target_split="validation", shuffle=False, generator=None,
                         **{k: data["loader"][k] for k in ("batch_size", "lookback_weeks", "max_seq_len")})
    truth, identity = common.population_identity(loader.dataset, "validation")
    common.verify_population(identity, data["inherited_data_identity"]["populations"]["validation"])
    accumulator, processed = shared.raw9.MetricAccumulator(data["quantity_boundaries_all_train_rows"][-1]), 0
    with Path(output).open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        with torch.no_grad():
            for _, dts, mask, part_ids, quantities in loader:
                value = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
                arrays = {k: value[k].cpu().numpy() for k in ("true_qty", "pred_qty", "time_loss", "history_length")}
                size = len(arrays["true_qty"])
                require(np.array_equal(arrays["true_qty"].astype(np.float64), truth[processed:processed + size]), "Validation target changed")
                accumulator.add(arrays["true_qty"], arrays["pred_qty"], arrays["time_loss"])
                for i in range(size):
                    part, end = loader.dataset.index[processed + i]
                    require(int(part_ids[i]) == int(part), "Validation order changed")
                    row = {"target_index": processed + i, "entity_id": str(loader.dataset.parts[part]),
                           "target_seq": int(loader.dataset.seq_lists[part][end + 1]), "split": "validation",
                           "raw_quantity": float(arrays["true_qty"][i]), "predicted_raw_quantity": float(arrays["pred_qty"][i]),
                           "time_nll": float(arrays["time_loss"][i]), "history_length": int(arrays["history_length"][i])}
                    zipped.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
                processed += size
    require(processed == identity["target_count"], "Incomplete Validation")
    require(before == canonical_state_dict_sha256(model.state_dict()), "Validation changed model state")
    return {"metrics": accumulator.finish(), "target_identity": identity, "prediction_rows": processed,
            "prediction_sha256": sha(output), "state_sha256": before, "split": "validation"}


def verify_selected_metrics(evaluation, summary):
    for metric, key in (("qty_rmse", "best_val_qty_rmse"), ("qty_mae", "best_val_qty_mae"),
                        ("time_nll", "best_val_time_nll")):
        require(math.isclose(evaluation["metrics"]["overall"][metric], summary[key], rel_tol=1e-5, abs_tol=1e-5),
                "Selected Validation metric did not reproduce training: " + metric)


def train(contract, host, seed, device):
    import torch
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.count_aware_tpp_backbone import training
    require(device == "cuda:0", "Official fits require the admitted CUDA device")
    require(seed in SEEDS and HOST_BY_SEED[str(seed)] == host, "Wrong seed placement")
    data, root = verify_files(contract, host)
    qualification = verify_qualification(contract, host, root)
    guard = gpu_guard(contract, host)
    folder = root / "jobs" / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(folder / "attempt.json", {"seed": seed, "host": host, "started_at": now(),
                                   "contract_sha256": canonical(contract), "gpu_guard": guard})
    native = runtime(device)
    require(native == qualification["synthetic"]["runtime"], "Training runtime differs from qualification")
    frame, input_receipt = base.prepare_admitted_data(data)
    require(set(frame["chronological_split"].unique().to_list()) == {"train", "validation"}, "Test materialized")
    write(folder / "input_receipt.json", input_receipt)
    interface = shared._training_interface(contract, data, frame, "full")
    args = shared._training_args(contract, data, folder, "full")
    args.device, args.execution_role = device, "phen_instacart_3seed_validation"
    print(json.dumps({"event": "training_entered", "seed": seed, "host": host, "time": now(),
                      "training": contract["training"], "loader": data["loader"], "runtime": native}), flush=True)
    with shared._engine_hooks(data, "full") as tracked:
        summary, _, _ = training.train_one(args=args, frame=frame,
            quantity_contract={"boundaries": data["quantity_boundaries_all_train_rows"], "strata": []},
            interface_meta=interface, backbone="full", quantity_variant=VARIANT, seed=seed)
        proof = tracked["tracker"].proof()
    fit_seconds = time.monotonic() - started
    model, checkpoint, digest = load_selected(contract, data, seed, folder, summary, proof, device)
    selected_bytes = sha(checkpoint)
    reference = evaluate_validation(data, frame, model, folder / "validation_predictions.jsonl.gz", device)
    verify_selected_metrics(reference, summary)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, _, replay_digest = load_selected(contract, data, seed, folder, summary, proof, device)
    replay = evaluate_validation(data, frame, model, folder / "validation_replay.jsonl.gz", device)
    count = common.compare_prediction_files(folder / "validation_replay.jsonl.gz", folder / "validation_predictions.jsonl.gz")
    require(digest == replay_digest and sha(checkpoint) == selected_bytes, "Checkpoint changed during replay")
    # Re-read every sealed byte after completion, before claiming success.
    verify_files(contract, host)
    receipt = {"schema": SCHEMA, "status": "complete", "host": host, "seed": seed,
               "contract_sha256": canonical(contract), "source_files_sha256": contract["source"]["files_sha256"],
               "selected_checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": selected_bytes,
               "state_sha256": digest, "selected_epoch": summary["best_epoch"],
               "completed_epochs": summary["completed_epochs"], "summary": summary,
               "time_head_training_verification": proof, "validation": reference, "validation_replay": replay,
               "individual_predictions_match": True, "replayed_rows": count,
               "fit_with_preparation_seconds": fit_seconds, "total_seconds": time.monotonic() - started,
               "runtime": native, "held_out_test_evaluated": False, "completed_at": now()}
    receipt["files"] = common.file_inventory(root, [p for p in folder.rglob("*") if p.is_file()])
    write(folder / "receipt.json", receipt)
    return receipt


def dispatch(contract, host, contract_path, contract_sha, device):
    _data, root = verify_files(contract, host)
    verify_qualification(contract, host, root)
    write(root / "dispatch_attempt.json", {"contract_sha256": canonical(contract), "host": host,
                                          "started_at": now(), "pid": os.getpid()})
    exits = []
    for seed in (s for s in SEEDS if HOST_BY_SEED[str(s)] == host):
        verify_files(contract, host)
        guard = gpu_guard(contract, host)
        require(not (root / "jobs" / f"seed{seed}").exists(), "Existing seed attempt; no resume or retry")
        command = [contract["hosts"][host]["python"], "-u", "-m", "paper.scripts.run_phen_instacart_3seed",
                   "--contract", str(Path(contract_path).resolve()), "--contract-sha", contract_sha,
                   "--host", host, "--mode", "train", "--seed", str(seed), "--device", device]
        stdout, stderr = root / f"seed{seed}.stdout.log", root / f"seed{seed}.stderr.log"
        started = time.monotonic()
        with stdout.open("xb") as out, stderr.open("xb") as err:
            child = subprocess.Popen(command, cwd=root / "source", stdout=out, stderr=err,
                                     env={**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONDONTWRITEBYTECODE": "1"})
            write(root / f"seed{seed}.launch.json", {"pid": child.pid, "seed": seed, "host": host,
                  "command": command, "contract_sha256": canonical(contract), "gpu_guard": guard, "started_at": now()})
            code = child.wait()
        actual = {"seed": seed, "host": host, "pid": child.pid, "actual_os_returncode": code,
                  "contract_sha256": canonical(contract), "elapsed_seconds": time.monotonic() - started,
                  "stdout": {"path": stdout.name, "bytes": stdout.stat().st_size, "sha256": sha(stdout)},
                  "stderr": {"path": stderr.name, "bytes": stderr.stat().st_size, "sha256": sha(stderr)}, "observed_at": now()}
        write(root / f"seed{seed}.exit.json", actual)
        exits.append(actual)
        if code != 0:
            write(root / "dispatch_receipt.json", {"status": "failed", "exits": exits, "contract_sha256": canonical(contract)})
            return code
        receipt = read(root / "jobs" / f"seed{seed}" / "receipt.json")
        require(receipt["status"] == "complete" and receipt["individual_predictions_match"]
                and receipt["contract_sha256"] == canonical(contract), "Successful child lacks verified receipt")
        gpu_guard(contract, host)
    write(root / "dispatch_receipt.json", {"status": "complete", "host": host, "exits": exits,
          "contract_sha256": canonical(contract), "supervisor_own_exit_directly_observed": False, "completed_at": now()})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("check", "synthetic", "capacity", "train", "dispatch"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    require(sha(args.contract) == args.contract_sha, "Contract bytes changed")
    contract = read(args.contract)
    if args.mode == "check":
        _data, root = verify_files(contract, args.host)
        result = {"status": "verified", "host": args.host, "contract_sha256": canonical(contract), "root": str(root)}
    elif args.mode in ("synthetic", "capacity"):
        result = qualify(contract, args.host, args.mode, args.device)
    elif args.mode == "train":
        require(args.seed is not None, "Train requires a seed")
        result = train(contract, args.host, args.seed, args.device)
    else:
        return dispatch(contract, args.host, args.contract, args.contract_sha, args.device)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
