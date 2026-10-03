#!/usr/bin/env python3
"""Validate or run the explicitly approved four-case quantity comparison.

Validation is metadata-only. Execution first qualifies synthetic CUDA replay,
then trains the frozen twelve arms under one process-group deadline. No retry,
automatic research resume, held-out evaluation, or source synchronization.
"""
from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "quantity_comparison_execution_v1"
CASES = ("B_log_original", "raw_original", "log_softplus", "raw_softplus")
DATASETS = ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
PACKAGES = {"paper", "models", "simple_lab_test", "data_loader", "utils"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_hashes(root=ROOT):
    """Resolve the first-party Python import closure without importing models."""
    root = Path(root).resolve()
    pending = [root / "paper/scripts/run_quantity_comparison.py"]
    visited = set()

    def add(module):
        if module.split(".")[0] not in PACKAGES:
            return
        stem = root.joinpath(*module.split("."))
        for path in (stem.with_suffix(".py"), stem / "__init__.py"):
            if path.is_file():
                pending.append(path)
        parent = stem.parent
        while parent != root:
            if (parent / "__init__.py").is_file():
                pending.append(parent / "__init__.py")
            parent = parent.parent

    while pending:
        path = pending.pop().resolve()
        require(path.is_relative_to(root), "Source dependency escapes snapshot")
        if path in visited:
            continue
        visited.add(path)
        package = list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join(prefix + ([node.module] if node.module else []))
                add(module)
                for alias in node.names:
                    if alias.name != "*":
                        add(f"{module}.{alias.name}")
    return {str(p.relative_to(root)): sha_file(p) for p in sorted(visited)}


def validate_contract(contract, *, check_source=True):
    require(contract["schema"] == SCHEMA, "Execution schema mismatch")
    require(contract["status"] == "frozen_pending_explicit_approval", "Unexpected execution status")
    require(contract["cases"] == list(CASES), "Four fixed cases required")
    require([d["dataset_id"] for d in contract["datasets"]] == list(DATASETS), "Dataset order mismatch")
    require(contract["seed"] == 42 and contract["epochs"] == 120, "Fixed seed/epoch budget mismatch")
    limits = contract["limits"]
    require(limits["max_wall_seconds"] == 172800, "48-hour total ceiling required")
    require(limits["qualification_seconds"] == 900, "Qualification ceiling mismatch")
    require(limits["max_concurrent_gpu_jobs"] == 1, "Single GPU job required")
    require(limits["output_bytes"] == 2 * 1024**3 and limits["minimum_free_bytes"] == 5 * 1024**3,
            "Storage ceiling mismatch")
    require(contract["policy"] == {"automatic_retry": False, "automatic_resume": False,
            "held_out": False, "budget_extension": False}, "Forbidden execution policy")
    steps = 0
    for data in contract["datasets"]:
        require(data["loader"]["batch_size"] == 128, "Batch budget mismatch")
        stats = data["statistics"]
        for key in ("train_log_mean", "train_log_std", "raw_scale"):
            require(math.isfinite(stats[key]) and stats[key] > 0, "Nonfinite train statistics")
        require(stats["raw_scale"] >= 1, "Raw scale must be at least one")
        n = data["inherited_data_identity"]["populations"]["train"]["target_count"]
        expected = math.ceil(n / 128) * 120
        require(data["expected_global_steps"] == expected, "Global step budget mismatch")
        steps += expected * 4
    require(steps == contract["total_optimizer_steps"] == 9088320, "Suite step budget mismatch")
    require(sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"], "Source manifest hash mismatch")
    if check_source:
        require(source_hashes() == contract["source"]["files"], "Source closure or contents changed")
        for directory in contract["source"].get("required_directories", []):
            require(directory in {"models", "utils", "sample_data"}, "Invalid snapshot directory")
            require((ROOT / directory).is_dir(), "Snapshot root sentinel missing")
    return {"status": "validated_metadata_only", "contract_sha256": sha_json(contract),
            "arms": 12, "total_optimizer_steps": steps, "gpu_started": False}


def validate_approval(contract, approval):
    require(approval.get("status") == "approved_by_user", "Explicit GPU approval receipt required")
    require(approval.get("execution_contract_sha256") == sha_json(contract), "Approval contract mismatch")
    require(approval.get("scope") == "synthetic_cuda_qualification_and_12_fresh_joint_arms",
            "Approval scope mismatch")
    require(bool(approval.get("user_instruction")), "Approval must record the actual user instruction")


class Budget:
    def __init__(self, contract, started, *, clock=time.time, monotonic=time.monotonic):
        self.contract, self.started, self.clock = contract, started, clock
        self.monotonic = monotonic
        self.monotonic_deadline = monotonic() + max(0, contract["limits"]["max_wall_seconds"] - (clock() - started))
        self.output = Path(contract["execution"]["output_dir"])
        self.last_scan = -math.inf

    def __call__(self, stage=None):
        now = self.clock()
        if now - self.started >= self.contract["limits"]["max_wall_seconds"] or self.monotonic() >= self.monotonic_deadline:
            raise TimeoutError("Suite wall-time ceiling reached")
        if stage == "before_epoch_commit" or now - self.last_scan >= 10:
            self.last_scan = now
            used = 0
            for path in self.output.rglob("*"):
                try:
                    info = path.stat()
                except FileNotFoundError:
                    # An atomic checkpoint rename may remove a temporary file
                    # while the supervising process scans the same directory.
                    continue
                if stat.S_ISREG(info.st_mode):
                    used += info.st_size
            if used >= self.contract["limits"]["output_bytes"] - 64 * 1024**2:
                raise RuntimeError("Artifact storage ceiling reached (64 MiB reserved)")


def configured_runtime(contract, device="cuda:0"):
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    from paper.scripts import run_time_quantity_diagnostic as diagnostic
    require(diagnostic.PROJECT_ROOT.resolve() == ROOT,
            "Import validator loaded from outside the frozen source root")
    diagnostic.validate_import_origins()
    configure_runtime(device, threads=4)
    actual = runtime_identity(device)
    if device == "cuda:0":
        require(actual == contract["runtime_expected"], "5090 runtime changed; do not silently replace the contract")
    return actual


def synthetic_inputs():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.run_time_quantity_diagnostic import reset_seed
    reset_seed(42)
    model, _ = build_count_aware_model("titantpp", hidden_dim=8, train_log_mean=1.0,
                                      train_log_std=1.0, max_seq_len=8, time_intercept_limit=300.0)
    dts = torch.tensor([[0., 0., 1., 1., 2., 1., 1., 2.]]).repeat(8, 1)
    dts[:, -1] += torch.arange(8) % 3
    mask = dts != 0
    quantities = (torch.arange(8, dtype=torch.float32).repeat(8, 1) + torch.arange(8)[:, None] / 8) * mask
    dataset = TensorDataset(dts, mask, quantities)
    train = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(42))
    validation = DataLoader(dataset, batch_size=4, shuffle=False, generator=torch.Generator().manual_seed(42))
    return model, train, validation


def probe_worker(contract, case, stage, started, device):
    from paper.scripts.quantity_comparison_engine import run_case
    from paper.scripts.quantity_objective_comparison import QuantityCase, QuantityStatistics
    runtime = configured_runtime(contract, device)
    model, train, validation = synthetic_inputs()
    identity = {"execution_contract_sha256": sha_json(contract), "runtime": runtime,
                "purpose": "synthetic_qualification", "source": contract["source"]["files_sha256"],
                "data": {"kind": "synthetic", "train_targets": 8, "validation_targets": 8}}
    output = Path(contract["execution"]["output_dir"]) / "qualification" / case / ("full" if stage == "full" else "split")
    return run_case(model=model, train_loader=train, validation_loader=validation,
                    case=QuantityCase(case), statistics=QuantityStatistics(1.0, 4.0),
                    output_dir=output, epochs=2, seed=42, identity=identity, device=device,
                    resume=stage == "resume", stop_after_epochs=1 if stage == "first" else None,
                    budget_check=Budget(contract, started),
                    cuda_qualification={"purpose": "synthetic_probe", "execution_contract_sha256": sha_json(contract), "runtime": runtime})


def compare_replay(full, split):
    for key in ("last_state_sha256", "global_step", "initial_state_sha256", "selectors", "history"):
        require(full[key] == split[key], f"Continuous/resumed replay mismatch: {key}")


def bind_frozen_statistics(data, metadata):
    """Verify recalculation, then use identical stored scalars in every arm."""
    bound = dict(metadata)
    for key in ("train_log_mean", "train_log_std", "raw_scale"):
        frozen = data["statistics"][key]
        require(math.isclose(metadata[key], frozen, rel_tol=0, abs_tol=1e-12), f"Train statistic drift: {key}")
        bound[key] = frozen
    require(metadata["all_train_rows"]["quantity_sha256"] == data["statistics"]["all_train_quantity_sha256"],
            "All-train row quantity identity drift")
    return bound


def audit_pairs(summaries):
    require(len(summaries) == 4, "Four completed cases required")
    baseline = summaries[0]
    for candidate in summaries:
        require(candidate["status"] == "complete", "Incomplete arm")
        require(candidate["initial_state_sha256"] == baseline["initial_state_sha256"], "Initial state differs")
        require(candidate["global_step"] == baseline["global_step"], "Training step mismatch")
        require(len(candidate["history"]) == len(baseline["history"]), "History length mismatch")
        for left, right in zip(baseline["history"], candidate["history"], strict=True):
            for key in ("epoch", "global_step", "train_count", "train_batches", "validation_count",
                        "validation_batches", "train_batch_order_sha256"):
                require(left[key] == right[key], f"Paired exposure mismatch: {key}")
    return {"status": "passed", "arms": 4, "epochs": len(baseline["history"]),
            "global_steps_per_case": baseline["global_step"]}


def production_shape_checks(contract, device, budget):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.quantity_objective_comparison import QuantityCase, QuantityStatistics, joint_causal_batch_objective
    from paper.scripts.run_time_quantity_diagnostic import reset_seed
    from paper.scripts.time_quantity_diagnostic import task_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, capture_rng_state, restore_rng_state
    for data in contract["datasets"]:
        stats = QuantityStatistics(data["statistics"]["train_log_mean"], data["statistics"]["raw_scale"])
        initial_hash = None
        for case in QuantityCase:
            budget()
            reset_seed(42)
            model, _ = build_count_aware_model(**data["model"], train_log_mean=stats.mu,
                         train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])
            model.to(device)
            fingerprint = canonical_state_dict_sha256(model.state_dict())
            state_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
            # Includes last + optimizer + both selectors + atomic temporary
            # copies, with a generous allowance for histories and metadata.
            require(12 * (32 * state_bytes + 4 * 1024**2) < contract["limits"]["output_bytes"] // 2,
                    "Projected checkpoint storage exceeds frozen budget")
            require(10 * state_bytes + 4 * 1024**2 < contract["limits"]["per_file_bytes"],
                    "Projected epoch checkpoint exceeds individual file budget")
            initial_hash = initial_hash or fingerprint
            require(fingerprint == initial_hash, "Production-shape initial state differs")
            dts = torch.ones((128, data["loader"]["max_seq_len"]), device=device)
            mask = torch.ones_like(dts, dtype=torch.bool)
            mask[::2, :4] = False
            quantities = torch.full_like(dts, stats.q0)
            rng = capture_rng_state()
            result = joint_causal_batch_objective(model, dts, mask, quantities, statistics=stats, case=case)
            if case is QuantityCase.B_LOG_ORIGINAL:
                restore_rng_state(rng)
                baseline = task_outputs(model, dts, mask, quantities, "joint")
                require(torch.equal(baseline["objective_loss"], result["objective_loss"]), "B production objective differs")
            result["objective_loss"].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01).step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Production-shape nonfinite parameters")
            model.eval()
            with torch.no_grad():
                before = joint_causal_batch_objective(model, dts, mask, quantities, statistics=stats, case=case)["pred_qty"]
                dts[~mask], quantities[~mask] = 999., 999.
                dts[:, -1], quantities[:, -1] = 31., 42.
                after = joint_causal_batch_objective(model, dts, mask, quantities, statistics=stats, case=case)["pred_qty"]
                require(torch.equal(before, after), "Production-shape target/padding causality failed")
            del model, result
            gc.collect()


def qualify(contract, contract_path, started, device="cuda:0"):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    output = Path(contract["execution"]["output_dir"])
    runtime = configured_runtime(contract, device)
    began = time.monotonic()
    budget = Budget(contract, started)
    summaries = []
    for case in CASES:
        for stage in ("full", "first", "resume"):
            budget()
            remaining = contract["limits"]["qualification_seconds"] - (time.monotonic() - began)
            require(remaining > 0, "Qualification time ceiling reached")
            command = [sys.executable, str(Path(__file__).resolve()), "_probe", "--contract", str(contract_path),
                       "--case", case, "--stage", stage, "--started", str(started), "--device", device]
            subprocess.run(command, check=True, timeout=remaining)
        folder = output / "qualification" / case
        full, split = read_json(folder / "full/summary.json"), read_json(folder / "split/summary.json")
        compare_replay(full, split)
        # These are newly generated synthetic checkpoints, never research models.
        a = torch_load_checkpoint(folder / "full/last_epoch_state.pt", map_location="cpu")
        b = torch_load_checkpoint(folder / "split/last_epoch_state.pt", map_location="cpu")
        for key in ("optimizer_state_sha256", "rng_state_sha256", "model_state_sha256"):
            require(a[key] == b[key], f"Synthetic replay checkpoint mismatch: {key}")
        summaries.append(full)
    audit_pairs(summaries)
    remaining = contract["limits"]["qualification_seconds"] - (time.monotonic() - began)
    require(remaining > 0, "Qualification time ceiling reached")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "_shapes", "--contract", str(contract_path),
                    "--started", str(started), "--device", device], check=True, timeout=remaining)
    require(time.monotonic() - began < contract["limits"]["qualification_seconds"], "Qualification time ceiling reached")
    require(configured_runtime(contract, device) == runtime, "Runtime drift after qualification")
    validate_contract(contract)
    receipt = {"passed": True, "qualifies_cuda": device == "cuda:0", "runtime": runtime,
               "execution_contract_sha256": sha_json(contract), "elapsed_seconds": time.monotonic() - began,
               "checks": ["fresh_process_exact_resume", "optimizer_and_rng_identity", "paired_batches",
                          "production_shapes", "B_objective_identity", "target_padding_causality"]}
    write_json(output / "qualification/receipt.json", receipt)
    return receipt


def run_suite(contract, contract_path, started):
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.quantity_comparison_engine import run_case
    from paper.scripts.quantity_objective_comparison import QuantityCase, QuantityStatistics
    from paper.scripts.run_time_quantity_diagnostic import build_arm_inputs
    output = Path(contract["execution"]["output_dir"])
    budget = Budget(contract, started)
    receipt = qualify(contract, contract_path, started)
    require(receipt["qualifies_cuda"], "CPU qualification cannot authorize CUDA training")
    completed = []
    for data in contract["datasets"]:
        budget()
        frame, metadata = prepare_quantity_comparison_data(data)
        metadata = bind_frozen_statistics(data, metadata)
        results = []
        for case in CASES:
            budget()
            require(configured_runtime(contract) == receipt["runtime"], "Runtime drift before arm")
            write_json(output / "status.json", {"status": "running", "dataset_id": data["dataset_id"],
                       "case": case, "completed_arms": len(completed), "started_at_unix": started})
            model, train, validation, _ = build_arm_inputs({**data, "seed": 42}, frame, metadata)
            identity = {"execution_contract_sha256": sha_json(contract), "runtime": receipt["runtime"],
                        "source": contract["source"], "data": data["inherited_data_identity"],
                        "dataset_id": data["dataset_id"]}
            summary = run_case(model=model, train_loader=train, validation_loader=validation,
                      case=QuantityCase(case), statistics=QuantityStatistics(metadata["train_log_mean"], metadata["raw_scale"]),
                      output_dir=output / data["dataset_id"] / case, epochs=120, seed=42, identity=identity,
                      device="cuda:0", lr=.001, weight_decay=.01, grad_clip=1., budget_check=budget,
                      cuda_qualification=receipt)
            require(summary["global_step"] == data["expected_global_steps"], "Arm budget not completed")
            results.append(summary)
            completed.append({"dataset_id": data["dataset_id"], "case": case})
            del model, train, validation
            gc.collect()
        audit = audit_pairs(results)
        write_json(output / data["dataset_id"] / "paired_comparison.json", audit)
        del frame
        gc.collect()
    write_json(output / "status.json", {"status": "complete", "completed_arms": len(completed),
               "arms": completed, "optimizer_steps": contract["total_optimizer_steps"],
               "held_out_evaluated": False, "started_at_unix": started, "elapsed_seconds": time.time() - started})


def stop_owned_process_group(process):
    """Clean up our descendants even if their group leader exited first."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute(contract, contract_path, approval):
    validate_contract(contract)
    validate_approval(contract, approval)
    require(str(Path(sys.executable).resolve()) == contract["execution"]["python"], "Pinned Python executable required")
    require(str(ROOT) == contract["execution"]["source_dir"], "Run from the frozen 5090 source snapshot")
    output = Path(contract["execution"]["output_dir"])
    require(not output.exists(), "Fresh suite directory required; automatic resume is forbidden")
    require(shutil.disk_usage(output.parent).free >= contract["limits"]["minimum_free_bytes"], "Insufficient free disk")
    # A held GPU lock serializes this experiment; foreign GPU users are rejected.
    import fcntl
    with (output.parent / "quantity_comparison_gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = subprocess.check_output(["nvidia-smi", "--id=" + contract["runtime_expected"]["gpu"]["uuid"],
                    "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        require(not running, "GPU already has a compute process")
        output.mkdir()
        started = time.time()
        write_json(output / "execution_receipt.json", {"contract_sha256": sha_json(contract), "approval": approval,
                   "started_at_unix": started, "deadline_unix": started + contract["limits"]["max_wall_seconds"]})
        environment = os.environ.copy()
        environment.update({k: v for k, v in contract["runtime_expected"]["environment"].items() if v is not None})
        for key, value in contract["runtime_expected"]["environment"].items():
            if value is None:
                environment.pop(key, None)
        process = None
        handlers = {}

        def interrupted(signum, frame):
            raise InterruptedError(f"Launcher received signal {signum}")

        try:
            for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                handlers[signum] = signal.signal(signum, interrupted)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_suite", "--contract", str(contract_path),
                                       "--started", str(started)], env=environment, start_new_session=True)
            began = time.monotonic()
            budget = Budget(contract, started)
            while True:
                budget()
                qualified = (output / "qualification/receipt.json").is_file()
                ceiling = contract["limits"]["max_wall_seconds"] if qualified else contract["limits"]["qualification_seconds"]
                remaining = ceiling - (time.monotonic() - began)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, ceiling)
                try:
                    code = process.wait(timeout=min(5, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
            require(code == 0, f"Suite failed with exit code {code}; no retry")
        except BaseException as error:
            stop_owned_process_group(process)
            previous = read_json(output / "status.json") if (output / "status.json").exists() else {}
            write_json(output / "status.json", {**previous, "status": "timeout" if isinstance(error, (subprocess.TimeoutExpired, TimeoutError)) else "failed",
                       "error": str(error), "started_at_unix": started, "automatic_retry": False})
            raise
        finally:
            for signum, previous_handler in handlers.items():
                signal.signal(signum, previous_handler)


def freeze_contract(design_path, statistics_path, destination, *, run_name="quantity_comparison_seed42_v1"):
    """Freeze already verified train metadata and an immutable local source copy."""
    require(re.fullmatch(r"quantity_comparison_seed42_v[1-9][0-9]*", run_name) is not None,
            "Run name must be a versioned quantity comparison identifier")
    design, statistics = read_json(design_path), read_json(statistics_path)
    require(statistics["comparison_contract"]["sha256"] == sha_file(design_path), "Statistics design identity mismatch")
    require(not Path(destination).exists(), "Refusing to replace an existing execution contract")
    dataset_metadata = {d["dataset_id"]: d for d in statistics["datasets"]}
    old = read_json(design["basis"]["completed_diagnosis_contract"]["path"])
    frozen_B = {d["dataset_id"]: d for d in old["datasets"]}
    datasets = []
    for d in design["datasets"]:
        observed = dataset_metadata[d["dataset_id"]]
        all_rows, targets = observed["all_train_rows"], observed["canonical_train_targets"]
        require(not observed["held_out_materialized"], "Held-out rows prohibited")
        require(math.isclose(all_rows["log1p_mean"], d["mu_all_train_rows"], rel_tol=0, abs_tol=1e-12), "Frozen B initialization drift")
        frozen_std = frozen_B[d["dataset_id"]]["train_log_std"]
        require(math.isclose(all_rows["log1p_std"], frozen_std, rel_tol=0, abs_tol=1e-12), "Frozen B standard deviation drift")
        dataset = {k: v for k, v in d.items() if k not in {"proposed_steps", "raw_scale_status", "raw_scale_canonical_train_targets"}}
        datasets.append({**dataset, "raw_scale_canonical_train_targets": targets["raw_scale"],
                         "raw_scale_status": "verified_train_only_frozen",
                         "statistics": {"train_log_mean": d["mu_all_train_rows"],
                         "train_log_std": frozen_std, "raw_scale": targets["raw_scale"],
                         "recomputation_absolute_tolerance": 1e-12,
                         "initialization_policy": "use original frozen B mu/std after verifying recalculation; no refitted replacement",
                         "all_train_quantity_sha256": all_rows["quantity_sha256"]},
                         "expected_global_steps": d["proposed_steps"]["global_steps_per_case_at_120_epochs"]})
    runtime = old["datasets"][0]["training_contract"]["runtime"]
    readiness_path = Path(statistics_path).resolve().parent / "server_readiness.json"
    readiness = read_json(readiness_path)
    require(all(runtime[key] == value for key, value in readiness["versions"].items()), "Read-only server versions differ")
    require(runtime["gpu"]["uuid"] == readiness["gpu"]["uuid"] and runtime["gpu"]["driver_version"] == readiness["gpu"]["driver_version"],
            "Read-only GPU identity differs")
    files = source_hashes()
    source_digest = sha_json(files)
    preparation = Path(statistics_path).resolve().parent
    snapshot_parent = preparation / "snapshots" / source_digest
    snapshot = snapshot_parent / "source"
    snapshot.mkdir(parents=True, exist_ok=False)
    # Existing repository-root discovery requires all three sentinels. The
    # sample_data directory is deliberately empty: data remain at pinned paths.
    directories = ["models", "utils", "sample_data"]
    for directory in directories:
        (snapshot / directory).mkdir(exist_ok=True)
    for name, digest in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        require(sha_file(target) == digest, "Snapshot copy mismatch")
    archive = snapshot_parent / "source.tar.gz"
    with archive.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for directory in directories:
                info = tarfile.TarInfo("source/" + directory)
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                bundle.addfile(info)
            for name in sorted(files):
                path = snapshot / name
                info = tarfile.TarInfo("source/" + name)
                info.size, info.mode = path.stat().st_size, 0o644
                with path.open("rb") as handle:
                    bundle.addfile(info, handle)
    remote = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + run_name
    contract = {"schema": SCHEMA, "status": "frozen_pending_explicit_approval",
        "design": {"path": str(Path(design_path).resolve()), "sha256": sha_file(design_path)},
        "statistics_receipt": {"path": str(Path(statistics_path).resolve()), "sha256": sha_file(statistics_path)},
        "source": {"files": files, "files_sha256": source_digest, "local_snapshot": str(snapshot),
                   "required_directories": directories,
                   "archive": str(archive), "archive_sha256": sha_file(archive),
                   "git_head_provenance": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "identity_authority": "content-addressed source closure including uncommitted implementation"},
        "runtime_expected": runtime,
        "runtime_evidence_status": "versions/GPU/driver verified by read-only SSH; execution must reverify full numerical runtime and pass new CUDA qualification",
        "server_readiness": {"path": str(readiness_path), "sha256": sha_file(readiness_path)},
        "execution": {"server_alias": "5090", "python": "/opt/miniconda3/envs/ai_env/bin/python3.12",
                      "source_dir": remote + "/source", "output_dir": remote + "/run",
                      "tmux_session": run_name,
                      "argv": ["/opt/miniconda3/envs/ai_env/bin/python3.12", remote + "/source/paper/scripts/run_quantity_comparison.py",
                               "execute", "--contract", remote + "/execution_contract.json", "--approval", remote + "/approval.json"]},
        "cases": list(CASES), "datasets": datasets, "epochs": 120, "seed": 42,
        "total_optimizer_steps": 9088320,
        "limits": {"max_wall_seconds": 172800, "qualification_seconds": 900,
                   "max_concurrent_gpu_jobs": 1, "output_bytes": 2 * 1024**3,
                   "minimum_free_bytes": 5 * 1024**3, "cpu_threads": 4,
                   "per_file_bytes": 64 * 1024**2, "termination_grace_seconds": 15},
        "policy": {"automatic_retry": False, "automatic_resume": False, "held_out": False, "budget_extension": False},
        "approval": {"required_scope": "synthetic_cuda_qualification_and_12_fresh_joint_arms",
                     "receipt_path": remote + "/approval.json", "approved": False},
        "budget_rationale": {"historical_J_mtime_window_seconds": [7784.747, 883.779, 17067.414],
             "fourfold_reference_hours": 28.59548888888889,
             "not_a_measured_ETA": True, "new_ceiling_hours": 48,
             "prior_72_hour_authorization_inherited": False},
        "result_scope": "separate quantity/time selectors and matched final epoch; no held-out or benchmark claim",
        "qualification_state": "not_run_on_cuda; no new GPU execution authorized"}
    validate_contract(contract)
    write_json(destination, contract)
    return {"contract": str(destination), "contract_sha256": sha_json(contract),
            "source_files": len(files), "snapshot": str(snapshot), "gpu_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "validate", "execute", "_suite", "_probe", "_shapes"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--started", type=float)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--stage", choices=("full", "first", "resume"))
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--design", type=Path, default=ROOT / "paper/contracts/quantity_objective_output_comparison_v1.json")
    parser.add_argument("--statistics", type=Path, default=ROOT / "search_artifacts/quantity_comparison_preparation_v1/train_statistics.json")
    parser.add_argument("--run-name", default="quantity_comparison_seed42_v1",
                        help="Fresh versioned run identifier for freeze only")
    args = parser.parse_args(argv)
    contract_path = args.contract.resolve()
    if args.mode == "freeze":
        print(json.dumps(freeze_contract(args.design, args.statistics, contract_path, run_name=args.run_name)))
        return 0
    contract = read_json(contract_path)
    if args.mode == "validate":
        print(json.dumps(validate_contract(contract)))
    elif args.mode == "execute":
        require(args.device == "cuda:0", "Execution requires explicit CUDA; CPU fallback is forbidden")
        require(args.approval is not None, "Explicit approval receipt is required")
        execute(contract, contract_path, read_json(args.approval))
    else:
        require(args.started is not None, "Worker requires inherited suite deadline")
        require(args.mode != "_suite" or args.device == "cuda:0", "Research suite requires explicit CUDA")
        validate_contract(contract)
        # GPU workers only run inside a previously approved suite directory.
        if args.device == "cuda:0":
            receipt = read_json(Path(contract["execution"]["output_dir"]) / "execution_receipt.json")
            validate_approval(contract, receipt["approval"])
            require(receipt["started_at_unix"] == args.started, "Worker deadline changed")
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE, (contract["limits"]["per_file_bytes"], contract["limits"]["per_file_bytes"]))
        if args.mode == "_suite":
            run_suite(contract, contract_path, args.started)
        elif args.mode == "_shapes":
            configured_runtime(contract, args.device)
            production_shape_checks(contract, args.device, Budget(contract, args.started))
        else:
            require(args.case is not None and args.stage is not None, "Probe case and stage required")
            probe_worker(contract, args.case, args.stage, args.started, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
