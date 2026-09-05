#!/usr/bin/env python3
"""Run exactly one authorized prior-prefix phase in a fresh source snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONTRACT_ID = "titans_mac_prior_prefix_v1"
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket")
BASELINE = "titantpp_titans_mac"
CANDIDATE = "titantpp_titans_mac_prior_prefix"
BACKBONES = (BASELINE, CANDIDATE)
VARIANT = "count_only_log_regression"
TEST = "simple_lab_test/search/tests/test_titans_mac_prior_prefix_contract.py"
CUDA_VALIDATOR = "paper/scripts/validate_titans_mac_prior_prefix_cuda.py"
COMPARATOR = "paper/scripts/compare_titans_mac_prior_prefix.py"
TRAINER = "paper/scripts/run_count_aware_tpp_backbone_control.py"
POLICY = "paper/scripts/run_with_titantpp_mac_dynamo_policy.py"
DIAGNOSTIC = "paper/scripts/diagnose_titantpp_mac_nonfinite.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite artifact number")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def validate_contract(contract):
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong prior-prefix contract")
    require(contract.get("backbone") == CANDIDATE and contract.get("baseline") == BASELINE,
            "Model comparison identity drift")
    require(tuple(contract.get("dataset_order", [])) == DATASETS, "Dataset order drift")
    training = contract.get("training", {})
    for key, value in {"quantity_variant": VARIANT, "lambda_log_qty": 1., "lambda_tail": 0.,
            "time_head": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10 / 3,
            "time_intercept_limit": 300., "effective_duration_cap": 10., "optimizer": "AdamW",
            "lr": .001, "batch_size": 128, "outer_gradient_clip": 1., "inner_gradient_clip": 1.,
            "max_epochs": 300, "min_epochs": 40, "patience": 40, "execution_backend": "optimized",
            "fresh": True, "resume": False, "held_out_test_evaluated": False}.items():
        require(training.get(key) == value, f"Frozen training contract changed: {key}")
    require(set(contract.get("datasets", {})) == set(DATASETS), "Three-dataset scope changed")
    for dataset, (lookback, length) in zip(DATASETS, ((168, 256), (520, 256), (52, 64))):
        row = contract["datasets"][dataset]
        require((row["lookback"], row["max_seq_len"]) == (lookback, length), "Dataset context drift")
        for name in ("data_path", "split_manifest_path"):
            path = Path(row[name])
            require(not path.is_absolute() and ".." not in path.parts, "Unsafe dataset path")
        for name in ("data_sha256", "split_manifest_sha256"):
            require(bool(re.fullmatch(r"[0-9a-f]{64}", row[name])), "Invalid input SHA256")
        require(row["train_targets"] > 0 and row["validation_targets"] > 0, "Missing coverage counts")


def verify_source(manifest_path, revision, contract_path, *, root=ROOT):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "Full source revision required")
    manifest = read(manifest_path)
    require(manifest.get("source_revision") == revision, "Source revision mismatch")
    require(manifest.get("contract_id") == CONTRACT_ID, "Wrong source package")
    require(manifest.get("origin") == "git_archive_committed_tree", "Source must be a committed archive")
    require(manifest.get("root_scripts_excluded") is True and bool(manifest.get("files")), "Invalid source package")
    require(manifest.get("scaffold_directories") == ["sample_data"], "Missing data-free root scaffold")
    scaffold = root / "sample_data"
    require(scaffold.is_dir() and not scaffold.is_symlink() and not any(scaffold.iterdir()),
            "Source sample_data scaffold must remain empty; use --data-project")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and not name.startswith("scripts/"), "Unsafe manifest path")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"Source file missing or linked: {name}")
        require(digest(path) == expected, f"Source file changed: {name}")
    required = (TEST, CUDA_VALIDATOR, COMPARATOR, TRAINER, POLICY, DIAGNOSTIC,
                "models/TPPs/CountAwareFactory.py", "models/Titan/common/titans_mac.py",
                "paper/scripts/run_titans_mac_prior_prefix_5090.py",
                "paper/contracts/titans_mac_prior_prefix_v1.json",
                "paper/results/titans_mac_prior_prefix_20260905/frozen_references.json")
    require(set(required) <= manifest["files"].keys(), "Source manifest lacks required execution files")
    packaged_contract = root / "paper/contracts/titans_mac_prior_prefix_v1.json"
    require(Path(contract_path).resolve() == packaged_contract.resolve(), "Contract must be inside this source snapshot")
    references = read(contract_path)["frozen_references"]
    require(references["path"] in manifest["files"]
            and manifest["files"][references["path"]] == references["sha256"], "Frozen references are missing or changed")
    return digest(manifest_path)


def verify_data(project, contract):
    for dataset in DATASETS:
        row = contract["datasets"][dataset]
        for path_key, hash_key in (("data_path", "data_sha256"),
                                  ("split_manifest_path", "split_manifest_sha256")):
            require(digest(project / row[path_key]) == row[hash_key], f"Input checksum changed: {dataset}/{path_key}")


def check_gpu(gpu_csv, compute_csv, gdm, names):
    rows = [line for line in gpu_csv.splitlines() if line.strip()]
    require(len(rows) == 1, "Expected exactly one GPU")
    name, free = [part.strip() for part in rows[0].split(",")]
    require("RTX 5090" in name and int(free) >= 12000, "Wrong GPU or insufficient free VRAM")
    require(not compute_csv.strip() and gdm.strip() == "inactive", "GPU busy or GDM active")
    require(not ({"gnome-shell", "Xwayland"} & set(names.split())), "Desktop GPU process present")


def gpu_preflight():
    def output(cmd, allowed=(0,)):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        require(result.returncode in allowed, f"GPU preflight command failed: {cmd}")
        return result.stdout
    gpu = output(["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"])
    compute = output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    gdm = output(["systemctl", "is-active", "gdm"], (0, 3))
    check_gpu(gpu, compute, gdm, output(["ps", "-eo", "comm="]))
    return {"gpu": gpu.strip(), "compute": compute.strip(), "gdm": gdm.strip()}


def training_command(data_project, output_root, contract, revision, phase, dataset, backbone, seed, *, python=sys.executable):
    require(phase in ("e1", "screening", "confirm"), "No training in CUDA-only phase")
    require(dataset in DATASETS and backbone in BACKBONES, "Unknown experiment row")
    require(seed in ((52, 62) if phase == "confirm" else (42,)), "Seed is outside this phase")
    row = contract["datasets"][dataset]
    output = Path(output_root) / dataset / backbone / f"seed_{seed}"
    epochs, minimum, patience = (1, 1, 1) if phase == "e1" else (300, 40, 40)
    role = "experimental" if backbone == BASELINE else "titans_mac_prior_prefix_screening"
    cmd = [python, "-s", str(ROOT / DIAGNOSTIC), "--snapshot-dir", str(output.parent / f"diagnostic_seed_{seed}"),
           str(ROOT / POLICY), "--recompile-limit", "64", "--accumulated-recompile-limit", "512",
           str(ROOT / TRAINER), "--data", str(Path(data_project) / row["data_path"]),
           "--split-manifest", str(Path(data_project) / row["split_manifest_path"]),
           "--output-dir", str(output), "--source-revision", revision,
           "--execution-role", f"prior_prefix_{phase}_5090", "--dataset-contract", dataset,
           "--model-role", role, "--device", "cuda", "--epochs", str(epochs),
           "--min-epochs", str(minimum), "--early-stopping-patience", str(patience),
           "--batch-size", "128", "--lr", "0.001", "--hidden-dim", "64",
           "--lookback-weeks", str(row["lookback"]), "--max-seq-len", str(row["max_seq_len"]),
           "--quantity-variants", "log_mse", "--lambda-log-qty", "1", "--lambda-tail", "0",
           "--backbones", backbone, "--seeds", str(seed), "--time-head-mode", "legacy_clamped_rmtpp",
           "--time-scale", "3", "--time-w-max", str(10 / 3), "--time-intercept-limit", "300",
           "--grad-clip", "1", "--titans-memory-gradient-clip", "1",
           "--titans-mac-execution-backend", "optimized", "--allow-partial-contract"]
    return output, cmd


def run_plan(args, contract):
    if args.phase == "cuda":
        return []
    seeds = (52, 62) if args.phase == "confirm" else (42,)
    return [(dataset, backbone, seed, *training_command(args.data_project, args.output_root,
                contract, args.source_revision, args.phase, dataset, backbone, seed))
            for dataset in DATASETS for seed in seeds for backbone in BACKBONES]


def audit_xml(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    require(bool(cases), "CUDA test suite executed no tests")
    require(not any(list(root.iter(tag)) for tag in ("failure", "error", "skipped")), "CUDA tests failed or skipped")
    require(all("test_titans_mac_prior_prefix_contract" in case.get("classname", "") for case in cases),
            "Unexpected CUDA test module")
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    require(sum(int(s.get("tests", 0)) for s in suites) == len(cases), "JUnit count mismatch")
    require(all(int(s.get(key, 0)) == 0 for s in suites for key in ("failures", "errors", "skipped")),
            "JUnit reports failed or skipped tests")
    return {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0, "xml_sha256": digest(path)}


def verify_proof(path, phase, revision, contract_hash, manifest_hash):
    require(path is not None, f"Explicit {phase} PASS proof is required")
    proof = read(path)
    require(proof.get("status") == "PASS" and proof.get("phase") == phase, f"{phase} proof did not pass")
    require(proof.get("source_revision") == revision and proof.get("contract_sha256") == contract_hash
            and proof.get("source_manifest_sha256") == manifest_hash, "Gate proof uses a different source or contract")
    require(proof.get("held_out_test_evaluated") is False, "Gate proof evaluated held-out test")
    run_root = Path(proof.get("run_root", ""))
    require(run_root.is_absolute() and run_root.resolve() == Path(path).resolve().parent,
            "Gate proof run_root must identify its own artifact directory")
    for name, expected in proof.get("evidence_files", {}).items():
        target = Path(name)
        if not target.is_absolute():
            target = Path(path).parent / target
        require(target.resolve().is_relative_to(run_root.resolve()), "Gate evidence is outside its run_root")
        require(digest(target) == expected, f"Gate evidence changed: {target}")
    require(bool(proof.get("evidence_files")), "Gate proof lacks evidence hashes")
    if phase == "cuda":
        audit = proof.get("audit", {})
        require(audit.get("status") == "PASS" and audit.get("device") == "cuda"
                and audit.get("cuda_available") is True and audit.get("cost_gate_pass") is True,
                "CUDA audit or cost gate failed")
        tests = proof.get("tests", {})
        require(tests.get("tests", 0) > 0 and all(tests.get(k) == 0 for k in ("failures", "errors", "skipped")),
                "Missing mandatory CUDA test evidence")
    elif phase == "e1":
        expected = {(d, b, 42) for d in DATASETS for b in BACKBONES}
        rows = proof.get("runs", [])
        require(len(rows) == len(expected) and {(r["dataset"], r["backbone"], r["seed"]) for r in rows} == expected
                and all(r.get("status") == "PASS" for r in rows), "Full paired three-dataset e1 audit missing")
    elif phase == "screening":
        require(proof.get("audit", {}).get("status") == "PASS"
                and proof.get("audit", {}).get("screening_pass") is True, "Seed42 screening gate failed")
        expected = {(d, b, 42) for d in DATASETS for b in BACKBONES}
        rows = proof.get("runs", [])
        require(len(rows) == len(expected)
                and {(r["dataset"], r["backbone"], r["seed"]) for r in rows} == expected
                and all(r.get("status") == "PASS" for r in rows), "Seed42 paired source evidence missing")
    return proof


def audit_run(output, row, dataset, backbone, seed, phase, revision, data_project):
    """Check full coverage, frozen identity, checkpoint/optimizer and replay evidence."""
    import torch
    import polars as pl
    from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
    from models.Titan.common.titans_mac_optimized import apply_titantpp_mac_semantic_optimization
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    for name in ("models.TPPs.CountAwareFactory", "models.Titan.common.titans_mac_optimized",
                 "paper.scripts.count_aware_tpp_backbone.core", "paper.scripts.count_aware_tpp_backbone.training"):
        require(Path(sys.modules[name].__file__).resolve().is_relative_to(ROOT),
                f"Imported source is outside the committed snapshot: {name}")

    launch = read(output / "launch_contract.json")
    leaf = output / "runs" / backbone / VARIANT / f"seed_{seed}"
    summary, history = read(leaf / "summary.json"), read(leaf / "history.json")["history"]
    epochs, minimum, patience = (1, 1, 1) if phase == "e1" else (300, 40, 40)
    expected = {"status": "complete", "completed_run_count": 1, "dataset": dataset,
                "backbones": [backbone], "seeds": [seed], "epochs": epochs, "batch_size": 128,
                "lr": .001, "hidden_dim": 64, "grad_clip": 1., "titans_memory_gradient_clip": 1.,
                "lambda_log_qty": 1., "lambda_tail": 0., "partial_smoke": False,
                "source_revision": revision, "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
                "lookback_weeks": row["lookback"], "max_seq_len": row["max_seq_len"],
                "data_sha256": row["data_sha256"], "split_manifest_sha256": row["split_manifest_sha256"]}
    for key, value in expected.items():
        require(launch.get(key) == value, f"Launch drift: {key}")
    require(set(launch["split_rows"]) == {"train", "validation"}, "Held-out frame materialized")
    for key, value in {"mode": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10 / 3,
                       "time_intercept_limit": 300., "time_head_lr_multiplier": 1.}.items():
        require(launch["time_head"].get(key) == value, f"Shared time head drift: {key}")
    require(launch["early_stopping"]["min_epochs"] == minimum
            and launch["early_stopping"]["patience"] == patience, "Checkpoint policy drift")
    require(summary.get("status") == "success" and summary.get("source_revision") == revision
            and summary.get("source_revision_history") == [revision]
            and summary.get("held_out_test_evaluated") is False, "Invalid or resumed summary")
    require(minimum <= len(history) <= epochs and summary["completed_epochs"] == len(history), "Epoch coverage incomplete")
    require([h["epoch"] for h in history] == list(range(1, len(history) + 1)), "Nonsequential epochs")
    require(all(h["train_event_count"] == row["train_targets"]
                and h["train_batch_count"] == math.ceil(row["train_targets"] / 128)
                and h["train_all_finite"] is True for h in history), "Full train coverage missing")
    require(all(sum(r["count"] for r in summary[key]) == row["validation_targets"]
                for key in ("quantity_rows", "history_rows")), "Full validation coverage missing")
    best = min(history, key=lambda h: h["val_joint_objective"])
    require(summary["best_epoch"] == best["epoch"] and math.isclose(summary["best_val_joint_objective"],
            best["val_joint_objective"], rel_tol=1e-7, abs_tol=1e-8), "Best checkpoint rule drift")
    require(len(history) == epochs or len(history) - best["epoch"] >= patience, "Premature training stop")
    finite({"launch": launch, "summary": summary, "history": history})
    checkpoint = torch.load(leaf / "best_val_joint_objective_model.pt", map_location="cpu", weights_only=False)
    last = torch.load(leaf / "last_epoch_state.pt", map_location="cpu", weights_only=False)
    def finite_tensors(value):
        if torch.is_tensor(value):
            require(bool(torch.isfinite(value).all()), "Non-finite checkpoint or optimizer tensor")
        elif isinstance(value, dict):
            for item in value.values():
                finite_tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite_tensors(item)
    for payload in (checkpoint, last):
        require(payload["source_revision"] == revision and payload["source_revision_history"] == [revision],
                "Checkpoint revision mismatch")
        validate_checkpoint_route(payload, backbone)
        finite_tensors(payload)
    state_hash = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    require(state_hash == summary["checkpoint_state_sha256"] == canonical_state_dict_sha256(last["best_state_dict"]),
            "Selected checkpoint state mismatch")
    if phase == "e1":
        require(state_hash == canonical_state_dict_sha256(last["model_state_dict"]), "e1 last/best state differs")
    optimizer = last["optimizer_state_dict"]
    require(len(optimizer["param_groups"]) == 1 and bool(optimizer["state"]), "Optimizer state is missing")
    steps = sum(h["train_batch_count"] for h in history)
    require(all(0 < float(s["step"]) <= steps for s in optimizer["state"].values()),
            "Optimizer steps are outside the fresh full-train history")
    meta = checkpoint["interface_meta"]
    model, encoder = build_count_aware_model(backbone, hidden_dim=64, max_seq_len=row["max_seq_len"],
        train_log_mean=meta["train_target_mean"], train_log_std=meta["train_target_std"],
        quantity_variant=VARIANT, time_head_mode="legacy_clamped_rmtpp", time_scale=3., time_w_max=10 / 3,
        time_intercept_limit=300., titans_memory_gradient_clip=1.)
    apply_titantpp_mac_semantic_optimization(model)
    require(checkpoint["encoder_config"].get("titans_memory_gradient_clip") == 1., "Missing inner clipping policy")
    require(checkpoint["encoder_config"].get("mac_execution_backend") == "optimized",
            "Checkpoint used a different MAC execution backend")
    model.load_state_dict(last["model_state_dict"], strict=True)
    restored_optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
    restored_optimizer.load_state_dict(optimizer)
    group = restored_optimizer.param_groups[0]
    require(group["lr"] == .001 and group["weight_decay"] == .01
            and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8,
            "Restored AdamW configuration drift")
    parameter_states = {name: restored_optimizer.state[param] for name, param in model.named_parameters()
                        if param in restored_optimizer.state}
    require(float(parameter_states["quantity_head.weight"]["step"]) == steps,
            "Quantity-head optimizer steps do not cover every train batch")
    writer_activity = {}
    for name, state in parameter_states.items():
        if ".neural_memory." in name and any(token in name for token in (
                "key_projection", "value_projection", "update_rate_projection",
                "momentum_projection", "forgetting_projection")):
            writer_activity[name] = {"step": int(float(state["step"])),
                                     "exp_avg_nonzero": bool(torch.count_nonzero(state["exp_avg"]))}
    if backbone == CANDIDATE:
        for token in ("key_projection.weight", "value_projection.weight"):
            require(any(name.endswith(token) and value["exp_avg_nonzero"]
                        for name, value in writer_activity.items()), f"Candidate writer did not learn: {token}")
    # Load two actual validation examples with their observed history. The lazy
    # scan filters held-out rows before collection, including series selection.
    available = pl.scan_parquet(Path(data_project) / row["data_path"]).filter(
        pl.col("chronological_split").is_in(["train", "validation"]))
    series = (available.filter(pl.col("chronological_split") == "validation")
              .select("oper_part_no").unique().sort("oper_part_no").limit(1).collect().item())
    frame = prepare_count_frame(available.filter(pl.col("oper_part_no") == series)
                                .collect().sort(["oper_part_no", "seq"]))
    loader = make_loader(frame, target_split="validation", batch_size=2,
                         lookback_weeks=row["lookback"], max_seq_len=row["max_seq_len"],
                         shuffle=False, generator=None)
    _, dt, mask, _, qty = next(iter(loader))
    predictions = []
    for state in (checkpoint["model_state_dict"], last["best_state_dict"]):
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            predictions.append(target_outputs(model, dt, mask, qty, lambda_log_qty=1.))
    for key in predictions[0]:
        require(torch.equal(predictions[0][key], predictions[1][key]), f"Checkpoint replay differs: {key}")
        finite_tensors(predictions[0][key])
    require(str(summary.get("training_device", "")).startswith("cuda"), "Training was not on CUDA")
    allocated, reserved = (summary.get(k, 0) for k in ("cuda_peak_memory_allocated_bytes", "cuda_peak_memory_reserved_bytes"))
    require(0 < allocated <= reserved, "CUDA peak memory telemetry missing")
    evidence = {str(p.resolve()): digest(p) for p in (output / "launch_contract.json", leaf / "summary.json",
                leaf / "history.json", leaf / "best_val_joint_objective_model.pt", leaf / "last_epoch_state.pt")}
    return {"status": "PASS", "dataset": dataset, "backbone": backbone, "seed": seed,
            "train_targets": row["train_targets"], "validation_targets": row["validation_targets"],
            "completed_epochs": len(history), "checkpoint_state_sha256": state_hash,
            "checkpoint_prediction_replay_exact": True, "optimizer_steps": steps,
            "checkpoint_replay_source": "first_sorted_validation_series_observed_history",
            "checkpoint_replay_rows": int(dt.shape[0]), "optimizer_restore_pass": True,
            "writer_optimizer_activity": writer_activity,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "quantity_contract": launch["quantity_contract"], "history_length_contract": launch["history_length_contract"],
            "cuda_peak_memory_allocated_bytes": allocated, "cuda_peak_memory_reserved_bytes": reserved,
            "evidence_files": evidence, "held_out_test_evaluated": False}


def execute(args):
    contract = read(args.contract)
    validate_contract(contract)
    plan = run_plan(args, contract)
    if args.dry_run:
        return {"phase": args.phase, "training_started": False,
                "run_count": len(plan), "commands": [r[-1] for r in plan]}
    require(Path(sys.executable).resolve() == Path("/opt/miniconda3/envs/ai_env/bin/python").resolve(),
            "Use the existing approved 5090 Python environment")
    require(platform.node() == "RTX5090-server", "This launcher is restricted to the configured 5090 host")
    import torch
    import polars
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "cuda": torch.version.cuda, "polars": polars.__version__}
    require(all(runtime[k] == contract["runtime"]["expected_" + k] for k in runtime), "Existing CUDA runtime changed")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(args.output_root.resolve() != args.data_project.resolve(), "Outputs must not overwrite the data project")
    manifest_hash = verify_source(args.source_manifest, args.source_revision, args.contract)
    contract_hash = digest(args.contract)
    verify_data(args.data_project, contract)
    if args.phase != "cuda":
        verify_proof(args.cuda_proof, "cuda", args.source_revision, contract_hash, manifest_hash)
    if args.phase in ("screening", "confirm"):
        verify_proof(args.e1_proof, "e1", args.source_revision, contract_hash, manifest_hash)
    if args.phase == "confirm":
        screening_proof = verify_proof(args.screening_proof, "screening", args.source_revision,
                                       contract_hash, manifest_hash)
    require(not args.output_root.exists(), "Fresh output directory required; resume/retry/overwrite are disabled")
    gpu = gpu_preflight()
    args.output_root.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "phase": args.phase, "source_revision": args.source_revision,
             "source_manifest_sha256": manifest_hash, "contract_sha256": contract_hash,
             "held_out_test_evaluated": False, "gpu_preflight": gpu, "runs": [], "evidence_files": {},
             "runtime": runtime, "run_root": str(args.output_root.resolve()),
             "started_at": datetime.now(timezone.utc).isoformat()}
    status_path = args.output_root / "status.json"
    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        save(status_path, state)
        print(json.dumps({k: state[k] for k in ("status", "phase", "updated_at")}), flush=True)
    def stop(signum, frame):
        raise InterruptedError(f"Launcher received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = None
    def run(cmd, logfile, extra_env=None):
        nonlocal child
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "42",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(ROOT), **(extra_env or {})}
        with logfile.open("x") as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            update(child_pid=child.pid, command=cmd, log_path=str(logfile))
            code = child.wait()
            child = None
            require(code == 0, f"Stage exited {code}; evidence retained: {logfile}")
    try:
        with (ROOT / ".titans_mac_prior_prefix_5090.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            update()
            if args.phase == "cuda":
                xml = args.output_root / "cuda_contract_tests.xml"
                run([sys.executable, "-s", "-m", "pytest", str(ROOT / TEST), "-q", f"--junitxml={xml}"],
                    args.output_root / "cuda_contract_tests.log", {"REQUIRE_CUDA": "1"})
                state["tests"] = audit_xml(xml)
                audit_path = args.output_root / "cuda_audit.json"
                run([sys.executable, "-s", str(ROOT / CUDA_VALIDATOR), "--output", str(audit_path),
                     "--source-revision", args.source_revision, "--contract", str(args.contract)],
                    args.output_root / "cuda_audit.log")
                audit = read(audit_path)
                require(audit.get("status") == "PASS" and audit.get("source_revision") == args.source_revision
                        and audit.get("contract_sha256") == contract_hash and audit.get("device") == "cuda"
                        and audit.get("cuda_available") is True and audit.get("cost_gate_pass") is True
                        and audit.get("held_out_test_evaluated") is False, "CUDA validator or cost gate did not pass")
                state["audit"] = audit
                state["evidence_files"] = {str(p.resolve()): digest(p) for p in (xml, audit_path)}
            else:
                for dataset, backbone, seed, output, cmd in plan:
                    verify_source(args.source_manifest, args.source_revision, args.contract)
                    verify_data(args.data_project, contract)
                    gpu_preflight()
                    update(current_dataset=dataset, current_backbone=backbone, current_seed=seed)
                    run(cmd, args.output_root / f"{dataset}_{backbone}_{seed}.log")
                    result = audit_run(output, contract["datasets"][dataset], dataset, backbone, seed,
                                       args.phase, args.source_revision, args.data_project)
                    state["runs"].append(result)
                    state["evidence_files"].update(result["evidence_files"])
                    update()
                for dataset in DATASETS:
                    rows = [r for r in state["runs"] if r["dataset"] == dataset]
                    require(len({r["parameter_count"] for r in rows}) == 1, "Paired parameter counts differ")
                    for field in ("quantity_contract", "history_length_contract"):
                        require(all(r[field] == rows[0][field] for r in rows), f"Paired {field} differs")
                if args.phase != "e1":
                    audit_path = args.output_root / "comparison.json"
                    command = [sys.executable, "-s", str(ROOT / COMPARATOR), "--contract", str(args.contract),
                               "--run-root", str(args.output_root), "--phase", args.phase,
                               "--output", str(audit_path)]
                    if args.phase == "confirm":
                        # Recheck the immutable screening evidence immediately
                        # before pooling seed42 with the two confirmatory seeds.
                        verify_proof(args.screening_proof, "screening", args.source_revision,
                                     contract_hash, manifest_hash)
                        command.extend(["--screening-run-root", screening_proof["run_root"]])
                    run(command, args.output_root / "comparison.log")
                    state["audit"] = read(audit_path)
                    state["evidence_files"][str(audit_path.resolve())] = digest(audit_path)
                    require(state["audit"].get("status") == "PASS", "Performance comparison gate failed")
                    state["performance_acceptance"] = state["audit"].get(
                        "screening_pass" if args.phase == "screening" else "confirmation_pass")
                    require(isinstance(state["performance_acceptance"], bool), "Comparator lacks a performance decision")
                else:
                    state["performance_acceptance"] = "not_evaluated_e1_only"
            verify_source(args.source_manifest, args.source_revision, args.contract)
            verify_data(args.data_project, contract)
            update(status="PASS", child_pid=None, completed_run_count=len(state["runs"]))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        update(status="FAIL", error=repr(exc), child_pid=None)
        raise
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cuda", "e1", "screening", "confirm"), required=True)
    parser.add_argument("--contract", type=Path, default=ROOT / "paper/contracts/titans_mac_prior_prefix_v1.json")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "source_manifest.json")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--data-project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-proof", type=Path)
    parser.add_argument("--e1-proof", type=Path)
    parser.add_argument("--screening-proof", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(execute(parse_args()), indent=2))
