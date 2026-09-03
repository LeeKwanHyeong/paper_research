#!/usr/bin/env python3
"""Run the single-change static retrieval candidate, never rerun baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONTRACT = ROOT / "paper/contracts/hard_lmm_weighted_static_v1.json"
VARIANT = "count_only_log_regression"
BACKBONE = "titantpp_weighted_static_memory"
SUMMARY = Path("runs") / BACKBONE / VARIANT / "seed_42/summary.json"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name="root"):
    if isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite metric: {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{name}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            finite(item, f"{name}[{i}]")


def verify_source(revision):
    manifest = read(ROOT / "source_manifest.json")
    require(manifest["source_revision"] == revision and len(revision) == 40, "Source revision mismatch")
    for relative, expected in manifest["files"].items():
        require(digest(ROOT / relative) == expected, f"Source mismatch: {relative}")
    return digest(ROOT / "source_manifest.json")


def preflight(minimum_free):
    def capture(command, codes=(0,)):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        require(result.returncode in codes, f"Preflight unavailable: {command}: {result.stderr}")
        return result.stdout.strip()
    gdm = capture(["systemctl", "is-active", "gdm"], codes=(0, 3))
    require(gdm == "inactive", f"GDM must be inactive, observed {gdm}")
    processes = capture(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    require(not processes, f"GPU already in use: {processes}")
    graphics = capture(["ps", "-eo", "comm="])
    require(not any(line.strip() in {"gnome-shell", "Xwayland"} for line in graphics.splitlines()), "Desktop GPU process present")
    memory = capture(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    require(len(memory.splitlines()) == 1 and int(memory) >= minimum_free, f"Insufficient VRAM: {memory}")
    kernel = capture(["journalctl", "-k", "-b", "--no-pager", "--since", "1 hour ago"])
    errors = [line for line in kernel.splitlines() if any(token in line.lower() for token in ("nvrm: xid", "out of memory", "oom-kill"))]
    require(not errors, f"Recent GPU/kernel error: {errors}")
    return {"free_vram_mib": int(memory), "gdm": gdm, "compute_processes": [], "recent_kernel_errors": []}


def baseline(row, project):
    artifact = project / row["artifact_dir"]
    summary_path = artifact / "runs/titantpp" / VARIANT / "seed_42/summary.json"
    checks = [(project / row["data_path"], row["data_sha256"]),
              (project / row["split_manifest_path"], row["split_manifest_sha256"]),
              (artifact / "launch_contract.json", row["contract_sha256"]),
              (summary_path, row["summary_sha256"]),
              (summary_path.parent / "best_val_joint_objective_model.pt", row["checkpoint_file_sha256"])]
    for path, expected in checks:
        require(digest(path) == expected, f"Baseline identity mismatch: {path}")
    c, s = read(artifact / "launch_contract.json"), read(summary_path)
    for key, expected in {"epochs": 300, "batch_size": 128, "lr": .001, "hidden_dim": 64,
                          "grad_clip": 1., "lambda_log_qty": 1.}.items():
        require(c[key] == expected, f"Baseline contract mismatch: {key}")
    for key, expected in {"min_epochs": 40, "patience": 40, "monitor": "validation_joint_objective"}.items():
        require(c["early_stopping"][key] == expected, f"Baseline selection mismatch: {key}")
    require(s["status"] == "success" and s["seed"] == 42 and s["backbone"] == "titantpp", "Baseline incomplete")
    require(s["checkpoint_state_sha256"] == row["checkpoint_state_sha256"], "Baseline state mismatch")
    require(not s["held_out_test_evaluated"] and s["evaluation_scope"] == "validation_only", "Baseline scope mismatch")
    finite(s)
    return c, s


def command(row, project, output, revision, phase):
    from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS
    tail = DATASET_CONTRACTS[row["contract_dataset"]]["tail_contract"]
    args = {"data": project / row["data_path"], "split-manifest": project / row["split_manifest_path"],
            "output-dir": output, "source-revision": revision, "execution-role": f"weighted_static_{phase}_5080",
            "dataset-contract": row["contract_dataset"], "model-role": "t0_weighted_static_retrieval",
            "device": "cuda", "epochs": 1 if phase == "smoke" else 300,
            "min-epochs": 1 if phase == "smoke" else 40, "early-stopping-patience": 1 if phase == "smoke" else 40,
            "batch-size": 128, "lr": .001, "grad-clip": 1., "hidden-dim": 64,
            "lookback-weeks": row["lookback"], "max-seq-len": row["max_seq_len"],
            "quantity-variants": "log_mse", "backbones": BACKBONE, "seeds": 42,
            "lambda-log-qty": 1., "lambda-tail": 0., "time-head-mode": "legacy_clamped_rmtpp",
            "time-scale": 3., "time-w-max": 10. / 3., "time-intercept-limit": 30.,
            "time-wd-safety-limit": 40., "time-head-lr-multiplier": 1.,
            "tail-threshold": tail["threshold"], "tail-normalization-scale": tail["normalization_scale"],
            "tail-clip-cap": tail["clip_cap"], "tail-huber-delta": tail["huber_delta"]}
    result = [sys.executable, "-s", "-u", str(ROOT / "paper/scripts/run_count_aware_tpp_backbone_control.py")]
    for key, value in args.items():
        result.extend([f"--{key}", str(value)])
    return result + ["--allow-partial-contract"]


def audit_run(output, row, reference, revision, phase, contract):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    c, s = read(output / "launch_contract.json"), read(output / SUMMARY)
    finite(s)
    finite(read((output / SUMMARY).parent / "history.json"))
    require(c["status"] == "complete" and c["completed_run_count"] == 1, "Run incomplete")
    require(c["backbones"] == [BACKBONE] and c["seeds"] == [42] and c["lambda_tail"] == 0., "Wrong candidate")
    require(c["source_revision"] == revision and s["source_revision"] == revision, "Run revision mismatch")
    require(s["source_revision_history"] == [revision], "Unexpected resumed checkpoint")
    require(c["model_role"] == "t0_weighted_static_retrieval" and not c["partial_smoke"], "Contract/subsample mismatch")
    require(c["evaluation_scope"] == s["evaluation_scope"] == "validation_only", "Non-validation scope")
    require(set(c["split_rows"]) == {"train", "validation"}, "Held-out rows materialized")
    require(not c["held_out_test_evaluated"] and not s["held_out_test_evaluated"], "Held-out evaluated")
    require(not list(output.rglob("*test_summary*")) and not list(output.rglob("*test_metrics*")), "Held-out artifacts found")
    require(c["time_head"]["mode"] == "legacy_clamped_rmtpp", "Time head changed")
    # These are launch arguments, not the separate derived train-time statistics.
    for key, expected in {"time_scale": 3., "time_w_max": 10. / 3., "time_intercept_limit": 30.,
                          "time_head_lr_multiplier": 1.}.items():
        require(c["time_head"][key] == expected, f"Time launch contract changed: {key}")
    require(c["early_stopping"]["monitor"] == "validation_joint_objective", "Selector changed")
    require(c["time_head"]["train_time_statistics"]["target_count"] == contract["train_target_counts"][row["dataset"]], "Train targets changed")
    require(sum(q["count"] for q in s["quantity_rows"]) == contract["validation_target_counts"][row["dataset"]], "Validation targets changed")
    require([(q["stratum"], q["count"]) for q in s["quantity_rows"]] ==
            [(q["stratum"], q["count"]) for q in reference["quantity_rows"]], "Quantity strata changed")
    require(s["parameter_count"] == reference["parameter_count"], "Parameter count changed")
    for key in ("train_target_mean", "train_target_std"):
        require(math.isclose(s["interface_meta"][key], reference["interface_meta"][key], rel_tol=0, abs_tol=1e-12),
                f"Train-only quantity statistics changed: {key}")
    require(c["lookback_weeks"] == row["lookback"] and c["max_seq_len"] == row["max_seq_len"], "Context changed")
    meta = s["encoder_config"]
    require(meta["static_retrieval_contract_id"] == "hard_lmm_weighted_static_v1" and
            meta["static_retrieval_temperature"] == 1. and meta["lmm_topk"] == 4 and
            meta["lmm_mem_size"] == 64 and meta["persistent_mem_size"] == 16, "Retrieval contract changed")
    checkpoint_path = (output / SUMMARY).parent / "best_val_joint_objective_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(canonical_state_dict_sha256(checkpoint["model_state_dict"]) == s["checkpoint_state_sha256"], "Checkpoint digest mismatch")
    expected_epochs = 1 if phase == "smoke" else 300
    require(c["epochs"] == expected_epochs and s["epochs"] == expected_epochs, "Epoch budget changed")
    minimum = 1 if phase == "smoke" else 40
    require(c["early_stopping"]["min_epochs"] == c["early_stopping"]["patience"] == minimum, "Early stopping changed")
    result = {"status": "passed", "dataset": row["dataset"], "phase": phase,
              "summary_sha256": digest(output / SUMMARY), "checkpoint_file_sha256": digest(checkpoint_path),
              "checkpoint_state_sha256": s["checkpoint_state_sha256"], "finite": True,
              "held_out_test_evaluated": False, "train_validation_counts_match": True}
    save(output / "audit.json", result)
    return s


def compare(reference, candidate, gate):
    def metrics(summary):
        rows = summary["quantity_rows"]
        body = [r for r in rows if r["stratum"] in {"le_p50", "p50_p90", "p90_p95"}]
        require(len(body) == 3, "Missing body strata")
        tail = next(r for r in rows if r["stratum"] == "gt_p99")
        return {"body_mae": sum(r["count"] * r["qty_mae"] for r in body) / sum(r["count"] for r in body),
                "qty_mae": summary["best_val_qty_mae"], "qty_rmse": summary["best_val_qty_rmse"],
                "gt_p99_mae": tail["qty_mae"], "time_nll": summary["best_val_time_nll"]}
    a, b = metrics(reference), metrics(candidate)
    relative = {key: b[key] / a[key] - 1. for key in ("body_mae", "qty_mae", "qty_rmse", "gt_p99_mae")}
    time_delta = b["time_nll"] - a["time_nll"]
    checks = {"body": relative["body_mae"] <= -gate["body_le_p95_mae_improvement_min"],
              "rmse": relative["qty_rmse"] <= gate["overall_rmse_regression_max"],
              "tail": relative["gt_p99_mae"] <= gate["gt_p99_mae_regression_max"],
              "time": time_delta <= gate["time_nll_absolute_increase_max"]}
    return {"baseline": a, "candidate": b, "relative_changes": relative,
            "time_nll_delta": time_delta, "gates": checks, "passed": all(checks.values())}


def run_logged(cmd, path, *, env=None):
    print("RUN", " ".join(cmd), flush=True)
    with path.open("w") as stream:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "screening"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--smoke-root", type=Path)
    args = parser.parse_args()
    # Never enter a prior artifact: the underlying legacy runner can auto-resume.
    args.output_root.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "phase": args.phase, "source_revision": args.source_revision,
              "started_at": datetime.now(timezone.utc).isoformat(), "completed": [], "current_dataset": None}
    status_path = args.output_root / "status.json"
    save(status_path, status)
    try:
        status["source_manifest_sha256"] = verify_source(args.source_revision)
        contract = read(CONTRACT)
        registry_path = ROOT / contract["baseline_registry"]
        require(digest(registry_path) == contract["baseline_registry_sha256"], "Registry changed")
        rows = {r["dataset"]: r for r in read(registry_path)["datasets"]}
        references = {name: baseline(rows[name], args.project_root)[1] for name in contract["dataset_order"]}
        save(args.output_root / "experiment_contract.json", contract)
        save(args.output_root / "baseline_registry.json", list(rows.values()))
        if args.phase == "screening":
            require(args.smoke_root is not None, "Screening requires a completed smoke")
            smoke = read(args.smoke_root / "status.json")
            require(smoke["status"] == "complete" and smoke["source_revision"] == args.source_revision and
                    smoke["source_manifest_sha256"] == status["source_manifest_sha256"], "Smoke/source mismatch")
            for name in contract["dataset_order"]:
                audit_run(args.smoke_root / name, rows[name], references[name], args.source_revision, "smoke", contract)
        else:
            status["cuda_preflight"] = preflight(contract["minimum_free_vram_mib"])
            save(status_path, status)
            env = dict(os.environ, WEIGHTED_STATIC_TEST_DEVICE="cuda")
            run_logged([sys.executable, "-s", "-m", "pytest", "-q", "simple_lab_test/search/tests/test_hard_lmm_weighted_static_memory.py"],
                       args.output_root / "cuda_contract_tests.log", env=env)
            status["cuda_contract_tests"] = "passed"
        comparisons = {}
        for name in contract["dataset_order"]:
            status["current_dataset"] = name
            status["preflight"] = preflight(contract["minimum_free_vram_mib"])
            verify_source(args.source_revision)
            cmd = command(rows[name], args.project_root, args.output_root / name, args.source_revision, args.phase)
            status["command"] = cmd
            save(status_path, status)
            run_logged(cmd, args.output_root / f"{name}.log")
            summary = audit_run(args.output_root / name, rows[name], references[name], args.source_revision, args.phase, contract)
            if args.phase == "screening":
                comparisons[name] = compare(references[name], summary, contract["per_dataset_gate"])
                save(args.output_root / "comparison.json", comparisons)
            status["completed"].append(name)
            save(status_path, status)
        status.update(status="complete", current_dataset=None, completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=datetime.now(timezone.utc).isoformat())
        save(status_path, status)
        raise
    save(status_path, status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
