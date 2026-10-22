"""Fabricated-only CPU regression of the unchanged native training entry path.

Run directly with --output-dir CPU_E2E_<unique>. The parent process records the
worker's actual OS exit and original stdout/stderr. No dataset or GPU is used.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
LOCKED_SCIENCE_SHA = "75b6a425b181175e89a6086f6f89a8e1d3f2f953897322389112c7363d042cc3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def worker(output):
    import numpy as np
    import torch
    from paper.scripts import run_shared_history_mlp_control as runner
    from paper.scripts import run_titantpp_shared_history_cap as fixture
    from paper.scripts import profile_shared_history_mlp_control as profiler
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    original_sources = {
        name: sha(ROOT / name)
        for name in (
            "paper/scripts/run_shared_history_mlp_control.py",
            "paper/scripts/manage_shared_history_mlp_control.py",
            "paper/scripts/run_multilag_detail_execution.py",
            "paper/scripts/count_aware_tpp_backbone/training.py",
            "paper/scripts/count_aware_tpp_backbone/core.py",
            "models/TPPs/CountAwareHistoryMLPControl.py",
        )
    }
    source = {"files": original_sources, "files_sha256": runner._canonical(original_sources),
              "base_git_revision": "fabricated_cpu_e2e_only"}
    fixture_dir = output / "fabricated"
    fixture_dir.mkdir()
    _, frame, data, _, quantity = fixture._engine_fixture(
        fixture_dir, "titantpp_history_mlp_shared15_v1")
    assert set(frame["chronological_split"].unique().to_list()) == {"train", "validation"}
    data.update(dataset_id="raf_spare_parts", optimizer={
        "name": "AdamW", "lr": 0.001, "weight_decay": 0.01, "grad_clip": 1.0},
        quantity_boundaries_all_train_rows=quantity["boundaries"])
    data["model"].update(lambda_log_qty=1.0, lambda_tail=0.0, time_head_lr_multiplier=1.0)
    data["loader"].update(batch_size=128, lookback_weeks=84, num_workers=0,
        drop_last=False, train_shuffle=True, validation_shuffle=False)
    temporal = derive_train_time_contract(frame, lookback_weeks=84, max_seq_len=84)
    data["time_statistics"] = {"train_time_scale": temporal["time_scale"],
        "train_log_scaled_mean": temporal["target_log_scaled_mean"],
        "train_log_scaled_std": temporal["target_log_scaled_std"]}
    populations = {}
    loaders = {}
    for split in ("train", "validation"):
        loaders[split] = make_loader(frame, target_split=split, shuffle=False, generator=None,
            batch_size=128, lookback_weeks=84, max_seq_len=84)
        dataset = loaders[split].dataset
        _, populations[split] = runner.population_identity(dataset, split)
        values = torch.tensor(np.asarray([
            max(1.0, float(dataset.dt_lists[p][i + 1])) for p, i in dataset.index]), dtype=torch.float64)
        data["time_statistics"][f"expected_{split}_targets"] = values.numel()
        data["time_statistics"][f"expected_{split}_target_dt_sha256"] = target_dt_sha256(values)
    data["inherited_data_identity"] = {"populations": populations}
    for key, filename in (("data", "fabricated_train_validation.parquet"),
                          ("split_manifest", "fabricated_manifest.json")):
        path = fixture_dir / filename
        data["inherited_data_identity"][key] = {"path": str(path), "sha256": sha(path)}
    jobs = []
    for dataset in runner.DATASETS:
        for seed in runner.SEEDS:
            for arm in runner.ARMS:
                job = {"id": f"{dataset}__{arm}__seed{seed}", "dataset": dataset,
                       "seed": seed, "arm": arm, "host": "5080" if seed == 42 else "5090"}
                if arm != "single_mlp":
                    job["reuse"] = {"underlying_arm": runner.UNDERLYING_ARMS[arm],
                        "parent_job_id": f"{dataset}__{runner.UNDERLYING_ARMS[arm]}__seed{seed}"}
                jobs.append(job)
    # The complete matrix is structural admission only; exactly one fabricated
    # MLP is trained. Other matrix entries are never run or read from disk.
    contract = {"schema": runner.SCHEMA, "campaign": runner.CAMPAIGN, "scope": "execution",
        "execution_scope": "validation_only", "evaluation_scope": "validation_only",
        "held_out_test_authorized": False, "model_role": "observed_time_joint_v1",
        "implementation_qualification": {"status": "passed", "training_allowed": True},
        "common_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
                     "batch_size": 128, "monitor": runner.MONITOR},
        "hosts": {host: {"root": str(output / host)} for host in ("5080", "5090")},
        "source": source, "input_files": {f"fabricated_{i}": "0" * 64 for i in range(9)},
        "architecture": {"scope": "fabricated_interface_identity_only"},
        "science_spec_sha256": LOCKED_SCIENCE_SHA, "datasets": [data], "jobs": jobs}
    science = ROOT / "reports/shared_history_mlp_control_20261022_v1/SCIENCE_SPEC.json"
    assert sha(science) == LOCKED_SCIENCE_SHA
    try:
        runner._training_interface(contract, data, frame, "single_mlp")
    except KeyError as error:
        assert error.args == ("design_sha256",)
        (output / "missing_design_original_traceback.txt").write_text(traceback.format_exc())
    else:
        raise AssertionError("Missing design SHA did not reproduce original entry failure")
    contract["design_sha256"] = LOCKED_SCIENCE_SHA
    assert contract["design_sha256"] == contract["science_spec_sha256"]
    job, _, host_root, folder = runner.select_job(contract, "5080", "raf_spare_parts__single_mlp__seed42")
    folder.mkdir(parents=True)
    args = runner._training_args(contract, data, folder, "single_mlp")
    assert (args.epochs, args.min_epochs, args.early_stopping_patience, args.batch_size) == (300, 40, 40, 128)
    assert args.lr == 0.001 and args.weight_decay == 0.01 and args.grad_clip == 1.0
    interface = runner._training_interface(contract, data, frame, "single_mlp")
    args.epochs, args.min_epochs, args.early_stopping_patience, args.device = 2, 1, 40, "cpu"
    with runner._engine_hooks(data, "single_mlp") as tracked:
        summary, _, _ = training.train_one(args=args, frame=frame,
            quantity_contract={"boundaries": quantity["boundaries"], "strata": []},
            interface_meta=interface, backbone="single_mlp", quantity_variant=runner.VARIANT, seed=42)
        proof = tracked["tracker"].proof()
    assert summary["completed_epochs"] == 2 and proof["optimizer_steps_checked"] > 0
    checkpoint = Path(summary["checkpoint_path"])
    checkpoint_before = sha(checkpoint)
    history = json.loads((checkpoint.parent / "history.json").read_text())["history"]
    assert [row["epoch"] for row in history] == [1, 2]
    assert runner.raw9.prepared.select_best_epoch(history) == summary["best_epoch"]
    receipt = {**runner._receipt_base(contract, "5080", job, "train"),
        "training_origin": "fresh_fit", "new_training_epochs": 2,
        "selected_checkpoint": str(checkpoint.relative_to(host_root)),
        "selected_epoch": summary["best_epoch"], "checkpoint_sha256": checkpoint_before,
        "state_sha256": summary["checkpoint_state_sha256"],
        "time_head_training_verification": proof}
    write(folder / "train_receipt.json", receipt)
    model, selected = runner._load_model(contract, data, job, receipt, host_root, "cpu")
    state_before = canonical_state_dict_sha256(model.state_dict())
    val1 = runner.evaluate_selected(contract, data, job, receipt, host_root, frame,
        "validation", folder / "validation1.jsonl.gz", device="cpu")
    val2 = runner.evaluate_selected(contract, data, job, receipt, host_root, frame,
        "validation", folder / "validation2.jsonl.gz", device="cpu")
    runner.raw9.compare_metrics(val1["metrics"], val2["metrics"])
    runner.compare_prediction_files(val1["prediction_path"], val2["prediction_path"])
    assert val1["prediction_rows"] == populations["validation"]["target_count"]
    assert val1["parameters_unchanged"] and val1["state_sha256"] == state_before
    with gzip.open(val1["prediction_path"], "rt") as stream:
        rows = [json.loads(line) for line in stream]
    errors = np.asarray([r["predicted_raw_quantity"] - r["raw_quantity"] for r in rows])
    assert np.isclose(float(np.sqrt(np.mean(errors ** 2))), summary["best_val_qty_rmse"], rtol=1e-10, atol=1e-8)
    assert np.isclose(float(np.abs(errors).mean()), summary["best_val_qty_mae"], rtol=1e-10, atol=1e-8)
    assert np.isclose(float(np.mean([r["time_nll"] for r in rows])), summary["best_val_time_nll"], rtol=1e-10, atol=1e-8)

    class CPUHooks:
        def sync(self): pass
        def clock(self): return time.perf_counter()
        def reset_peak(self): pass
        def memory(self): return {"allocated": 0, "reserved": 0}
        def peak(self): return {"allocated": 0, "reserved": 0}
        def transfer(self, batch): return {key: value.to("cpu") for key, value in batch.items()}

    _, dts, mask, _, quantities = next(iter(loaders["train"]))
    batches = [{"dts": dts.clone(), "mask": mask.clone(), "quantities": quantities.clone()} for _ in range(37)]
    costs = {}
    for mode in ("train", "eval"):
        cost_model = copy.deepcopy(model).requires_grad_(True)
        optimizer = profiler._optimizer(cost_model, data) if mode == "train" else None
        costs[mode] = profiler.measure_batches(cost_model, batches, mode, optimizer, CPUHooks(),
            profiler._target_outputs, grad_clip=1.0)
        assert costs[mode]["optimizer_steps"] == (37 if mode == "train" else 0)
        after = canonical_state_dict_sha256(cost_model.state_dict())
        assert (after != state_before) if mode == "train" else (after == state_before)
        assert sha(checkpoint) == checkpoint_before
    assert canonical_state_dict_sha256(model.state_dict()) == state_before
    assert all(sha(ROOT / name) == digest for name, digest in original_sources.items())
    outcome = {"status": "PASS", "scope": "fabricated_cpu_only", "remote_calls": 0,
        "real_data_loaded": False, "gpu_used": False, "original_sources_unchanged": original_sources,
        "missing_design_failure_reproduced": True, "design_sha256": LOCKED_SCIENCE_SHA,
        "native_training_args_asserted": {"epochs": 300, "min_epochs": 40, "patience": 40, "batch_size": 128},
        "test_only_overrides": {"epochs": 2, "min_epochs": 1, "device": "cpu"},
        "actual_completed_epochs": summary["completed_epochs"], "optimizer_steps_checked": proof["optimizer_steps_checked"],
        "selected_epoch": summary["best_epoch"], "selection_is_first_strict_minimum": True,
        "selected_checkpoint_sha256": checkpoint_before, "selected_state_sha256": state_before,
        "reload_validation_passes": 2, "validation_rows": len(rows), "validation_metrics": val1["metrics"],
        "validation_matches_training_summary": True, "individual_validation_reproduced": True,
        "time_head_training_verification": proof, "checkpoint_binary_unchanged": True,
        "cpu_cost_path": {"train_optimizer_steps": costs["train"]["optimizer_steps"],
            "eval_optimizer_steps": costs["eval"]["optimizer_steps"], "eval_state_unchanged": True,
            "batches_per_mode": 37, "data": "repeated fabricated batch; not benchmark data",
            "timing_memory_scientific_measurement": False, "cpu_memory_placeholders_not_measurements": True}}
    write(output / "outcome.json", outcome)
    print(json.dumps(outcome, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    assert output.parent == Path(__file__).resolve().parent and output.name.startswith("CPU_E2E_")
    if args.worker:
        worker(output)
        return
    output.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        process = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
            "--output-dir", str(output), "--worker"], cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
            timeout=180)
    inventory = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(output.rglob("*")) if p.is_file()}
    receipt = {"status": "PASS" if process.returncode == 0 else "FAIL",
        "actual_os_exit_code": process.returncode, "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
        "scope": "fabricated_cpu_only", "files": inventory}
    write(output / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
