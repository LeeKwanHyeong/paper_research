#!/usr/bin/env python3
"""Run fixed-checkpoint probes in independent, fail-closed dataset processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset, collate_week_lookback
from paper.scripts.analyze_count_aware_b0_retrieval import (
    DatasetSpec, checkpoint_path, summary_path, restore_b0, sha256_file, validate_dataset_contract,
)
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, target_outputs
from paper.scripts.hard_lmm_frozen_probe import (
    FrozenResidualProbe, acceptance, extract_features, fit_probe, metric_values,
    predict, sample_indices, summarize, write_event_deltas,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

CONTRACT_PATH = SOURCE_ROOT / "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json"


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text())


def fresh_directory(path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.mkdir(parents=True)


def verify_hashes(root, files):
    for name, expected in files.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")


def verify_source(contract, manifest_path, required):
    verify_hashes(SOURCE_ROOT, contract["frozen_base_files"])
    if manifest_path is None:
        if required:
            raise ValueError("Full execution requires a committed source manifest")
        return {"source_revision": "local_test_only", "frozen_base_unchanged": True}
    manifest = load_json(manifest_path)
    expected = set(contract["frozen_base_files"]) | {
        "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json",
        "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.md",
        "paper/scripts/hard_lmm_frozen_probe.py", "paper/scripts/run_hard_lmm_frozen_probe.py",
    }
    if not expected <= set(manifest["files"]):
        raise ValueError("Incomplete execution source manifest")
    revision = manifest["source_revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Full source revision required")
    verify_hashes(SOURCE_ROOT, manifest["files"])
    return manifest


def verify_dataset(project, row):
    artifact = project / row["artifact_dir"]
    files = {row["data_path"]: row["data_sha256"], row["split_manifest_path"]: row["split_manifest_sha256"],
        str((artifact / "launch_contract.json").relative_to(project)): row["contract_sha256"],
        str(summary_path(artifact, 42).relative_to(project)): row["summary_sha256"],
        str(checkpoint_path(artifact, 42).relative_to(project)): row["checkpoint_file_sha256"]}
    verify_hashes(project, files)
    spec = DatasetSpec(row["dataset"], row["contract_dataset"], artifact, project / row["data_path"])
    launch = validate_dataset_contract(spec)
    summary = load_json(summary_path(artifact, 42))
    if summary["status"] != "success" or summary["held_out_test_evaluated"] is not False:
        raise ValueError("Baseline summary is not successful validation-only evidence")
    if (int(launch["lookback_weeks"]), int(launch["max_seq_len"])) != (row["lookback"], row["max_seq_len"]):
        raise ValueError("Baseline context differs from the pinned contract")
    return spec, launch, summary


def gpu_preflight():
    def read(cmd, allowed=(0,)):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode not in allowed:
            raise RuntimeError(f"Preflight failed: {cmd}: {result.stderr}")
        return result.stdout.strip()
    gpu = read(["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"])
    compute = read(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    gdm = read(["systemctl", "is-active", "gdm"], (0, 3))
    names = read(["ps", "-eo", "comm="])
    if len(gpu.splitlines()) != 1:
        raise RuntimeError("Expected one RTX 5080")
    name, free = (field.strip() for field in gpu.split(","))
    if "RTX 5080" not in name or int(free) < 12000 or compute or gdm != "inactive":
        raise RuntimeError(f"GPU resource preflight rejected: {gpu}, processes={compute}, gdm={gdm}")
    if {"gnome-shell", "Xwayland"} & set(names.split()):
        raise RuntimeError("Desktop GPU processes must be absent")
    return {"gpu": gpu, "compute_processes": compute, "gdm": gdm}


def visible_frame(path):
    # Predicate pushdown excludes test rows before any tensor/dataframe materialization.
    frame = pl.scan_parquet(path).filter(pl.col("chronological_split").is_in(["train", "validation"])).collect()
    if set(frame["chronological_split"].unique().to_list()) != {"train", "validation"}:
        raise ValueError("Both train and validation splits are required")
    return prepare_count_frame(frame)


def make_dataset(frame, row, split):
    if split not in ("train", "validation"):
        raise ValueError("Held-out targets are prohibited")
    return RMTPPWeekLookbackDataset(frame, lookback_weeks=row["lookback"], max_seq_len=row["max_seq_len"],
        mode="all", split_col="chronological_split", target_splits={split})


def build_cache(model, dataset, indices, device, progress):
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=128, shuffle=False,
        collate_fn=collate_week_lookback, num_workers=0)
    tensors, offset, started = {}, 0, time.monotonic()
    for batch_number, (_, dts, mask, parts, quantities) in enumerate(loader):
        dts, mask, quantities = dts.to(device), mask.to(device), quantities.to(device)
        batch = extract_features(model, dts, mask, quantities)
        if batch_number == 0:
            with torch.no_grad():
                official = target_outputs(model, dts, mask, quantities, lambda_log_qty=1)
            expected = torch.nn.functional.softplus(batch["z"]).expm1()
            torch.testing.assert_close(expected, official["pred_qty"].cpu(), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(batch["time_nll"], official["time_loss"].cpu(), rtol=1e-5, atol=1e-6)
        count = len(batch["z"])
        selected = indices[offset:offset + count]
        batch["target_index"] = selected.clone()
        batch["series_index"] = parts.cpu()
        batch["context_end"] = torch.tensor([dataset.index[int(i)][1] for i in selected])
        for key, value in batch.items():
            tensors.setdefault(key, []).append(value)
        offset += count
        if batch_number % 100 == 0 or offset == len(indices):
            progress({"completed_targets": offset, "total_targets": len(indices),
                      "elapsed_seconds": time.monotonic() - started})
    if offset != len(indices):
        raise AssertionError("Feature cache target count mismatch")
    return {key: torch.cat(values) for key, values in tensors.items()}


def verify_baseline_replay(observed, summary):
    differences = {}
    for key in ("qty_mae", "qty_rmse", "time_nll", "joint_objective"):
        differences[key] = abs(observed[key] - summary[f"best_val_{key}"])
        if differences[key] > 1e-5:
            raise AssertionError(f"Frozen baseline replay mismatch: {key}, difference={differences[key]}")
    return differences


def dataset_worker(args, contract, row, output, status):
    spec, launch, summary = verify_dataset(args.project_root, row)
    model, audit = restore_b0(checkpoint_path(spec.artifact_dir, 42), launch, args.device)
    if audit["model_state_sha256"] != row["checkpoint_state_sha256"] or audit["checkpoint_source_revision"] != row["checkpoint_source_revision"]:
        raise ValueError("Restored baseline provenance differs from pinned checkpoint")
    model.requires_grad_(False).eval()
    before = canonical_state_dict_sha256(model.state_dict())
    frame = visible_frame(spec.data_path)
    cache, cache_audit = {}, {}
    for split in ("train", "validation"):
        dataset = make_dataset(frame, row, split)
        indices = (sample_indices(len(dataset), contract["probe_training"]["maximum_train_targets"])
                   if split == "train" else torch.arange(len(dataset)))
        if args.smoke:
            indices = indices[:256]
        save_json(output / f"{split}_series.json", dataset.parts)
        status(stage=f"extract_{split}", current_dataset=row["dataset"])
        cache[split] = build_cache(model, dataset, indices, args.device, lambda value: status(stage=f"extract_{split}", **value))
        path = output / f"{split}_cache.pt"
        torch.save(cache[split], path)
        cache_audit[split] = {"available_targets": len(dataset), "selected_targets": len(indices),
            "sha256": sha256_file(path), "sample_indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest()}
    peak_vram = torch.cuda.max_memory_allocated() if args.device == "cuda" else 0
    model.cpu()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    baseline = metric_values(cache["validation"]["z"], cache["validation"]["quantity"], cache["validation"]["time_nll"])
    replay = verify_baseline_replay(baseline, summary) if not args.smoke else {"status": "partial_smoke_only"}
    save_json(output / "baseline_audit.json", {**audit, "replay": replay, "baseline": baseline,
        "cache": cache_audit, "peak_cuda_allocated_bytes": peak_vram, "held_out_test_evaluated": False})
    results = {}
    for candidate in contract["candidates"]:
        policy = dict(contract["probe_training"])
        if args.smoke:
            policy.update(maximum_epochs=2, minimum_epochs=1, patience=2)
        history_file = output / f"{candidate}_history.jsonl"
        def progress(value):
            with history_file.open("a") as handle:
                handle.write(json.dumps(value, allow_nan=False) + "\n")
            status(stage="fit_cpu", candidate=candidate, **value)
        started = time.monotonic()
        probe, history, selection = fit_probe(candidate, cache["train"], cache["validation"], policy, progress)
        z, gate, correction = predict(probe, cache["validation"])
        path = output / f"{candidate}_best.pt"
        torch.save({"state_dict": probe.state_dict(), "candidate": candidate, "feature_dim": cache["train"]["features"].shape[-1],
            "baseline_state_sha256": before, "contract_sha256": sha256_file(CONTRACT_PATH), "selection": selection}, path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        restored = FrozenResidualProbe(payload["feature_dim"], candidate)
        restored.load_state_dict(payload["state_dict"], strict=True)
        reproduced = predict(restored, cache["validation"])
        if not all(torch.equal(a, b) for a, b in zip((z, gate, correction), reproduced)):
            raise AssertionError("Adapter checkpoint replay differs")
        after = canonical_state_dict_sha256(model.state_dict())
        if before != after or any(p.grad is not None for p in model.parameters()):
            raise AssertionError("Frozen base changed during adapter training")
        scopes = summarize(cache["validation"], z, gate, correction, launch["quantity_contract"]["boundaries"])
        decision = acceptance(scopes, selection["best_epoch"], not args.smoke)
        write_event_deltas(output / f"{candidate}_validation_events.parquet", cache["validation"], z, gate, correction)
        results[candidate] = {"selection": selection, "scopes": scopes, "decision": decision,
            "elapsed_seconds": time.monotonic() - started, "base_unchanged": before == after,
            "baseline_state_sha256_before": before, "baseline_state_sha256_after": after,
            "adapter_replay_exact": True, "adapter_sha256": sha256_file(path), "history": history}
        save_json(output / f"{candidate}_summary.json", results[candidate])
    metric_rows = []
    for candidate, result in results.items():
        for scope, values in result["scopes"].items():
            for role in ("baseline", "candidate"):
                metric_rows.append({"dataset": row["dataset"], "candidate": candidate, "role": role,
                    "scope": scope, "count": values["count"], "status": values["status"],
                    **values.get(role, {})})
    pl.DataFrame(metric_rows).write_csv(output / "scope_metrics.csv")
    save_json(output / "summary.json", {"status": "complete", "dataset": row["dataset"], "seed": 42,
        "full_validation": not args.smoke, "held_out_test_evaluated": False, "baseline": baseline, "candidates": results})
    status(status="complete", stage="dataset_complete", completed_candidates=2)


def run_guarded(output, action, metadata):
    fresh_directory(output)
    current = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), **metadata}
    def status(**changes):
        current.update(changes)
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "probe_status.json", current)
        print(json.dumps(changes, allow_nan=False), flush=True)
    status(stage="verify")
    try:
        action(status)
    except BaseException as error:
        status(status="failed", error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    contract = load_json(CONTRACT_PATH)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rows = [row for row in contract["datasets"] if not args.dataset or args.dataset == row["dataset"]]
    if not rows:
        raise ValueError("Unknown dataset")
    if args.worker and len(rows) != 1:
        raise ValueError("A worker must have exactly one dataset")
    if args.verify_only:
        source = verify_source(contract, args.source_manifest, not args.smoke)
        for row in rows:
            verify_dataset(args.project_root, row)
        print(json.dumps({"status": "verified", "datasets": [r["dataset"] for r in rows], **source}))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir required")
    args.output_dir = args.output_dir.resolve()
    def stop(signum, _frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def action(status):
        source = verify_source(contract, args.source_manifest, not args.smoke)
        save_json(args.output_dir / "source_manifest.json", source)
        save_json(args.output_dir / "launch_contract.json", {**contract, "partial_smoke": args.smoke,
            "selected_datasets": [r["dataset"] for r in rows], "device": args.device,
            "torch_version": torch.__version__, "fit_device": "cpu"})
        if args.worker:
            if args.device == "cuda":
                save_json(args.output_dir / "resource_preflight.json", gpu_preflight())
            dataset_worker(args, contract, rows[0], args.output_dir, status)
            return
        summaries = {}
        # Validate the whole grid before spending GPU time on the first dataset.
        for row in rows:
            verify_dataset(args.project_root, row)
        for row in rows:
            status(stage="dataset_worker", current_dataset=row["dataset"], completed_candidates=len(summaries) * 2)
            command = [sys.executable, str(Path(__file__).resolve()), "--project-root", str(args.project_root),
                "--output-dir", str(args.output_dir / row["dataset"]), "--dataset", row["dataset"], "--worker", "--device", args.device]
            if args.source_manifest:
                command.extend(("--source-manifest", str(args.source_manifest.resolve())))
            if args.smoke:
                command.append("--smoke")
            with (args.output_dir / f"{row['dataset']}.log").open("x") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            if code != 0:
                raise RuntimeError(f"Dataset worker failed: {row['dataset']} exit={code}; see dataset log/status")
            summaries[row["dataset"]] = load_json(args.output_dir / row["dataset"] / "summary.json")
        save_json(args.output_dir / "summary.json", {"status": "complete", "full_validation": not args.smoke,
            "held_out_test_evaluated": False, "datasets": summaries})
        status(status="complete", stage="complete", completed_candidates=len(summaries) * 2)
    run_guarded(args.output_dir, action, {"seed": 42, "partial_smoke": args.smoke, "held_out_test_evaluated": False})


if __name__ == "__main__":
    main()
