#!/usr/bin/env python3
"""Replicate registered frozen readouts on two existing Intermittent checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.analyze_count_aware_b0_retrieval import restore_b0, summary_path
from paper.scripts.hard_lmm_frozen_probe import (
    acceptance, metric_values, predict, sample_indices, summarize, write_event_deltas,
)
from paper.scripts.run_hard_lmm_frozen_probe import (
    build_cache, load_json, make_dataset, run_guarded, save_json, sha256_file,
    verify_baseline_replay, verify_hashes, visible_frame,
)
from paper.scripts.run_hard_lmm_readout_factorial import (
    SELECTORS, evaluate, fit, preflight, restore_checkpoint, save_checkpoint,
)
from paper.scripts.run_hard_lmm_smooth_shrinkage import checked_json
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_readout_seed_replication_v1.json"
ALIGNMENT = ("target_index", "series_index", "context_end", "history_length", "quantity")


def verify_alignment(cache, reference):
    if set(cache) != set(reference) or cache["features"].shape != reference["features"].shape:
        raise ValueError("Cache schema/shape mismatch")
    for key in ALIGNMENT:
        if not torch.equal(cache[key], reference[key]):
            raise ValueError(f"Seed-42 cache alignment mismatch: {key}")


def read_reference(contract, split, digests):
    if split not in ("train", "validation"):
        raise ValueError("Held-out split forbidden")
    root = ROOT / contract["parent_feature_artifact"]
    audit = checked_json(root / "baseline_audit.json", digests)
    path = root / f"{split}_cache.pt"
    if sha256_file(path) != audit["cache"][split]["sha256"]:
        raise ValueError("Seed-42 feature cache changed")
    digests[str(path)] = sha256_file(path)
    return torch.load(path, map_location="cpu", weights_only=True), audit["cache"][split]


def inherited_contract(contract, digests):
    if sha256_file(ROOT / contract["parent_contract"]) != contract["parent_contract_sha256"]:
        raise ValueError("Registered seed-42 contract changed")
    parent = checked_json(ROOT / contract["parent_contract"], digests)
    if contract["adapter_initialization_and_shuffle_seed"] != parent["training"]["seed"] or parent["training"]["shuffle_seed"] != 42:
        raise ValueError("Adapter seed must remain the discovery seed")
    if contract["heads"] != ["constant", "linear"] or contract["objectives"] != parent["objectives"] or contract["selectors"] != list(SELECTORS):
        raise ValueError("Replication grid changed")
    if contract["new_seeds"] != [52, 62] or [r["seed"] for r in contract["checkpoints"]] != [52, 62]:
        raise ValueError("Expected both new backbone seeds")
    return parent


def checked_model(contract, row, launch):
    if sha256_file(ROOT / row["path"]) != row["checkpoint_file_sha256"]:
        raise ValueError("Checkpoint file changed")
    model, audit = restore_b0(ROOT / row["path"], launch, "cpu")
    if audit["model_state_sha256"] != row["checkpoint_state_sha256"] or audit["checkpoint_source_revision"] != launch["source_revision"]:
        raise ValueError("Checkpoint identity mismatch")
    return model.requires_grad_(False).eval(), audit


def require_frozen(model, row):
    if canonical_state_dict_sha256(model.state_dict()) != row["checkpoint_state_sha256"] or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise AssertionError("Original backbone was modified")


def execute(contract, output, status):
    digests = {}
    parent = inherited_contract(contract, digests)
    if str(torch.__version__) != parent["runtime"]["torch_version"] or torch.get_num_threads() != 1:
        raise ValueError("Use existing torch2.7.1 one-thread CPU runtime")
    source_root = ROOT / contract["parent_artifact"]
    old_manifest = checked_json(source_root / "source_manifest.json", digests)
    if old_manifest["source_revision"] != contract["parent_execution_revision"]:
        raise ValueError("Discovery source revision mismatch")
    locked = {**old_manifest["parent_files"], **old_manifest["files"]}
    verify_hashes(ROOT, locked)
    old_outputs = checked_json(source_root / "output_digests.json", digests)
    verify_hashes(source_root, old_outputs)
    for path, digest in old_outputs.items():
        digests[str(source_root / path)] = digest
    for path, digest in load_json(source_root / "input_digests.json").items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Discovery input changed: {path}")
        digests[path] = digest
    own = [CONTRACT_PATH, Path(__file__), ROOT / "paper/scripts/validate_hard_lmm_readout_seed_replication.py"]
    files = {str(p.relative_to(ROOT)): sha256_file(p) for p in own}
    for path, digest in files.items():
        if hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest() != digest:
            raise ValueError(f"Uncommitted replication file: {path}")
    manifest = {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": files, "parent_files": locked, "python": sys.executable,
        "torch_version": str(torch.__version__), "device": "cpu", "threads": 1}
    save_json(output / "source_manifest.json", manifest)
    save_json(output / "launch_contract.json", contract)
    save_json(output / "inherited_contract.json", parent)
    original = checked_json(ROOT / parent["parent_contract"], digests)["datasets"][0]
    launch_path = ROOT / contract["baseline_artifact"] / "launch_contract.json"
    if sha256_file(launch_path) != original["contract_sha256"]:
        raise ValueError("Original launch contract changed")
    launch = checked_json(launch_path, digests)
    verify_hashes(ROOT, {original["data_path"]: original["data_sha256"],
        original["split_manifest_path"]: original["split_manifest_sha256"],
        contract["checkpoint_audit"]: contract["checkpoint_audit_sha256"]})
    for path in (original["data_path"], original["split_manifest_path"], contract["checkpoint_audit"]):
        digests[str(ROOT / path)] = sha256_file(ROOT / path)
    registry = load_json(ROOT / contract["checkpoint_audit"])
    summaries = {}
    for row in contract["checkpoints"]:
        seed = row["seed"]
        historical = next(r for r in registry if r["dataset"] == contract["dataset"] and r["seed"] == seed)
        if historical["checkpoint_file_sha256"] != row["checkpoint_file_sha256"] or historical["model_state_sha256"] != row["checkpoint_state_sha256"]:
            raise ValueError("Historical checkpoint registry mismatch")
        path = summary_path(ROOT / contract["baseline_artifact"], seed)
        if sha256_file(path) != row["summary_sha256"]:
            raise ValueError("Original summary changed")
        summaries[seed] = checked_json(path, digests)
        if summaries[seed]["seed"] != seed or summaries[seed]["checkpoint_state_sha256"] != row["checkpoint_state_sha256"] or summaries[seed]["status"] != "success" or summaries[seed]["held_out_test_evaluated"]:
            raise ValueError("Wrong seed or non-validation baseline")
        digests[str(ROOT / row["path"])] = row["checkpoint_file_sha256"]
    # The existing predicate excludes held-out rows before dataframe materialization.
    frame = visible_frame(ROOT / original["data_path"])
    train_dataset = make_dataset(frame, original, "train")
    train_indices = sample_indices(len(train_dataset), 65536, seed=42)
    reference_train, ref_train_audit = read_reference(contract, "train", digests)
    if len(train_dataset) != contract["cache"]["train_available"] or not torch.equal(train_indices, reference_train["target_index"]):
        raise ValueError("Train subset changed")
    cells = list(itertools.product(contract["heads"], contract["objectives"]))
    all_preflights, audits = {}, {}
    for row in contract["checkpoints"]:
        seed = row["seed"]
        directory = output / f"seed_{seed}"
        directory.mkdir()
        model, audit = checked_model(contract, row, launch)
        start = time.monotonic()
        preview = build_cache(model, train_dataset, train_indices[:256], "cpu", lambda _: None)
        if not torch.equal(preview["target_index"], train_indices[:256]):
            raise AssertionError("Inference preflight alignment mismatch")
        status(stage="train_inference_preflight_passed", seed=seed,
            preview_seconds=time.monotonic() - start, inference_only=True)
        del preview
        cache = build_cache(model, train_dataset, train_indices, "cpu",
            lambda values: status(stage="extract_train", seed=seed, **values))
        verify_alignment(cache, reference_train)
        require_frozen(model, row)
        torch.save(cache, directory / "train_cache.pt")
        audit.update(cache={"train": {**ref_train_audit, "sha256": sha256_file(directory / "train_cache.pt")}},
            seed=seed, held_out_test_evaluated=False, frozen_base_verified=True)
        audits[seed] = audit
        save_json(directory / "baseline_audit.json", audit)
        all_preflights[str(seed)] = {}
        for head, objective in cells:
            status(stage="train_only_readout_preflight", seed=seed, cell=f"{head}_{objective}")
            all_preflights[str(seed)][f"{head}_{objective}"] = preflight(head, objective, cache, parent["training"])
        del model, cache
    save_json(output / "train_only_preflight.json", all_preflights)
    reference_val, ref_val_audit = read_reference(contract, "validation", digests)
    val_dataset = make_dataset(frame, original, "validation")
    indices = torch.arange(len(val_dataset))
    if len(indices) != contract["cache"]["validation_selected"] or not torch.equal(indices, reference_val["target_index"]):
        raise ValueError("Full validation changed")
    for row in contract["checkpoints"]:
        seed = row["seed"]
        directory = output / f"seed_{seed}"
        model, _ = checked_model(contract, row, launch)
        cache = build_cache(model, val_dataset, indices, "cpu",
            lambda values: status(stage="extract_validation", seed=seed, **values))
        verify_alignment(cache, reference_val)
        require_frozen(model, row)
        baseline = metric_values(cache["z"], cache["quantity"], cache["time_nll"])
        audit = audits[seed]
        audit.update(baseline=baseline, replay=verify_baseline_replay(baseline, summaries[seed]))
        torch.save(cache, directory / "validation_cache.pt")
        audit["cache"]["validation"] = {**ref_val_audit, "sha256": sha256_file(directory / "validation_cache.pt")}
        save_json(directory / "baseline_audit.json", audit)
        del model, cache
    results = {"42": {f"{h}_{o}": checked_json(source_root / "intermittent_v2" / f"{h}_{o}_summary.json", digests)
        for h, o in cells}}
    boundaries = launch["quantity_contract"]["boundaries"]
    for row in contract["checkpoints"]:
        seed = row["seed"]
        directory = output / f"seed_{seed}"
        train = torch.load(directory / "train_cache.pt", map_location="cpu", weights_only=True)
        val = torch.load(directory / "validation_cache.pt", map_location="cpu", weights_only=True)
        results[str(seed)] = {}
        for head, objective in cells:
            cell = f"{head}_{objective}"
            status(stage="matched_fit", seed=seed, cell=cell)
            def progress(record):
                with (directory / f"{cell}_history.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            selected, final, result = fit(head, objective, train, val, boundaries[2], parent["training"], progress)
            result.update(backbone_seed=seed, boundaries=boundaries, train_targets=len(train["z"]),
                validation_targets=len(val["z"]), baseline_state_sha256=row["checkpoint_state_sha256"],
                held_out_test_evaluated=False, selections={})
            for tag, model in {**selected, "final": final}.items():
                path = directory / f"{cell}_{tag}.pt"
                save_checkpoint(path, model, row["checkpoint_state_sha256"], sha256_file(CONTRACT_PATH))
                replay = predict(restore_checkpoint(path, row["checkpoint_state_sha256"], sha256_file(CONTRACT_PATH)), val)
                if not all(torch.equal(a, b) for a, b in zip(predict(model, val), replay)):
                    raise AssertionError("Readout checkpoint restore mismatch")
                result[f"{tag}_checkpoint_sha256"] = sha256_file(path)
                if tag == "final":
                    continue
                z, gate, correction = replay
                scopes = summarize(val, z, gate, correction, boundaries)
                result["selections"][tag] = {"best_epoch": result["best_epochs"][tag], "scopes": scopes,
                    "decision": acceptance(scopes, result["best_epochs"][tag], True), "train": evaluate(model, train, boundaries[2])}
                write_event_deltas(directory / f"{cell}_{tag}_events.parquet", val, z, gate, correction)
            finite_json(result)
            save_json(directory / f"{cell}_summary.json", result)
            results[str(seed)][cell] = result
    for path, digest in digests.items():
        if sha256_file(Path(path)) != digest:
            raise AssertionError(f"Input modified: {path}")
    verify_hashes(ROOT, {**locked, **files})
    save_json(output / "input_digests.json", digests)
    save_json(output / "summary.json", {"status": "complete", "new_fits": 8, "reused_seed42_fits": 4,
        "seeds": results, "held_out_test_evaluated": False, "fresh_training_authorized": False})
    save_json(output / "output_digests.json", {str(p.relative_to(output)): sha256_file(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "probe_status.json"})
    status(status="complete", stage="replication_fits_complete", new_fits=8, reused_seed42_fits=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_guarded(args.output.resolve(), lambda status: execute(load_json(CONTRACT_PATH), args.output.resolve(), status),
        {"held_out_test_evaluated": False, "fresh_backbone_training": False})
