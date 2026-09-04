#!/usr/bin/env python3
"""Paired frozen train-only retrieval diagnosis; never optimize a TPP model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hard-lmm-query-mpl")

import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset, collate_week_lookback
from models.TPPs.CountAwareFactory import validate_checkpoint_route
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
from paper.scripts.analyze_count_aware_b0_retrieval import checkpoint_path, restore_b0
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, target_outputs
from paper.scripts.diagnose_hard_lmm_weighted_static import restore
from paper.scripts.hard_lmm_frozen_probe import sample_indices
from paper.scripts.hard_lmm_query_diagnostic_features import extract
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

CONTRACT = ROOT / "paper/contracts/hard_lmm_query_diagnostic_v1.json"
RESULT = ROOT / "paper/results/hard_lmm_query_diagnostic_20260905"
RAW = ROOT / "search_artifacts/hard_lmm_query_diagnostic_20260905"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def fold_for_series(series):
    return int.from_bytes(hashlib.sha256(f"20260905:{int(series)}".encode()).digest()[:8], "big") % 2


def load_train_frame(path):
    frame = pl.scan_parquet(path).filter(pl.col("chronological_split") == "train").collect()
    check(set(frame["chronological_split"].unique().to_list()) == {"train"}, "Non-train or empty population")
    return frame


def source_hashes():
    tracked = subprocess.check_output(["git", "ls-files", "models", "data_loader", "paper/scripts", "simple_lab_test/search/common"], cwd=ROOT, text=True).splitlines()
    diagnostic = [str(p.relative_to(ROOT)) for p in (ROOT / "paper/scripts").glob("*hard_lmm_query_diagnostic*.py")]
    return {name: digest(ROOT / name) for name in sorted(set(tracked + diagnostic)) if (ROOT / name).is_file()}


def numeric_summary(values):
    x = values.detach().double().flatten()
    check(bool(torch.isfinite(x).all()) and x.numel() > 0, "Invalid numeric summary")
    return {"mean": x.mean().item(), "std": x.std(unbiased=False).item(),
            "p05": x.quantile(.05).item(), "p50": x.quantile(.5).item(), "p95": x.quantile(.95).item(),
            "min": x.min().item(), "max": x.max().item()}


def retrieval_summary(cache, bank_size=64):
    counts = torch.bincount(cache["indices"].reshape(-1), minlength=bank_size).double()
    probs = counts / counts.sum()
    active = probs[probs > 0]
    p = cache["prediction"].double()
    local_p = (cache["z"] - cache["projection"]).double().exp()
    q = cache["quantity"].double()
    entropy = -(cache["weights"].double().clamp_min(1e-300).log() * cache["weights"].double()).sum(-1)
    result = {
        "sample_count": len(q), "series_count": cache["series_index"].unique().numel(),
        "fold_counts": [int((cache["fold"] == i).sum()) for i in (0, 1)],
        "prototype_selection_counts": counts.long().tolist(),
        "effective_prototypes": active.mul(active.log()).sum().neg().exp().item(),
        "most_used_four_selection_share": probs.topk(4).values.sum().item(),
        "weight_entropy": numeric_summary(entropy),
        "max_weight": numeric_summary(cache["weights"].max(-1).values),
        "projection": numeric_summary(cache["projection"]),
        "selected_value_projection_spread": numeric_summary(cache["value_projections"].max(-1).values - cache["value_projections"].min(-1).values),
        "h_norm": numeric_summary(cache["h_norm"]),
        "r_norm": numeric_summary(cache["r_norm"]),
        "log_residual": numeric_summary(cache["log_residual"]),
        "train_sample_mae": (p - q).abs().mean().item(),
        "memory_on_minus_off_mae": ((p - q).abs() - (local_p - q).abs()).mean().item(),
        "history_strata": [],
    }
    # Outcome-independent descriptive strata. No thresholds enter model input.
    for column, label in enumerate(("history_level", "latest_deviation")):
        values = cache["stats"][:, column].double()
        boundaries = values.quantile(torch.tensor([.25, .5, .75], dtype=torch.float64))
        buckets = torch.bucketize(values, boundaries)
        for group in range(4):
            selected = buckets == group
            record = {"feature": label, "quartile": group + 1, "count": int(selected.sum()), "boundaries": boundaries.tolist()}
            if bool(selected.any()):
                record.update({"feature_mean": values[selected].mean().item(),
                    "projection_mean": cache["projection"][selected].double().mean().item(),
                    "needed_log1p_correction_mean": cache["log_residual"][selected].double().mean().item(),
                    "memory_on_minus_off_mae": ((p[selected]-q[selected]).abs()-(local_p[selected]-q[selected]).abs()).mean().item()})
            result["history_strata"].append(record)
    return result


def paired_summary(a, b):
    for key in ("target_index", "series_index", "fold", "context_end", "quantity", "stats", "history_length"):
        check(torch.equal(a[key], b[key]), f"Paired input/identity mismatch: {key}")
    overlap = (a["indices"].unsqueeze(-1) == b["indices"].unsqueeze(-2)).any(-1).sum(-1)
    return {"paired_target_count": len(overlap), "same_top4_set_fraction": (overlap == 4).double().mean().item(),
            "shared_prototype_count": numeric_summary(overlap),
            "prediction_difference": numeric_summary(b["prediction"] - a["prediction"]),
            "projection_difference": numeric_summary(b["projection"] - a["projection"]),
            "caveat": "Same slot IDs across independently fitted models do not establish identical semantic prototypes."}


def extract_cache(model, dataset, indices, label):
    model.requires_grad_(False).eval()
    before = canonical_state_dict_sha256(model.state_dict())
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=64, shuffle=False,
                        collate_fn=collate_week_lookback, num_workers=0)
    arrays, offset = {}, 0
    started = time.monotonic()
    for batch_number, (_, dts, mask, parts, quantities) in enumerate(loader):
        batch = extract(model, dts, mask, quantities)
        if batch_number == 0:
            with torch.no_grad():
                official = target_outputs(model, dts, mask, quantities, lambda_log_qty=1)
            torch.testing.assert_close(batch["prediction"], official["pred_qty"].cpu(), rtol=1e-5, atol=1e-6)
        count = len(batch["z"])
        selected = indices[offset:offset + count]
        batch.update({"target_index": selected.clone(), "series_index": parts.cpu(),
                      "context_end": torch.tensor([dataset.index[int(i)][1] for i in selected]),
                      "fold": torch.tensor([fold_for_series(part) for part in parts])})
        for key, tensor in batch.items():
            check(bool(torch.isfinite(tensor).all()), f"Nonfinite cache field {key}")
            arrays.setdefault(key, []).append(tensor)
        offset += count
        if batch_number % 32 == 0 or offset == len(indices):
            print(json.dumps({"model": label, "completed": offset, "total": len(indices),
                              "seconds": round(time.monotonic()-started, 2)}), flush=True)
    check(offset == len(indices), "Incomplete sample")
    check(before == canonical_state_dict_sha256(model.state_dict()), "Frozen model state mutated")
    return {key: torch.cat(parts) for key, parts in arrays.items()}


def execute_extraction():
    check(not RAW.exists() and not RESULT.exists(), "Refusing to overwrite previous diagnostic output")
    contract = read(CONTRACT)
    check(digest(ROOT / contract["original_registry"]) == contract["original_registry_sha256"], "Registry changed")
    check(digest(ROOT / contract["separate_key_audit"]) == contract["separate_key_audit_sha256"], "Separate-key audit changed")
    registry = read(ROOT / contract["original_registry"])
    kv_audit = read(ROOT / contract["separate_key_audit"])
    torch.set_num_threads(contract["sampling"]["torch_threads"])
    torch.manual_seed(42)
    RAW.mkdir(parents=True)
    RESULT.mkdir(parents=True)
    manifest = {"status": "extracting", "started_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "contract_sha256": digest(CONTRACT), "source_files": source_hashes(),
        "reference_document_hashes": {contract["original_registry"]: contract["original_registry_sha256"],
                                      contract["separate_key_audit"]: contract["separate_key_audit_sha256"]},
        "runtime": {"python": sys.version, "torch": torch.__version__, "polars": pl.__version__, "platform": platform.platform(), "device": "cpu", "threads": torch.get_num_threads()},
        "validation_rows_materialized": False, "held_out_rows_materialized": False,
        "model_parameter_updates": False, "server_accessed": False, "datasets": {}}
    save(RESULT / "execution_manifest.json", manifest)
    try:
        for name in contract["datasets"]:
            row = next(r for r in registry["datasets"] if r["dataset"] == name)
            original = checkpoint_path(ROOT / row["artifact_dir"], 42)
            candidate = ROOT / contract["separate_key_artifact"] / name / "runs" / KEY_VALUE_BACKBONE / "count_only_log_regression/seed_42/best_val_joint_objective_model.pt"
            source_files = {row["data_path"]: row["data_sha256"], row["split_manifest_path"]: row["split_manifest_sha256"],
                str(original.relative_to(ROOT)): row["checkpoint_file_sha256"],
                str(candidate.relative_to(ROOT)): kv_audit["runs"][name]["checkpoint_file_sha256"],
                row["artifact_dir"] + "/launch_contract.json": row["contract_sha256"]}
            for file, expected in source_files.items():
                check(digest(ROOT / file) == expected, f"Input integrity failed: {file}")
            frame = load_train_frame(ROOT / row["data_path"])
            dataset = RMTPPWeekLookbackDataset(prepare_count_frame(frame), lookback_weeks=row["lookback"],
                max_seq_len=row["max_seq_len"], mode="all", split_col="chronological_split", target_splits={"train"})
            check(len(dataset) == kv_audit["runs"][name]["train_targets_per_epoch"], "Train population mismatch")
            indices = sample_indices(len(dataset), contract["sampling"]["maximum_targets_per_dataset"], seed=42)
            metadata = {"input_hashes": source_files, "train_rows": frame.height, "train_targets": len(dataset),
                        "train_series": len(dataset.parts), "sample_targets": len(indices),
                        "sample_index_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(), "models": {}}
            caches = {}
            for label, path in (("original", original), ("separate_key", candidate)):
                if label == "original":
                    model, audit = restore_b0(path, read(ROOT / row["artifact_dir"] / "launch_contract.json"), "cpu")
                    check(audit["model_state_sha256"] == row["checkpoint_state_sha256"], "Original state digest")
                else:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    validate_checkpoint_route(payload, KEY_VALUE_BACKBONE)
                    check(payload["encoder_config"]["max_len"] == row["max_seq_len"], "Separate-key context mismatch")
                    check(canonical_state_dict_sha256(payload["model_state_dict"]) == kv_audit["runs"][name]["checkpoint_state_sha256"], "Separate-key state digest")
                    model, _ = restore(path, KEY_VALUE_BACKBONE)
                cache = extract_cache(model, dataset, indices, f"{name}/{label}")
                for fold in (0, 1):
                    check(cache["series_index"][cache["fold"] == fold].unique().numel() >= 10, "Too few series per fold")
                cache_path = RAW / name / f"{label}_cache.pt"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cache, cache_path)
                metadata["models"][label] = {"cache_path": str(cache_path.relative_to(ROOT)), "cache_sha256": digest(cache_path),
                    "checkpoint_state_unchanged": True, "official_prediction_parity_first_batch": True,
                    "retrieval": retrieval_summary(cache)}
                caches[label] = cache
                del model
            metadata["paired"] = paired_summary(caches["original"], caches["separate_key"])
            manifest["datasets"][name] = metadata
            save(RESULT / "execution_manifest.json", manifest)
            del caches, frame, dataset
        check(digest(CONTRACT) == manifest["contract_sha256"], "Diagnostic contract changed during extraction")
        check(source_hashes() == manifest["source_files"], "Source changed during extraction")
        for file, expected in manifest["reference_document_hashes"].items():
            check(digest(ROOT / file) == expected, f"Reference changed during extraction: {file}")
        for dataset_info in manifest["datasets"].values():
            for file, expected in dataset_info["input_hashes"].items():
                check(digest(ROOT / file) == expected, f"Input changed during extraction: {file}")
        manifest.update(status="extracted", extraction_completed_at=datetime.now(timezone.utc).isoformat())
        save(RESULT / "execution_manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(RESULT / "execution_manifest.json", manifest)
        raise


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    execute_extraction()
