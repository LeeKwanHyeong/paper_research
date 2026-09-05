#!/usr/bin/env python3
"""Run the frozen Instacart raw64-versus-h64 accessibility diagnostic.

The input phase is deliberately separate from target/checkpoint access.  All
large tensors are stored under ignored ``search_artifacts``; reader-facing
evidence is written under ``paper/results``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hard-lmm-raw-history-mpl")
for _thread_variable in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_variable] = "4"

import numpy as np
import torch

from paper.scripts import hard_lmm_instacart_raw_history_analysis as analysis
from paper.scripts.hard_lmm_instacart_blind_inputs import read_body_labels
from paper.scripts.run_hard_lmm_query_diagnostic import check, digest, read, save
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


CONTRACT = ROOT / "paper/contracts/hard_lmm_instacart_raw_history_access_v1.json"
RESULT = ROOT / "paper/results/hard_lmm_instacart_raw_history_20260905"
RAW = ROOT / "search_artifacts/hard_lmm_instacart_raw_history_20260905"
SOURCE_FILES = (
    "paper/contracts/hard_lmm_instacart_raw_history_access_v1.json",
    "paper/contracts/hard_lmm_instacart_raw_history_access_v1.md",
    "paper/scripts/hard_lmm_instacart_raw_history_analysis.py",
    "paper/scripts/run_hard_lmm_instacart_raw_history.py",
    "simple_lab_test/search/tests/test_hard_lmm_instacart_raw_history_analysis.py",
    "simple_lab_test/search/tests/test_hard_lmm_instacart_raw_history_runner.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def verify_hashes(hashes):
    for name, expected in hashes.items():
        check((ROOT / name).is_file(), f"Missing frozen file: {name}")
        check(digest(ROOT / name) == expected, f"Frozen file changed: {name}")


def source_hashes():
    tracked = subprocess.check_output(
        ["git", "ls-files", "models", "data_loader", "paper/scripts", "simple_lab_test/search/common"],
        cwd=ROOT, text=True,
    ).splitlines()
    names = sorted(set(tracked).union(SOURCE_FILES))
    return {name: digest(ROOT / name) for name in names if (ROOT / name).is_file()}


def load_contract():
    contract = read(CONTRACT)
    check(contract["contract_id"] == "hard_lmm_instacart_raw_history_access_v1", "Wrong contract")
    check(contract["cohort"]["expected_source_rows"] == 65536, "Source cohort changed")
    check(contract["cohort"]["expected_kept_rows"] == 65525, "Kept cohort changed")
    check(contract["input_boundary"]["raw_dimension"] == contract["input_boundary"]["h_dimension"] == 64,
          "Matched dimension changed")
    check(contract["probe"]["families"] == ["linear", "random128"], "Probe families changed")
    check(contract["probe"]["ridge"] == 1.0 and contract["probe"]["random_width"] == 128,
          "Probe capacity changed")
    check(contract["primary_decision"]["paired_series_bootstrap"]["repeats"] == 10000,
          "Bootstrap count changed")
    check(abs(contract["primary_decision"]["paired_series_bootstrap"]["lower_quantile"] - .05 / 24) < 1e-15,
          "Simultaneous lower quantile changed")
    check(contract["models"]["primary"] == "separate_key", "Primary checkpoint changed")
    return contract


def _load_frozen_inputs(contract):
    verify_hashes(contract["frozen_inputs"])
    gate = read(ROOT / "paper/results/hard_lmm_instacart_balanced_20260905/input_gate.json")
    check(gate.get("stage") == "input" and gate.get("passed") is True, "Prior input-only gate did not pass")
    check(gate.get("chosen_target_labels_read") is False and gate.get("checkpoints_restored") is False,
          "Prior input gate crossed the outcome barrier")
    histories = torch.load(
        ROOT / "search_artifacts/hard_lmm_instacart_balanced_20260905/histories.pt",
        map_location="cpu", weights_only=False,
    )
    selection = torch.load(
        ROOT / "search_artifacts/hard_lmm_instacart_balanced_20260905/selection.pt",
        map_location="cpu", weights_only=False,
    )
    return histories, selection


def _validate_histories(histories, selection, contract):
    required = {"dts", "quantities", "mask", "history_length"}
    check(required <= set(histories), "Incomplete observed histories")
    n = contract["cohort"]["expected_source_rows"]
    dts, quantities, mask, length = (histories[name] for name in ("dts", "quantities", "mask", "history_length"))
    check(dts.shape == quantities.shape == mask.shape == (n, 64), "Unexpected history tensor shape")
    check(length.shape == (n,) and mask.dtype == torch.bool, "Invalid history length or mask")
    check(torch.equal(length, torch.as_tensor(selection["history_length"])), "Selection/history length mismatch")
    expected_mask = torch.arange(64)[None, :] < length[:, None]
    check(torch.equal(mask, expected_mask), "Observed history must be prefix-valid and right padded")
    check(bool(torch.isfinite(dts).all() and torch.isfinite(quantities).all()), "Nonfinite history input")
    check(bool((dts[mask] > 0).all() and (quantities[mask] > 0).all()), "Observed values must be positive")
    check(bool((dts[~mask] == 0).all() and (quantities[~mask] == 0).all()), "Padding values must be zero")
    for name in ("series_index", "fold", "target_physical_row_id"):
        values = np.asarray(selection[name])
        check(values.shape == (n,) and np.isfinite(values).all(), f"Invalid selection field: {name}")
    series, folds = np.asarray(selection["series_index"], dtype=np.int64), np.asarray(selection["fold"], dtype=np.int64)
    check(np.unique(series).size == n, "Exactly one context per series is required")
    check(set(folds.tolist()) == {0, 1}, "Both frozen folds are required")
    return dts, quantities, mask, length, series, folds


def prepare_phase():
    contract = load_contract()
    check(not RESULT.exists() and not RAW.exists(), "Refusing to overwrite raw-history diagnostic artifacts")
    torch.set_num_threads(contract["runtime"]["torch_threads"])
    histories, selection = _load_frozen_inputs(contract)
    dts, quantities, _mask, length, series, folds = _validate_histories(histories, selection, contract)
    keep = (length >= 3) & (length <= 32)
    positions = torch.nonzero(keep, as_tuple=True)[0]
    check(len(positions) == contract["cohort"]["expected_kept_rows"], "Frozen H<=32 count changed")
    check(int((length > 32).sum()) == contract["cohort"]["expected_excluded_history_gt32"], "H>32 count changed")
    selected_length = length[positions]
    selected_series = torch.as_tensor(series)[positions]
    selected_folds = torch.as_tensor(folds)[positions]
    check(selected_series.unique().numel() == len(positions), "Retained series are not unique")
    check(set(selected_folds.tolist()) == {0, 1}, "Retained rows lost a fold")
    history_dt = dts[positions, :32].contiguous()
    history_quantity = quantities[positions, :32].contiguous()
    raw64 = analysis.build_raw64(history_dt, history_quantity, selected_length)
    history_only64 = analysis.build_h_only64(selected_length)
    cache = {
        "history_dt": history_dt,
        "history_quantity": history_quantity,
        "history_length": selected_length.clone(),
        "raw64": torch.as_tensor(raw64),
        "history_only64": torch.as_tensor(history_only64),
        "selection_position": positions.clone(),
        "series_id": selected_series.clone(),
        "fold": selected_folds.clone(),
        "target_physical_row_id": torch.as_tensor(selection["target_physical_row_id"])[positions].clone(),
    }
    for value in cache.values():
        check(bool(torch.isfinite(value).all()), "Nonfinite prepared input cache")
    RAW.mkdir(parents=True)
    RESULT.mkdir(parents=True)
    input_path = RAW / "input_cache.pt"
    torch.save(cache, input_path)
    sources = source_hashes()
    manifest = {
        "status": "prepared",
        "started_at": now(),
        "updated_at": now(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "contract_sha256": digest(CONTRACT),
        "source_hashes": sources,
        "frozen_input_hashes": contract["frozen_inputs"],
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "platform": platform.platform(), "device": "cpu", "torch_threads": torch.get_num_threads(),
                    "thread_environment": {name: os.environ[name] for name in
                                           ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")}},
        "cohort": {"source_rows": len(length), "kept_rows": len(positions),
                   "excluded_h_gt32": int((length > 32).sum()),
                   "fold_rows": {str(f): int((selected_folds == f).sum()) for f in (0, 1)},
                   "unique_series": int(selected_series.unique().numel()),
                   "history_min": int(selected_length.min()), "history_max": int(selected_length.max())},
        "input_cache": {"path": relative(input_path), "sha256": digest(input_path)},
        "target_quantities_read": False,
        "target_gaps_read": False,
        "checkpoints_restored": False,
        "validation_rows_materialized": False,
        "held_out_rows_materialized": False,
        "model_parameter_updates": False,
        "server_accessed": False,
    }
    save(RESULT / "execution_contract.json", contract)
    save(RESULT / "input_manifest.json", {
        "stage": "input_only_frozen",
        "passed": True,
        "completed_at": now(),
        "input_cache": manifest["input_cache"],
        "source_hashes": sources,
        "contract_sha256": manifest["contract_sha256"],
        "cohort": manifest["cohort"],
        "target_quantities_read": False,
        "target_gaps_read": False,
        "checkpoints_restored": False,
    })
    manifest["input_manifest_sha256"] = digest(RESULT / "input_manifest.json")
    save(RESULT / "execution_manifest.json", manifest)
    print(json.dumps(manifest["cohort"], sort_keys=True), flush=True)


def _require_stage(expected):
    manifest = read(RESULT / "execution_manifest.json")
    check(manifest.get("status") == expected, f"Expected execution stage {expected}")
    check(digest(CONTRACT) == manifest["contract_sha256"], "Contract changed after input freeze")
    verify_hashes(manifest["source_hashes"])
    verify_hashes(manifest["frozen_input_hashes"])
    check(digest(ROOT / manifest["input_cache"]["path"]) == manifest["input_cache"]["sha256"], "Input cache changed")
    marker = read(RESULT / "input_manifest.json")
    check(digest(RESULT / "input_manifest.json") == manifest["input_manifest_sha256"], "Input marker changed")
    check(marker.get("stage") == "input_only_frozen" and marker.get("passed") is True, "Input barrier missing")
    check(marker.get("target_quantities_read") is False and marker.get("checkpoints_restored") is False,
          "Input marker is not outcome blind")
    return manifest


@torch.inference_mode()
def extract_h_and_base(model, histories, positions, batch_size=128):
    check(not model.training and not any(parameter.requires_grad for parameter in model.parameters()),
          "Diagnostic checkpoint must be frozen eval")
    outputs_h, outputs_base = [], []
    for start in range(0, len(positions), batch_size):
        index = torch.as_tensor(positions[start:start + batch_size], dtype=torch.long)
        dts, quantities, mask = (histories[name][index] for name in ("dts", "quantities", "mask"))
        check(torch.equal(mask, torch.arange(mask.shape[1])[None, :] < mask.sum(1)[:, None]),
              "Inference histories must be prefix-valid observed-only inputs")
        local = model._encode_base(dts, quantities, mask, memory_write_mask=mask)
        rows, last = torch.arange(len(index)), mask.sum(1) - 1
        residual, _ = model.lmm.retrieve(local)
        h = local[rows, last]
        fused = h + residual[rows, last]
        base_logpred, _ = model.predict_quantity(fused)
        check(h.shape == (len(index), 64) and base_logpred.shape == (len(index),), "Frozen output shape changed")
        check(bool(torch.isfinite(h).all() and torch.isfinite(base_logpred).all()), "Nonfinite frozen output")
        outputs_h.append(h.detach().cpu())
        outputs_base.append(base_logpred.detach().cpu())
    return torch.cat(outputs_h), torch.cat(outputs_base)


def _restore(label, contract):
    # Imports and checkpoint reads stay below the persisted input-only barrier.
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
    from paper.scripts.analyze_count_aware_b0_retrieval import checkpoint_path, restore_b0
    from paper.scripts.diagnose_hard_lmm_weighted_static import restore

    registry = read(ROOT / "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json")
    prior = read(ROOT / "paper/contracts/hard_lmm_query_diagnostic_v1.json")
    kv = read(ROOT / "paper/results/hard_lmm_key_value_screening_5090_20260904/final_audit.json")
    row = next(record for record in registry["datasets"] if record["dataset"] == "insta_market_basket")
    if label == "original":
        launch_path = ROOT / row["artifact_dir"] / "launch_contract.json"
        check(digest(launch_path) == row["contract_sha256"], "Original launch contract changed")
        path = checkpoint_path(ROOT / row["artifact_dir"], contract["models"]["checkpoint_seed"])
        expected_file, expected_state = row["checkpoint_file_sha256"], row["checkpoint_state_sha256"]
        check(digest(path) == expected_file, "Original checkpoint file changed")
        model, _ = restore_b0(path, read(launch_path), "cpu")
    elif label == "separate_key":
        path = (ROOT / prior["separate_key_artifact"] / "insta_market_basket" / "runs" / KEY_VALUE_BACKBONE
                / "count_only_log_regression" / "seed_42" / "best_val_joint_objective_model.pt")
        record = kv["runs"]["insta_market_basket"]
        expected_file, expected_state = record["checkpoint_file_sha256"], record["checkpoint_state_sha256"]
        check(digest(path) == expected_file, "Separate-key checkpoint file changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_route(payload, KEY_VALUE_BACKBONE)
        del payload
        model, _ = restore(path, KEY_VALUE_BACKBONE)
    else:
        raise ValueError(f"Unknown checkpoint label: {label}")
    model.requires_grad_(False).eval()
    check(canonical_state_dict_sha256(model.state_dict()) == expected_state, "Checkpoint state hash changed")
    return model, path, expected_file, expected_state


def extract_phase():
    contract = load_contract()
    manifest = _require_stage("prepared")
    check(not (RAW / "probe_cache.pt").exists(), "Refusing to overwrite extracted probe cache")
    manifest.update(status="extracting", updated_at=now(), target_quantity_access_authorized=True)
    save(RESULT / "execution_manifest.json", manifest)
    inputs = torch.load(ROOT / manifest["input_cache"]["path"], map_location="cpu", weights_only=False)
    histories, selection = _load_frozen_inputs(contract)
    positions = inputs["selection_position"].numpy()
    labels = read_body_labels(
        ROOT / "sample_data/insta_market_basket/instacart_marked_target_with_split.parquet",
        selection, positions,
    )
    check(np.array_equal(labels["selection_position"], positions), "Target position order changed")
    check(np.array_equal(labels["target_physical_row_id"], inputs["target_physical_row_id"].numpy()),
          "Target row identity changed")
    quantity = torch.from_numpy(labels["quantity"].copy()).float()
    check(bool(torch.isfinite(quantity).all() and (quantity >= 0).all()), "Invalid target quantities")
    torch.set_num_threads(contract["runtime"]["torch_threads"])
    models, checkpoint_records = {}, {}
    for label in ("original", "separate_key"):
        model, path, file_hash, state_hash = _restore(label, contract)
        before = canonical_state_dict_sha256(model.state_dict())
        h64, base_logpred = extract_h_and_base(
            model, histories, positions,
            batch_size=contract["runtime"]["inference_batch_size"],
        )
        after = canonical_state_dict_sha256(model.state_dict())
        check(before == after == state_hash, "Frozen checkpoint mutated during extraction")
        models[label] = {"h64": h64, "base_logpred": base_logpred}
        checkpoint_records[label] = {
            "path": relative(path), "file_sha256": file_hash, "state_sha256": state_hash,
            "state_unchanged": True, "rows": len(quantity),
        }
        print(json.dumps({"model": label, "rows": len(quantity)}), flush=True)
        del model
    cache = {"quantity": quantity, "log_quantity": quantity.log1p(), "models": models}
    cache_path = RAW / "probe_cache.pt"
    torch.save(cache, cache_path)
    manifest.update(
        status="extracted", updated_at=now(), target_quantities_read=True,
        target_gaps_read=False, checkpoints_restored=True, model_parameter_updates=False,
        probe_cache={"path": relative(cache_path), "sha256": digest(cache_path)},
        checkpoints=checkpoint_records,
    )
    save(RESULT / "execution_manifest.json", manifest)


def _numpy_tree(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(_numpy_tree(child, f"{prefix}__{key}" if prefix else str(key)))
    elif isinstance(value, (np.ndarray, torch.Tensor)):
        result[prefix] = np.asarray(value)
    return result


def analyze_phase():
    contract = load_contract()
    manifest = _require_stage("extracted")
    manifest.update(status="analyzing", updated_at=now())
    save(RESULT / "execution_manifest.json", manifest)
    check(digest(ROOT / manifest["probe_cache"]["path"]) == manifest["probe_cache"]["sha256"], "Probe cache changed")
    inputs = torch.load(ROOT / manifest["input_cache"]["path"], map_location="cpu", weights_only=False)
    extracted = torch.load(ROOT / manifest["probe_cache"]["path"], map_location="cpu", weights_only=False)
    caches = {}
    for label in ("original", "separate_key"):
        caches[label] = {
            "history_dt": inputs["history_dt"],
            "history_quantity": inputs["history_quantity"],
            "history_length": inputs["history_length"],
            "raw64": inputs["raw64"],
            "history_only64": inputs["history_only64"],
            "h64": extracted["models"][label]["h64"],
            "quantity": extracted["quantity"],
            "log_quantity": extracted["log_quantity"],
            "base_logpred": extracted["models"][label]["base_logpred"],
            "series_id": inputs["series_id"],
            "fold": inputs["fold"],
            "body_threshold": contract["targets"]["body_threshold"],
            "tail_threshold": contract["targets"]["tail_threshold"],
        }
    summary, arrays = analysis.analyze_models(caches, bootstrap_repeats=10000)
    save(RESULT / "analysis.json", summary)
    save(RESULT / "evidence_decision.json", summary["decision"])
    oof_path = RAW / "oof_predictions.npz"
    np.savez_compressed(oof_path, **_numpy_tree(arrays))
    manifest.update(
        status="complete", updated_at=now(), analysis_sha256=digest(RESULT / "analysis.json"),
        evidence_decision_sha256=digest(RESULT / "evidence_decision.json"),
        oof_predictions={"path": relative(oof_path), "sha256": digest(oof_path)},
        target_gaps_read=False, validation_rows_materialized=False,
        held_out_rows_materialized=False, model_parameter_updates=False, server_accessed=False,
    )
    save(RESULT / "execution_manifest.json", manifest)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "extract", "analyze"))
    args = parser.parse_args()
    {"prepare": prepare_phase, "extract": extract_phase, "analyze": analyze_phase}[args.phase]()
