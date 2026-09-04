#!/usr/bin/env python3
"""One frozen active-context timing/error screen and bounded input sensitivity."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from models.TPPs.CountAwareFactory import validate_checkpoint_route
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
from paper.scripts.analyze_count_aware_b0_retrieval import checkpoint_path, restore_b0
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.diagnose_hard_lmm_weighted_static import restore
from paper.scripts.hard_lmm_temporal_features import observed_features, reverse_interior
from paper.scripts.hard_lmm_temporal_analysis import analyze
from paper.scripts.run_hard_lmm_query_diagnostic import read, save, digest, check, numeric_summary, load_train_frame
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

CONTRACT = ROOT / "paper/contracts/hard_lmm_temporal_diagnostic_v1.json"
RESULT = ROOT / "paper/results/hard_lmm_temporal_diagnostic_20260905"
RAW = ROOT / "search_artifacts/hard_lmm_temporal_diagnostic_20260905"


def summarize(values):
    values = torch.as_tensor(values)
    return numeric_summary(values) if values.numel() else None


def extract_histories(dataset, cache):
    """Use original selected dataset rows; remove target before feature API."""
    records, histories, maxlen_truncations, outside_context = [], [], 0, 0
    for offset, target_id in enumerate(cache["target_index"].tolist()):
        sample = dataset[target_id]
        dts = sample["dts"][sample["mask"]][:-1].clone()
        qty = sample["values"][sample["mask"]][:-1].clone()
        part, end = dataset.index[target_id]
        check(part == int(cache["series_index"][offset]) and end == int(cache["context_end"][offset]), "Dataset identity mismatch")
        check(len(qty) == int(cache["history_length"][offset]), "History length mismatch")
        check(sample["values"][sample["mask"]][-1] == cache["quantity"][offset], "Train target label mismatch")
        feature = observed_features(dts, qty)
        torch.testing.assert_close(torch.tensor([feature["mean_log_quantity"], feature["latest_deviation"]]), cache["stats"][offset], rtol=1e-5, atol=1e-6)
        feature.update(target_index=target_id, series_index=part, context_end=end, fold=int(cache["fold"][offset]))
        seq = np.asarray(dataset.seq_lists[part])
        start = int(np.searchsorted(seq, seq[end] - (dataset.W - 1)))
        maxlen_truncations += end - start + 2 > dataset.max_len
        outside_context += end + 1 > len(qty)
        records.append(feature)
        histories.append((dts, qty))
    arrays = {key: np.asarray([r[key] if r[key] is not None else np.nan for r in records]) for key in records[0]}
    valid = np.isfinite(arrays["age_distortion"])
    profile = {"sample_count": len(records), "eligible_age_count": int(valid.sum()),
        "age_distortion": summarize(arrays["age_distortion"][valid]),
        "equal_spacing_fraction_among_eligible": float(np.mean(arrays["age_distortion"][valid] <= 1e-12)) if valid.any() else None,
        "max_length_truncated_count": int(maxlen_truncations), "older_series_events_outside_context_count": int(outside_context),
        "observed_features": {key: {"defined_count": int(np.isfinite(value).sum()), "summary": summarize(value[np.isfinite(value)])}
                              for key, value in arrays.items() if key not in ("target_index", "series_index", "context_end", "fold")}}
    return arrays, histories, profile


@torch.no_grad()
def represent(model, histories, positions, max_len, modality=None):
    arrays = {key: [] for key in ("h", "z", "log_prediction", "time_intercept", "indices")}
    changed = []
    for start in range(0, len(positions), 64):
        batch = positions[start:start + 64]
        dts, qty = torch.zeros(len(batch), max_len), torch.zeros(len(batch), max_len)
        mask = torch.zeros_like(dts, dtype=torch.bool)
        lengths = []
        for row, position in enumerate(batch):
            original_dt, original_qty = histories[int(position)]
            dt, q = reverse_interior(original_dt, original_qty, modality) if modality else (original_dt, original_qty)
            changed.append(not torch.equal(dt, original_dt) or not torch.equal(q, original_qty))
            length = len(dt)
            dts[row, :length], qty[row, :length], mask[row, :length] = dt, q, True
            lengths.append(length)
        local = model._encode_base(dts, qty, mask, memory_write_mask=mask)
        residual, trace = model.lmm.retrieve(local)
        rows, previous = torch.arange(len(batch)), torch.tensor(lengths) - 1
        h, r = local[rows, previous], residual[rows, previous]
        z = model.quantity_head(h + r).squeeze(-1)
        batch_outputs = {"h": h, "z": z, "log_prediction": F.softplus(z),
            "time_intercept": (model.v_t(h + r).squeeze(-1) + model.b_t).clamp(max=model.time_intercept_limit),
            "indices": trace["prototype_indices"][rows, previous]}
        for key, value in batch_outputs.items():
            check(bool(torch.isfinite(value).all()), f"Nonfinite sensitivity output: {key}")
            arrays[key].append(value.cpu())
    return {**{key: torch.cat(value) for key, value in arrays.items()}, "input_changed": torch.tensor(changed)}


def sensitivity(model, histories, positions, cache, max_len):
    before = canonical_state_dict_sha256(model.state_dict())
    base = represent(model, histories, positions, max_len)
    parity = {}
    for key in ("h", "z"):
        torch.testing.assert_close(base[key], cache[key][positions], rtol=1e-5, atol=1e-6)
        parity[key + "_max_abs_error"] = float((base[key] - cache[key][positions]).abs().max())
    check(torch.equal(base["indices"], cache["indices"][positions]), "Base retrieval cache mismatch")
    output = {"sample_count": len(positions), "cache_positions_sha256": hashlib.sha256(positions.numpy().tobytes()).hexdigest(), "base_cache_parity": parity,
              "counterfactual_error_evaluated": False, "perturbations": {}}
    raw = {"base": base, "cache_positions": positions}
    for modality in ("gap", "quantity"):
        changed = represent(model, histories, positions, max_len, modality)
        raw[modality] = changed
        active = changed["input_changed"]
        shared = (base["indices"].unsqueeze(-1) == changed["indices"].unsqueeze(-2)).any(-1).sum(-1)
        values = {
            "local_relative_l2": (changed["h"] - base["h"]).norm(dim=1) / base["h"].norm(dim=1).clamp_min(1e-12),
            "local_cosine_distance": 1 - F.cosine_similarity(base["h"], changed["h"]),
            "absolute_logit_change": (changed["z"] - base["z"]).abs(),
            "absolute_log1p_prediction_change": (changed["log_prediction"] - base["log_prediction"]).abs(),
            "absolute_time_intercept_change": (changed["time_intercept"] - base["time_intercept"]).abs(),
        }
        output["perturbations"][modality] = {"changed_histories": int(active.sum()),
            "same_top4_fraction_all": float((shared == 4).double().mean()),
            "same_top4_fraction_changed": float((shared[active] == 4).double().mean()) if active.any() else None,
            "all_histories": {key: summarize(value) for key, value in values.items()},
            "changed_histories_only": {key: summarize(value[active]) for key, value in values.items()}}
    check(canonical_state_dict_sha256(model.state_dict()) == before, "Sensitivity changed frozen state")
    output["checkpoint_state_unchanged"] = True
    return output, raw


def main():
    check(not RESULT.exists() and not RAW.exists(), "Refusing to overwrite temporal diagnosis")
    contract = read(CONTRACT)
    check(digest(ROOT / contract["prior_manifest"]) == contract["prior_manifest_sha256"], "Prior manifest mismatch")
    prior = read(ROOT / contract["prior_manifest"])
    old_contract = read(ROOT / contract["prior_contract"])
    verified = {contract["prior_manifest"]: contract["prior_manifest_sha256"],
                contract["prior_contract"]: prior["contract_sha256"], **prior["source_files"], **prior["reference_document_hashes"]}
    for path, expected in verified.items():
        check(digest(ROOT / path) == expected, f"Source/reference mismatch: {path}")
    new_sources = [CONTRACT, Path(__file__), *(ROOT / "paper/scripts").glob("hard_lmm_temporal_*.py"),
                   *(ROOT / "simple_lab_test/search/tests").glob("test_hard_lmm_temporal_*.py")]
    verified.update({str(p.relative_to(ROOT)): digest(p) for p in new_sources})
    registry = read(ROOT / old_contract["original_registry"])
    kv_audit = read(ROOT / old_contract["separate_key_audit"])
    torch.set_num_threads(contract["runtime"]["threads"])
    a = contract["association"]
    policy = {"min_cell_rows": a["minimum_rows_per_cell_arm"], "min_cell_series": a["minimum_series_per_cell_arm"],
        "min_fold_arm_rows": a["minimum_rows_per_fold_arm"], "min_fold_arm_series": a["minimum_series_per_fold_arm"],
        "min_retained_fraction": a["minimum_retained_extreme_body_fraction_per_fold"], "max_smd": a["maximum_control_smd_per_fold"],
        "minimum_relative_excess": a["relevance_gate"]["minimum_pooled_relative_excess_body_mae"],
        "bootstrap_samples": a["bootstrap_samples"], "bootstrap_seed": a["bootstrap_seed"],
        "j_min_separation": a["minimum_feature_cutpoint_separation"]}
    manifest = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "threads": torch.get_num_threads()},
        "verified_hashes": verified, "association_policy": policy, "datasets": {},
        "held_out_rows_materialized": False, "validation_rows_materialized": False,
        "target_gap_used": False, "model_parameter_updates": False, "server_accessed": False}
    RESULT.mkdir(parents=True)
    RAW.mkdir(parents=True)
    save(RESULT / "execution_manifest.json", manifest)
    outputs, feature_rows = {}, []
    try:
        for name in contract["datasets"]:
            row = next(r for r in registry["datasets"] if r["dataset"] == name)
            for path, expected in prior["datasets"][name]["input_hashes"].items():
                check(digest(ROOT / path) == expected, f"Input changed: {path}")
                verified[path] = expected
            caches = {}
            for label in contract["models"]:
                meta = prior["datasets"][name]["models"][label]
                check(digest(ROOT / meta["cache_path"]) == meta["cache_sha256"], "Paired cache mismatch")
                verified[meta["cache_path"]] = meta["cache_sha256"]
                caches[label] = torch.load(ROOT / meta["cache_path"], map_location="cpu", weights_only=True)
            for key in ("target_index", "series_index", "context_end", "fold", "history_length", "quantity", "stats"):
                check(torch.equal(caches["original"][key], caches["separate_key"][key]), f"Paired rows differ: {key}")
            frame = load_train_frame(ROOT / row["data_path"])
            dataset = RMTPPWeekLookbackDataset(prepare_count_frame(frame), lookback_weeks=row["lookback"],
                max_seq_len=row["max_seq_len"], mode="all", target_splits={"train"})
            check(len(dataset) == prior["datasets"][name]["train_targets"], "Train target population mismatch")
            features, histories, profile = extract_histories(dataset, caches["original"])
            raw_path = RAW / name
            raw_path.mkdir()
            torch.save({key: torch.from_numpy(value) for key, value in features.items()}, raw_path / "features.pt")
            for index in range(len(histories)):
                feature_rows.append({"dataset": name, **{key: (value[index].item() if np.isfinite(value[index]) else None) for key, value in features.items()}})
            # H=1 controls are undefined; these rows cannot enter J comparisons.
            analysis_features = {key: np.nan_to_num(value) if key != "age_distortion" else value for key, value in features.items()}
            labels = {label: {key: cache[key].numpy() for key in ("prediction", "quantity", "log_residual")} for label, cache in caches.items()}
            launch = read(ROOT / row["artifact_dir"] / "launch_contract.json")
            check(launch["quantity_contract"]["boundaries"][2] == a["body_thresholds"][name], "Body p95 contract mismatch")
            association, assignment = analyze(analysis_features, labels, a["body_thresholds"][name], policy)
            torch.save({key: torch.as_tensor(value) for key, value in assignment.items()}, raw_path / "association_rows.pt")
            outputs[name] = {"profile": profile, "association": association, "sensitivity": {}}
            eligible = torch.from_numpy(np.flatnonzero(features["history_length"] >= 4))
            order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(contract["sensitivity"]["seed"]))
            positions = eligible[order[:contract["sensitivity"]["maximum_histories_per_dataset"]]].sort().values
            check(len(positions) > 0, "No sensitivity histories")
            state_hashes = {}
            for label in contract["models"]:
                if label == "original":
                    path = checkpoint_path(ROOT / row["artifact_dir"], 42)
                    model, _ = restore_b0(path, launch, "cpu")
                    expected_state = row["checkpoint_state_sha256"]
                else:
                    path = ROOT / old_contract["separate_key_artifact"] / name / "runs" / KEY_VALUE_BACKBONE / "count_only_log_regression/seed_42/best_val_joint_objective_model.pt"
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    validate_checkpoint_route(payload, KEY_VALUE_BACKBONE)
                    model, _ = restore(path, KEY_VALUE_BACKBONE)
                    del payload
                    expected_state = kv_audit["runs"][name]["checkpoint_state_sha256"]
                model.requires_grad_(False).eval()
                check(model.time_head_mode == "legacy_clamped_rmtpp", "Unexpected time head")
                check(canonical_state_dict_sha256(model.state_dict()) == expected_state, "Canonical state mismatch")
                state_hashes[label] = expected_state
                summary, sensitivity_raw = sensitivity(model, histories, positions, caches[label], row["max_seq_len"])
                outputs[name]["sensitivity"][label] = summary
                torch.save(sensitivity_raw, raw_path / f"{label}_sensitivity.pt")
                del model
            manifest["datasets"][name] = {"sample_count": len(histories), "train_rows": frame.height,
                "checkpoint_states": state_hashes, "raw_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(raw_path.glob("*.pt"))}}
            save(RESULT / "analysis.json", outputs)
            save(RESULT / "execution_manifest.json", manifest)
            print(f"Completed {name}: {len(histories)} train histories, {len(positions)} sensitivity histories per checkpoint", flush=True)
            del dataset, frame, caches, histories
        with (RAW / "history_features.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(feature_rows)
        for path, expected in verified.items():
            check(digest(ROOT / path) == expected, f"Source/input changed during diagnosis: {path}")
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                        output_hashes={"analysis.json": digest(RESULT / "analysis.json")},
                        raw_feature_table={"path": str((RAW / "history_features.csv").relative_to(ROOT)), "sha256": digest(RAW / "history_features.csv")})
        save(RESULT / "execution_manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(RESULT / "execution_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
