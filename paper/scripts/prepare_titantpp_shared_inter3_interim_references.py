"""Audit stored original Intermittent references; never construct or infer a model.

Test21 and MLP16 Validation3 have individual prediction shards. External
Validation18 are existing selected endpoint aggregates, explicitly identified as
such; their original checkpoint/state/history and population are bound below.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

from paper.scripts import prepare_titantpp_shared_raf_taxi6_references as parent

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/titantpp_shared_inter3_interim_test_20261020_v1"
DATASET = "intermittent_frozen_5000"
DATA = "sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet"
DATA_SHA = "85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f"
MANIFEST = "sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json"
MANIFEST_SHA = "393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04"
MLP16 = "titantpp_history_mlp_width16"
MODELS = (MLP16, "rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head")
SEEDS = (42, 52, 62)
COUNTS = {"validation": 86285, "test": 88019}
TAIL_THRESHOLD = 187.0
require, path, sha, read, compact, bind = parent.require, parent.path, parent.sha, parent.read, parent.compact, parent.bind


def populations_and_rows(bindings):
    import numpy as np
    import polars as pl
    from paper.scripts import run_titantpp_shared_history_raw9 as frozen
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader

    bind(bindings, DATA, DATA_SHA)
    bind(bindings, MANIFEST, MANIFEST_SHA)
    for source in ("paper/scripts/run_titantpp_shared_history_raw9.py",
                   "paper/scripts/run_titantpp_mlp_cap36.py",
                   "paper/scripts/count_aware_tpp_backbone/core.py",
                   "paper/scripts/run_taxi_quantity_interface_ablation.py",
                   "data_loader/event_seq_data_module.py"):
        bind(bindings, source)
    full = pl.read_parquet(path(DATA)).sort(["oper_part_no", "seq"])
    raw_rows = {(str(row["oper_part_no"]), int(row["seq"])): row for row in full.iter_rows(named=True)}
    populations, expected = {}, {}
    for split in COUNTS:
        frame = full if split == "test" else full.filter(pl.col("chronological_split").is_in(["train", "validation"]))
        dataset = make_loader(prepare_count_frame(frame), target_split=split, batch_size=128,
                              lookback_weeks=520, max_seq_len=256, shuffle=False, generator=None).dataset
        _, identity = frozen.population_identity(dataset, split)
        require(identity["target_count"] == COUNTS[split], "Intermittent population count differs")
        durations = np.asarray([dataset.dt_lists[p][end + 1] for p, end in dataset.index], dtype="<f8")
        identity["recorded_time_sha256"] = hashlib.sha256(
            b"frozen_lognormal_target_dt_v1\0" + durations.tobytes()).hexdigest()
        identity["component_time_sha256"] = hashlib.sha256(
            b"titantpp_components_recorded_target_time_v1\0" + split.encode() + b"\0" + durations.tobytes()).hexdigest()
        populations[split] = identity
        expected[split] = []
        for index, (part, end) in enumerate(dataset.index):
            entity, seq = str(dataset.parts[part]), int(dataset.seq_lists[part][end + 1])
            source = raw_rows[entity, seq]
            seqs = np.asarray(dataset.seq_lists[part], dtype=np.int64)
            left = max(int(np.searchsorted(seqs, seqs[end] - (dataset.W - 1), side="left")),
                       end + 2 - dataset.max_len)
            history = np.asarray(dataset.val_lists[part][left:end + 1], dtype=np.float64)
            expected[split].append({"target_index": index, "entity_id": entity, "seq": seq, "split": split,
                "raw_quantity": float(dataset.val_lists[part][end + 1]),
                "recorded_gap": float(dataset.dt_lists[part][end + 1]), "history_length": len(history),
                "last_quantity": float(history[-1]), "mean_quantity": float(history.mean()),
                "recorded_date": str(source["demand_dt"]), "time_bucket": None,
                "site_id": entity.split("::")[0]})
    return populations, expected


def recompute_metrics(frame):
    import numpy as np
    true = frame["raw_quantity"].to_numpy().astype(np.float64)
    pred = frame["predicted_raw_quantity"].to_numpy().astype(np.float64)
    time = frame["time_nll"].to_numpy().astype(np.float64)
    require(np.isfinite(true).all() and np.isfinite(pred).all() and np.isfinite(time).all(),
            "Nonfinite stored predictions")
    result = {}
    for name, mask in (("overall", np.ones(len(true), dtype=bool)), ("tail", true > TAIL_THRESHOLD)):
        error = pred[mask] - true[mask]
        require(len(error) > 0, "Empty expected Intermittent cohort")
        result[name] = {"count": len(error), "qty_rmse": float(np.sqrt(np.mean(error * error))),
                       "qty_mae": float(np.mean(np.abs(error))), "time_nll": float(np.mean(time[mask])),
                       "qty_bias": float(np.mean(error))}
    return result


def selected_history(history):
    require(bool(history) and [r["epoch"] for r in history] == sorted(set(r["epoch"] for r in history)),
            "Unordered or duplicate history epoch")
    finite = [r for r in history if type(r["val_qty_rmse"]) in (int, float) and math.isfinite(r["val_qty_rmse"])]
    require(bool(finite), "No finite Validation RMSE")
    return min(finite, key=lambda r: r["val_qty_rmse"])


def checkpoint_lineage(receipt, bindings):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    checkpoint = bind(bindings, receipt["checkpoint_path"], receipt["checkpoint_file_sha256"])
    history_path, summary_path = bind(bindings, checkpoint.parent / "history.json"), bind(bindings, checkpoint.parent / "summary.json")
    history, summary = read(history_path)["history"], read(summary_path)
    selected = selected_history(history)
    require(selected["epoch"] == receipt["selected_epoch"] == summary["best_epoch"], "Selection epoch differs")
    for row in history:
        require(math.isclose(row["train_joint_objective"], row["train_quantity_loss"] + row["train_time_nll"],
                             rel_tol=1e-6, abs_tol=1e-6), "Training alpha1 equation differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(payload["best_epoch"] == receipt["selected_epoch"] and payload["seed"] == receipt["seed"]
            and payload["backbone"] == receipt["model"] and payload["variant"] == "count_only_log_regression",
            "Checkpoint metadata differs")
    require(canonical_state_dict_sha256(payload["model_state_dict"]) == receipt["state_tensor_sha256"],
            "Checkpoint tensor identity differs")
    args = payload["resume_identity"]["arguments"]
    require(args["data_sha256"] == DATA_SHA and args["split_manifest_sha256"] == MANIFEST_SHA
            and args["lambda_log_qty"] == 1.0 and args["lambda_tail"] == 0.0,
            "Original uncapped data/log-MSE contract differs")
    time_head = payload["encoder_config"].get("time_head", payload["interface_meta"].get("time_head"))
    require(time_head["mode"] == "heteroscedastic_lognormal_duration"
            and time_head["observation_likelihood"] == {"mode": "positive_integer_round_clamp_v1", "top_code": None, "unit": "week"}
            and time_head["time_scale"] == 3.0, "Common time head differs")
    return {"checkpoint_path": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
            "state_tensor_sha256": receipt["state_tensor_sha256"],
            "history_path": str(history_path.relative_to(ROOT)), "history_sha256": sha(history_path),
            "source_closure_sha256": receipt["source_closure_sha256"], "time_head": time_head}, selected


def audit_prediction_receipt(receipt_path, bindings, populations, expected, common_truth):
    import polars as pl
    receipt_path = bind(bindings, receipt_path)
    receipt = read(receipt_path)
    split, model, seed = receipt["split"], receipt["model"], receipt["seed"]
    require(receipt["status"] == "complete" and receipt["dataset"] == DATASET
            and model in MODELS and seed in SEEDS and split in COUNTS
            and receipt["full_population"] is True and receipt["prediction_rows"] == COUNTS[split]
            and receipt["expected_target_count"] == COUNTS[split] and receipt["max_batches"] is None
            and receipt["data_file_sha256"] == DATA_SHA, "Foreign/incomplete stored receipt")
    lineage, selected = checkpoint_lineage(receipt, bindings)
    parts, frames = [], []
    for item in receipt["parts"]:
        file = bind(bindings, receipt_path.parent / item["path"], item["sha256"])
        frame = pl.read_parquet(file)
        require(frame.height == item["rows"], "Shard row count differs")
        frames.append(frame)
        parts.append({"path": str(file.relative_to(ROOT)), "sha256": item["sha256"], "rows": frame.height})
    frame = pl.concat(frames)
    require(frame.height == COUNTS[split], "Incomplete stored shards")
    fields = receipt["truth_digest_fields"]
    target_digest, truth_digest = hashlib.sha256(), hashlib.sha256()
    truth_rows = []
    for index, row in enumerate(frame.iter_rows(named=True)):
        target_digest.update((row["target_id"] + "\n").encode())
        truth_digest.update(compact([row[key] for key in fields]) + b"\n")
        truth_rows.append([row[key] for key in fields])
        for key, value in expected[split][index].items():
            if key == "mean_quantity":
                require(math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-12), "History mean differs")
            else:
                require(row[key] == value, "Full original data row mismatch: " + key)
        target_id = hashlib.sha256(compact([DATASET, DATA_SHA, row["entity_id"], row["seq"], split, row["recorded_date"]])).hexdigest()
        require(row["target_id"] == target_id, "Legacy target ID differs")
    require(target_digest.hexdigest() == receipt["target_identity_sha256"]
            and truth_digest.hexdigest() == receipt["truth_sha256"], "Legacy truth digest differs")
    if split in common_truth:
        require(truth_rows == common_truth[split], "Different stored target/truth/time/history/order")
    else:
        common_truth[split] = truth_rows
    metrics = recompute_metrics(frame)
    if split == "validation":
        for key in ("qty_rmse", "qty_mae", "time_nll"):
            require(math.isclose(metrics["overall"][key], selected["val_" + key], rel_tol=1e-5, abs_tol=1e-5),
                    "Stored Validation overall metric differs from selected history")
    return {"dataset": DATASET, "model": model, "seed": seed, "split": split,
        "selected_epoch": receipt["selected_epoch"], "metrics": metrics,
        "selection_rule": "first strict finite minimum fullValidation raw quantity RMSE; MAE uses identical checkpoint",
        "tail_threshold": TAIL_THRESHOLD, "tail_rule": "raw quantity > Train-frozen187",
        "time_metric": "recorded_positive_integer_time_nll", "data_variant": "original_uncapped",
        "quantity_capping_applied": False, "time_alpha": 1.0, "quantity_loss": "log_mse",
        "training_host": "5080", "evaluation_runtime": receipt["runtime"],
        "path": str(receipt_path.relative_to(ROOT)), "sha256": sha(receipt_path), **lineage,
        "prediction_parts": parts, "individual_predictions_available": True,
        "target_identity": {key: populations[split][key] for key in
                            ("target_count", "target_identity_sha256", "target_quantity_sha256")},
        "recorded_time_sha256": populations[split]["recorded_time_sha256"],
        "legacy_target_identity_sha256": receipt["target_identity_sha256"],
        "legacy_truth_sha256": receipt["truth_sha256"], "legacy_truth_digest_fields": fields,
        "retrained": False, "reevaluated": False,
        "checks": {"stored_rows_match_original_uncapped_data": True,
                   "checkpoint_binary_tensor_epoch_history_verified": True,
                   "first_finite_validation_minimum": True, "alpha1_all_training_epochs": True}}, receipt, selected


def audit_external_validation(test_row, receipt, selected, bindings, populations, expected):
    replay_path = bind(bindings, path(test_row["checkpoint_path"]).parent / "endpoint_replays.json")
    replay = read(replay_path)
    endpoint = replay["selected"]
    require(replay["best_epoch"] == test_row["selected_epoch"]
            and endpoint["state_sha256"] == test_row["state_tensor_sha256"]
            and endpoint["count"] == COUNTS["validation"]
            and endpoint["evaluation_scope"] == "validation_only" and endpoint["held_out_test_evaluated"] is False,
            "Foreign selected Validation endpoint")
    require(endpoint["quantity_boundaries"] == [2, 31, 46, 187]
            and endpoint["tail"]["count"] == sum(r["raw_quantity"] > TAIL_THRESHOLD for r in expected["validation"]),
            "External Validation tail contract differs")
    metrics = {"overall": {key: endpoint[key] for key in ("count", "qty_rmse", "qty_mae", "time_nll")},
               "tail": {key: endpoint["tail"][key] for key in ("count", "qty_rmse", "qty_mae", "time_nll")}}
    for key in ("qty_rmse", "qty_mae", "time_nll"):
        require(math.isclose(metrics["overall"][key], selected["val_" + key], rel_tol=1e-5, abs_tol=1e-5),
                "External selected Validation/history differs")
    row = {key: copy.deepcopy(value) for key, value in test_row.items()
           if key not in ("prediction_parts", "legacy_target_identity_sha256", "legacy_truth_sha256", "legacy_truth_digest_fields")}
    row.update(split="validation", metrics=metrics, path=str(replay_path.relative_to(ROOT)), sha256=sha(replay_path),
               prediction_parts=[], individual_predictions_available=False,
               target_identity={key: populations["validation"][key] for key in
                                ("target_count", "target_identity_sha256", "target_quantity_sha256")},
               recorded_time_sha256=populations["validation"]["recorded_time_sha256"],
               checks={"checkpoint_binary_tensor_epoch_history_verified": True,
                       "first_finite_validation_minimum": True, "alpha1_all_training_epochs": True,
                       "selected_endpoint_overall_metrics_match_history": True,
                       "population_and_tail_count_match_original_data": True,
                       "individual_validation_rows_verified": False},
               limitation="Historical selected endpoint aggregates only; individual Validation predictions/truth order unavailable and not newly inferred.")
    return row


def main():
    targets = [REPORT / name for name in ("REFERENCES.json", "REFERENCES.md", "REFERENCE_QA.json")]
    require(not any(file.exists() for file in targets), "Existing registry output; no overwrite")
    bindings = {}
    populations, expected = populations_and_rows(bindings)
    rows, common_truth = [], {}
    for model in MODELS:
        for seed in SEEDS:
            if model == MLP16:
                base = f"reports/titantpp_completed_validation_test_20261004_v1/width16/runs"
                test_path = f"{base}/test/{DATASET}__{model}__seed{seed}/receipt.json"
            else:
                test_path = f"reports/titantpp_legacy_evaluation_20261003_v1/runs/test/attempt1/{DATASET}__{model}__seed{seed}/receipt.json"
            test, receipt, selected = audit_prediction_receipt(test_path, bindings, populations, expected, common_truth)
            rows.append(test)
            if model == MLP16:
                validation, _, _ = audit_prediction_receipt(test_path.replace("/test/", "/validation/"), bindings,
                                                          populations, expected, common_truth)
            else:
                validation = audit_external_validation(test, receipt, selected, bindings, populations, expected)
            rows.append(validation)
    require(len(rows) == 42 and {(r["model"], r["seed"], r["split"]) for r in rows}
            == {(m, s, sp) for m in MODELS for s in SEEDS for sp in COUNTS}, "Incomplete stored reference matrix")
    legacy = {}
    for split in COUNTS:
        first = next(r for r in rows if r["split"] == split and r["individual_predictions_available"])
        legacy[split] = {"count": COUNTS[split], "target_identity_sha256": first["legacy_target_identity_sha256"],
                         "truth_sha256": first["legacy_truth_sha256"], "truth_digest_fields": first["legacy_truth_digest_fields"]}
    data = {"path": DATA, "sha256": DATA_SHA, "split_manifest_path": MANIFEST, "split_manifest_sha256": MANIFEST_SHA}
    descriptor = {"data": data, "full_data": {"path": DATA, "sha256": DATA_SHA},
                  "full_split_manifest": {"path": MANIFEST, "sha256": MANIFEST_SHA},
                  "populations": populations, "legacy_populations": legacy, "legacy_test_population": legacy["test"]}
    aggregates = {split: {model: {cohort: {metric: {
        "mean": statistics.mean(r["metrics"][cohort][metric] for r in rows if r["model"] == model and r["split"] == split),
        "sample_sd_ddof1": statistics.stdev(r["metrics"][cohort][metric] for r in rows if r["model"] == model and r["split"] == split)}
        for metric in ("qty_rmse", "qty_mae", "time_nll")} for cohort in ("overall", "tail")} for model in MODELS} for split in COUNTS}
    bind(bindings, __file__)
    bind(bindings, parent.__file__)
    result = {"schema": "shared_inter3_stored_validation_test_references_v1", "status": "verified_stored_references",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "data_variant": "original_uncapped",
        "reference_count": 42, "reference_count_by_split": {"validation": 21, "test": 21},
        "baseline_reference_count": 6, "external_reference_count": 36, "seed_count": 3, "models": list(MODELS),
        "seeds": list(SEEDS), "datasets": {DATASET: descriptor}, "references": rows, "aggregates": aggregates,
        "source_bindings": bindings, "verified_saved_prediction_rows": 21 * COUNTS["test"] + 3 * COUNTS["validation"],
        "individual_prediction_reference_count": 24, "historical_aggregate_reference_count": 18,
        "new_inference_calls": 0, "new_training_calls": 0, "remote_calls": 0,
        "existing_results_modified": False, "candidate_results_included": False,
        "limitations": ["External Validation18 have existing selected endpoint aggregates only; individual row/order audit unavailable.",
                        "Historical common-head adapted encoders, not all native-paper heads or tuning.",
                        "Exploratory reuse of previously observed Test; not a new independent holdout or significance claim."]}
    lines = ["# 원본 Intermittent 저장 Validation·Test 참조 검증", "",
        "MLP16·외부6종 × 3seed × 2split, 총42개 기존 결과다. 재학습·모델 재추론·원격 호출은 없다.",
        "Test21과 MLP16 Validation3은 저장 예측의 대상·수량·시간·이력·순서를 원본 데이터와 대조하고 여섯 지표를 재집계했다.",
        "외부 Validation18은 기존 selected endpoint 집계다. checkpoint·state·첫 유한 최소 Validation RMSE epoch·전체 지표·모집단·tail 개수를 확인했으나 개별 예측/행 순서는 검증하지 못했다.",
        "각 모델의 동일 Validation-selected checkpoint를 유지했으며 Test로 재선택하지 않았다.", "",
        "| split | 모델 | RMSE 평균±표본SD | MAE 평균±표본SD |", "|---|---|---:|---:|"]
    for split in COUNTS:
        for model in MODELS:
            metric = aggregates[split][model]["overall"]
            lines.append(f"| {split} | {model} | {metric['qty_rmse']['mean']:.6f}±{metric['qty_rmse']['sample_sd_ddof1']:.6f} | "
                         f"{metric['qty_mae']['mean']:.6f}±{metric['qty_mae']['sample_sd_ddof1']:.6f} |")
    lines += ["", "큰수량은 Train 고정 q>187이다. 전체·큰수량 RMSE·MAE·recorded positive-integer TimeNLL과 모든 seed 원값은 JSON에 보존했다.",
              "표본SD는 ddof=1이다. 기존 Runtime 차이와 외부 공통 head 적응 범위가 남는다."]
    REPORT.mkdir(parents=True, exist_ok=True)
    for file, value in ((targets[0], result), (targets[1], "\n".join(lines) + "\n")):
        with file.open("x", encoding="utf-8") as stream:
            stream.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    qa = {"status": "PASS", "reference_count": 42, "source_bindings_verified": len(bindings),
        "individual_prediction_reference_count": 24, "aggregate_only_external_validation_count": 18,
        "verified_saved_prediction_rows": result["verified_saved_prediction_rows"],
        "checks": {"stored_prediction_rows_original_data_truth_time_history_order": True,
                   "all_checkpoints_binary_tensor_epoch_history": True, "earliest_finite_validation_minimum": True,
                   "alpha1_joint_training_equation": True, "six_metrics_all_seeds": True,
                   "legacy_modern_population_row_bridge": True},
        "limitations": result["limitations"], "output_files": {str(file.relative_to(ROOT)): sha(file) for file in targets[:2]},
        "new_inference_calls": 0, "new_training_calls": 0, "remote_calls": 0}
    with targets[2].open("x", encoding="utf-8") as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"files": {str(file.relative_to(ROOT)): sha(file) for file in targets}, "populations": populations}, indent=2))


if __name__ == "__main__":
    main()
