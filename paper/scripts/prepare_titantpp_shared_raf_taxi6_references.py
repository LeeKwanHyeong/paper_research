"""Audit existing original RAF/Taxi Test references without model inference.

Only the newly authorized reference registry and its audit are written.
Existing prediction shards, checkpoint binaries and source artifacts are read-only.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/titantpp_shared_raf_taxi6_test_20261020_v1"
RAF_REFERENCE = ROOT / "reports/titantpp_shared_raw_raf3_test_20261020_v1/REFERENCES.json"
RAF_REFERENCE_SHA = "c365cf7a6d6ddf8edd0c1e233c307a89779923b0b8b26efdbc55fca4ae3b6d18"
TAXI_DATA = "sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet"
TAXI_DATA_SHA = "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46"
TAXI_MANIFEST = "sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json"
TAXI_MANIFEST_SHA = "4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85"
TAXI = "yellow_trip_hourly"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    result = (ROOT / value).resolve()
    require(result.is_relative_to(ROOT), "Artifact escapes project")
    return result


def sha(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(file):
    return json.loads(Path(file).read_text())


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def bind(bindings, file, expected=None):
    file = path(file)
    value = sha(file)
    require(expected is None or value == expected, "Artifact SHA differs: " + str(file))
    key = str(file.relative_to(ROOT))
    require(key not in bindings or bindings[key] == value, "Conflicting SHA binding")
    bindings[key] = value
    return file


def taxi_populations(bindings):
    import numpy as np
    import polars as pl
    from paper.scripts import run_titantpp_shared_history_raw9 as frozen
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader

    bind(bindings, TAXI_DATA, TAXI_DATA_SHA)
    bind(bindings, TAXI_MANIFEST, TAXI_MANIFEST_SHA)
    for source in ("paper/scripts/run_titantpp_shared_history_raw9.py",
                   "paper/scripts/run_titantpp_mlp_cap36.py",
                   "paper/scripts/count_aware_tpp_backbone/core.py",
                   "paper/scripts/run_taxi_quantity_interface_ablation.py",
                   "data_loader/event_seq_data_module.py"):
        bind(bindings, source)
    full = pl.read_parquet(path(TAXI_DATA)).sort(["oper_part_no", "seq"])
    populations, test_dataset = {}, None
    for split in ("validation", "test"):
        frame = full if split == "test" else full.filter(pl.col("chronological_split").is_in(["train", "validation"]))
        dataset = make_loader(prepare_count_frame(frame), target_split=split, batch_size=128,
                              lookback_weeks=168, max_seq_len=256, shuffle=False, generator=None).dataset
        _, identity = frozen.population_identity(dataset, split)
        durations = np.asarray([dataset.dt_lists[p][end + 1] for p, end in dataset.index], dtype="<f8")
        identity["recorded_time_sha256"] = hashlib.sha256(
            b"frozen_lognormal_target_dt_v1\0" + durations.tobytes()).hexdigest()
        identity["component_time_sha256"] = hashlib.sha256(
            b"titantpp_components_recorded_target_time_v1\0" + split.encode() + b"\0" + durations.tobytes()).hexdigest()
        populations[split] = identity
        if split == "test":
            test_dataset = dataset
    require(populations["validation"]["target_count"] == 8268
            and populations["validation"]["target_identity_sha256"] == "19493e9265c6b733b03a08fafc391a86abf2f46b4efb0fc140c8d48534e407fb"
            and populations["validation"]["target_quantity_sha256"] == "acaa58691a7370d2708c092da777a3d1c14b511d84bbb36b6ed9fb64b2e723cd"
            and populations["test"]["target_count"] == 8327, "Taxi population differs")
    raw_rows = {(str(row["oper_part_no"]), int(row["seq"])): row for row in full.iter_rows(named=True)}
    expected = []
    for index, (part, end) in enumerate(test_dataset.index):
        entity, seq = str(test_dataset.parts[part]), int(test_dataset.seq_lists[part][end + 1])
        row = raw_rows[entity, seq]
        seqs = np.asarray(test_dataset.seq_lists[part], dtype=np.int64)
        left = int(np.searchsorted(seqs, seqs[end] - (test_dataset.W - 1), side="left"))
        left = max(left, end + 2 - test_dataset.max_len)
        history = np.asarray(test_dataset.val_lists[part][left:end + 1], dtype=np.float64)
        expected.append({"target_index": index, "entity_id": entity, "seq": seq, "split": "test",
                         "raw_quantity": float(test_dataset.val_lists[part][end + 1]),
                         "recorded_gap": float(test_dataset.dt_lists[part][end + 1]),
                         "history_length": len(history), "last_quantity": float(history[-1]),
                         "mean_quantity": float(history.mean()), "recorded_date": str(row["demand_dt"]),
                         "time_bucket": str(row["time_bucket"]), "site_id": None})
    return populations, expected


def recompute_metrics(frame):
    import numpy as np
    result = {}
    true = frame["raw_quantity"].to_numpy().astype(np.float64)
    pred = frame["predicted_raw_quantity"].to_numpy().astype(np.float64)
    time = frame["time_nll"].to_numpy().astype(np.float64)
    require(np.isfinite(true).all() and np.isfinite(pred).all() and np.isfinite(time).all(),
            "Nonfinite stored prediction")
    for name, mask in (("overall", np.ones(len(true), dtype=bool)), ("tail", true > 3449)):
        error = pred[mask] - true[mask]
        require(len(error) > 0, "Empty expected Taxi cohort")
        result[name] = {"count": len(error), "qty_rmse": float(np.sqrt(np.mean(error * error))),
                        "qty_mae": float(np.mean(np.abs(error))), "time_nll": float(np.mean(time[mask])),
                        "qty_bias": float(np.mean(error))}
    return result


def audit_taxi(template, bindings, populations, expected, common_truth):
    import numpy as np
    import polars as pl
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    receipt_path = bind(bindings, template["path"].replace("raf_spare_parts", TAXI))
    receipt = read(receipt_path)
    model, seed = template["model"], template["seed"]
    require(receipt["status"] == "complete" and receipt["dataset"] == TAXI
            and receipt["model"] == model and receipt["seed"] == seed and receipt["split"] == "test"
            and receipt["full_population"] is True and receipt["prediction_rows"] == 8327
            and receipt["expected_target_count"] == 8327 and receipt["max_batches"] is None
            and receipt["data_file_sha256"] == TAXI_DATA_SHA, "Foreign/incomplete Taxi receipt")
    checkpoint = bind(bindings, receipt["checkpoint_path"], receipt["checkpoint_file_sha256"])
    history_path = bind(bindings, checkpoint.parent / "history.json")
    summary_path = bind(bindings, checkpoint.parent / "summary.json")
    history, summary = read(history_path)["history"], read(summary_path)
    finite = [row for row in history if math.isfinite(row["val_qty_rmse"])]
    selected = min(finite, key=lambda row: row["val_qty_rmse"])
    require(selected["epoch"] == receipt["selected_epoch"] == summary["best_epoch"], "Selection epoch differs")
    if model == "titantpp_history_mlp_width16":
        require(receipt["selected_epoch"] == {42: 27, 52: 82, 62: 94}[seed], "MLP16 epoch differs")
    for row in history:
        require(math.isclose(row["train_joint_objective"], row["train_quantity_loss"] + row["train_time_nll"],
                             rel_tol=1e-6, abs_tol=1e-6), "Training alpha1 equation differs")
    # Locally owned, SHA-bound historical checkpoints; tensors only, no model construction/inference.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(payload["best_epoch"] == receipt["selected_epoch"] and payload["seed"] == seed
            and payload["backbone"] == model and payload["variant"] == "count_only_log_regression",
            "Checkpoint metadata differs")
    require(canonical_state_dict_sha256(payload["model_state_dict"]) == receipt["state_tensor_sha256"],
            "Checkpoint tensor identity differs")
    time_head = payload["encoder_config"].get("time_head", payload["interface_meta"].get("time_head"))
    require(time_head["mode"] == "heteroscedastic_lognormal_duration", "Time head differs")
    parts, frames = [], []
    for item in receipt["parts"]:
        file = bind(bindings, receipt_path.parent / item["path"], item["sha256"])
        frame = pl.read_parquet(file)
        require(frame.height == item["rows"], "Shard row count differs")
        frames.append(frame)
        parts.append({"path": str(file.relative_to(ROOT)), "sha256": item["sha256"], "rows": frame.height})
    frame = pl.concat(frames)
    require(frame.height == 8327, "Incomplete Taxi shards")
    truth_fields = receipt["truth_digest_fields"]
    target_digest, truth_digest = hashlib.sha256(), hashlib.sha256()
    truth_rows = []
    for index, row in enumerate(frame.iter_rows(named=True)):
        target_digest.update((row["target_id"] + "\n").encode())
        truth_digest.update(compact([row[key] for key in truth_fields]) + b"\n")
        truth_rows.append([row[key] for key in truth_fields])
        for key, value in expected[index].items():
            if key == "mean_quantity":
                require(math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-12), "History mean differs")
            else:
                require(row[key] == value, "Full-data row mismatch: " + key)
        target_id = hashlib.sha256(compact([TAXI, TAXI_DATA_SHA, row["entity_id"], row["seq"],
                                            "test", row["recorded_date"]])).hexdigest()
        require(row["target_id"] == target_id, "Legacy target ID differs")
    require(target_digest.hexdigest() == receipt["target_identity_sha256"]
            and truth_digest.hexdigest() == receipt["truth_sha256"], "Legacy truth digest differs")
    if common_truth is not None:
        require(truth_rows == common_truth, "Different reference target/truth/history/order")
    metrics = recompute_metrics(frame)
    row = {"dataset": TAXI, "model": model, "seed": seed, "split": "test",
           "selected_epoch": receipt["selected_epoch"], "selection_rule": template["selection_rule"],
           "metrics": metrics, "tail_threshold": 3449.0, "tail_rule": "raw quantity > Train-frozen3449",
           "time_metric": "recorded_positive_integer_time_nll", "data_variant": "original_uncapped",
           "quantity_capping_applied": False, "time_alpha": 1.0, "quantity_loss": "log_mse",
           "time_head": time_head, "training_host": "5080", "evaluation_runtime": receipt["runtime"],
           "path": str(receipt_path.relative_to(ROOT)), "sha256": sha(receipt_path),
           "checkpoint_path": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
           "state_tensor_sha256": receipt["state_tensor_sha256"],
           "history_path": str(history_path.relative_to(ROOT)), "history_sha256": sha(history_path),
           "source_closure_sha256": receipt["source_closure_sha256"], "prediction_parts": parts,
           "target_identity": {key: populations["test"][key] for key in
                               ("target_count", "target_identity_sha256", "target_quantity_sha256")},
           "recorded_time_sha256": populations["test"]["recorded_time_sha256"],
           "legacy_target_identity_sha256": receipt["target_identity_sha256"],
           "legacy_truth_sha256": receipt["truth_sha256"], "legacy_truth_digest_fields": truth_fields,
           "retrained": False, "reevaluated": False,
           "checks": {"receipt_parts_checkpoint_history_sha_verified": True,
                      "test_rows_match_full_uncapped_data": True,
                      "all21_exact_same_target_order_quantity_time_and_history_truth": True,
                      "validation_first_minimum_checkpoint_preserved": True,
                      "alpha1_joint_equation_all_training_epochs_verified": True}}
    return row, truth_rows


def main():
    targets = [REPORT / name for name in ("REFERENCES.json", "REFERENCES.md", "REFERENCE_QA.json")]
    require(not any(file.exists() for file in targets), "Existing registry output; no overwrite")
    bindings = {}
    bind(bindings, RAF_REFERENCE, RAF_REFERENCE_SHA)
    raf = read(RAF_REFERENCE)
    require(raf["reference_count"] == 21 and len(raf["references"]) == 21, "RAF reference matrix differs")
    for file, digest in raf["source_bindings"].items():
        bind(bindings, file, digest)
    populations, expected = taxi_populations(bindings)
    taxi, common_truth = [], None
    for template in raf["references"]:
        row, common_truth = audit_taxi(template, bindings, populations, expected, common_truth)
        taxi.append(row)
    models, seeds = raf["models"], [42, 52, 62]
    require({(row["model"], row["seed"]) for row in taxi} == {(m, s) for m in models for s in seeds},
            "Taxi matrix incomplete")
    aggregates = {"raf_spare_parts": copy.deepcopy(raf["aggregates"]), TAXI: {}}
    for model in models:
        rows = [row for row in taxi if row["model"] == model]
        aggregates[TAXI][model] = {
            cohort: {metric: {"mean": statistics.mean(row["metrics"][cohort][metric] for row in rows),
                              "sample_sd": statistics.stdev(row["metrics"][cohort][metric] for row in rows),
                              "ddof": 1}
                     for metric in ("qty_rmse", "qty_mae", "time_nll")}
            for cohort in ("overall", "tail")}
    data = {"path": TAXI_DATA, "sha256": TAXI_DATA_SHA, "split_manifest_path": TAXI_MANIFEST,
            "split_manifest_sha256": TAXI_MANIFEST_SHA}
    legacy = {"count": 8327, "target_identity_sha256": taxi[0]["legacy_target_identity_sha256"],
              "truth_sha256": taxi[0]["legacy_truth_sha256"],
              "truth_digest_fields": taxi[0]["legacy_truth_digest_fields"]}
    bind(bindings, __file__)
    result = {"schema": "shared_raf_taxi6_stored_test_references_v1",
              "status": "verified_stored_test_references", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "data_variant": "original_uncapped", "reference_count": 42, "baseline_reference_count": 6,
              "external_reference_count": 36, "seed_count": 3, "models": models, "seeds": seeds,
              "datasets": {"raf_spare_parts": {key: copy.deepcopy(raf[key]) for key in
                                             ("data", "populations", "legacy_test_population")},
                           TAXI: {"data": data, "populations": populations, "legacy_test_population": legacy}},
              "references": copy.deepcopy(raf["references"]) + taxi, "aggregates": aggregates,
              "source_bindings": bindings, "verified_saved_prediction_rows": 109746 + 21 * 8327,
              "raf_references_reused_unchanged": True, "new_inference_calls": 0, "new_training_calls": 0,
              "remote_calls": 0, "existing_results_modified": False, "intermittent_test_read": False,
              "candidate_results_included": False, "limitations": raf["limitations"]}
    lines = ["# 원본 RAF·Taxi 저장 Test 참조 검증", "",
             "기존 MLP16·외부6종 × 3seed × 2데이터셋, 총42개 저장 결과다. 재학습·모델 재추론·원격 호출은 없다.",
             "RAF21개는 기존 검증 레코드와 원본 SHA를 그대로 재사용했다. Taxi21개는 저장 예측에서 여섯 지표를 재집계했다.",
             "Checkpoint는 각 학습의 첫 유한 최소 Validation RMSE epoch로 고정했고, Test로 재선택하지 않았다.",
             "", "| 데이터 | 모델 | RMSE 평균±표본SD | MAE 평균±표본SD |", "|---|---|---:|---:|"]
    for dataset in ("raf_spare_parts", TAXI):
        for model in models:
            metrics = aggregates[dataset][model]["overall"]
            lines.append(f"| {dataset} | {model} | {metrics['qty_rmse']['mean']:.6f}±{metrics['qty_rmse']['sample_sd']:.6f} | "
                         f"{metrics['qty_mae']['mean']:.6f}±{metrics['qty_mae']['sample_sd']:.6f} |")
    lines += ["", "전체·큰수량(q>RAF200/Taxi3449) RMSE·MAE·시간NLL, 모든 seed 원값은 REFERENCES.json에 보존했다.",
              "표본SD는 ddof=1이다. 데이터셋 간 원시 오류는 평균내지 않았다.",
              "Legacy ID와 현대 Window population ID는 다른 계약이다. 개별 개체·seq·수량·간격·이력·순서를 대조해 연결했다.",
              "MLP16 seed42 CPU와 나머지 기존5080, 새Shared5090의 Runtime 차이는 남는다.",
              "외부 비교군은 공통 수량·시간 head를 맞춘 결과다. 반복 Test 참고 탐색이며 독립 일반화·통계적 유의성의 증거가 아니다."]
    REPORT.mkdir(parents=True, exist_ok=True)
    with targets[0].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    with targets[1].open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    qa = {"status": "PASS", "reference_count": 42, "source_bindings_verified": len(bindings),
          "raf_reused_reference_sha256": RAF_REFERENCE_SHA, "taxi_rows_reaggregated": 21 * 8327,
          "checks": {"exact_target_truth_time_history_order": True, "checkpoint_binary_tensor_epoch": True,
                     "first_finite_validation_minimum": True, "alpha1_joint_training_equation": True,
                     "six_metrics_all_cohorts_seeds": True, "legacy_modern_population_row_bridge": True},
          "output_files": {str(file.relative_to(ROOT)): sha(file) for file in targets[:2]},
          "new_inference_calls": 0, "remote_calls": 0, "intermittent_test_read": False}
    with targets[2].open("x", encoding="utf-8") as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"files": {str(file.relative_to(ROOT)): sha(file) for file in targets},
                      "taxi_populations": populations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
