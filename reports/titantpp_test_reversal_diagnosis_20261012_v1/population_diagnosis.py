#!/usr/bin/env python3
"""Descriptive frozen-target population audit; no checkpoint or model inference.

Uses the original frozen loader index, preserves all eligible targets, and checks
the original population/duration identities before reporting distributions.
Outputs are confined to this report directory; frozen artifacts are read-only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import polars as pl

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / "search_artifacts/titantpp_history_instacart_test_20261012_v1"
OUT = Path(__file__).resolve().parent
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(o: object) -> bytes:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def array_sha(prefix: bytes, a: np.ndarray) -> str:
    return hashlib.sha256(prefix + a.astype("<f8").tobytes()).hexdigest()


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    assert len(a) and np.isfinite(a).all()
    return {"count": len(a), "mean": float(a.mean()), "population_sd": float(a.std()),
            "minimum": float(a.min()), "maximum": float(a.max()),
            "quantiles": {str(q): float(np.quantile(a, q)) for q in (0.01, .1, .25, .5, .75, .9, .95, .99)},
            "zero_count": int((a == 0).sum()), "one_count": int((a == 1).sum())}


def histogram(a: np.ndarray, boundaries: list[float]) -> dict:
    bins = np.searchsorted(boundaries, a, side="left")
    counts = np.bincount(bins, minlength=len(boundaries) + 1)
    return {"boundaries": boundaries, "rule": "left-closed to boundary: first <= b0; then (b[i-1],b[i]]; final > last",
            "counts": counts.tolist(), "proportions": (counts / len(a)).tolist()}


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def population_identity(dataset, split: str) -> tuple[dict, dict]:
    index = np.asarray(dataset.index, dtype=np.int64).reshape(-1, 2)
    part, pos = index[:, 0], index[:, 1] + 1
    seq = np.fromiter((dataset.seq_lists[p][t] for p, t in zip(part, pos)), dtype=np.int64)
    q = np.fromiter((dataset.val_lists[p][t] for p, t in zip(part, pos)), dtype=np.float64)
    gap = np.fromiter((dataset.dt_lists[p][t] for p, t in zip(part, pos)), dtype=np.float64)
    assert all(dataset.split_lists[p][t] == split for p, t in zip(part, pos))
    h = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode() + b"\0")
    h.update(json.dumps([str(p) for p in dataset.parts], ensure_ascii=False, separators=(",", ":")).encode())
    for a in (part, pos, seq):
        h.update(a.astype("<i8").tobytes())
    result = {"target_count": len(q), "target_identity_sha256": h.hexdigest(),
              "target_quantity_sha256": array_sha(b"hard_lmm_target_quantity_v1\0", q)}
    return result, {"part": part, "pos": pos, "seq": seq, "q": q, "gap": gap}


def target_statistics(ds: str, split: str, frame: pl.DataFrame, dataset, definition: dict, data_sha: str,
                      native: dict | None) -> dict:
    identity, arrays = population_identity(dataset, split)
    part, pos, seq, q, gap = (arrays[k] for k in ("part", "pos", "seq", "q", "gap"))
    # Native loader stores quantities/durations as float32, as used by metrics.
    assert np.equal(q, q.astype(np.float32).astype(np.float64)).all()
    if split != "test":
        assert identity == definition["inherited_data_identity"]["populations"][split]
        gap_sha = array_sha(b"frozen_lognormal_target_dt_v1\0", gap)
        assert gap_sha == definition["time_statistics"][f"expected_{split}_target_dt_sha256"]
    else:
        assert native is not None and identity == native["original_population_identity"]
        assert len(q) == native["metrics"]["count"]
        gap_sha = array_sha(b"frozen_lognormal_target_dt_v1\0", gap)

    history_length = np.empty(len(q), dtype=np.int64)
    truncated = np.empty(len(q), dtype=np.bool_)
    history_mean = np.empty(len(q), dtype=np.float64)
    last_q = np.empty(len(q), dtype=np.float64)
    previous_split = []
    unique_parts, starts, counts = np.unique(part, return_index=True, return_counts=True)
    for p, first, number in zip(unique_parts, starts, counts):
        target_indices = np.arange(first, first + number)
        assert (part[target_indices] == p).all()
        positions = pos[target_indices]
        sequences = np.asarray(dataset.seq_lists[p], dtype=np.int64)
        quantities = np.asarray(dataset.val_lists[p], dtype=np.float32).astype(np.float64)
        ends = positions - 1
        start = np.searchsorted(sequences, sequences[ends] - (dataset.W - 1), side="left")
        unclipped = positions - start
        start = np.maximum(start, positions - (dataset.max_len - 1))
        lengths = positions - start
        assert (lengths >= 1).all() and (lengths < dataset.max_len).all()
        history_length[target_indices] = lengths
        truncated[target_indices] = unclipped > dataset.max_len - 1
        prefix = np.concatenate(([0.], np.cumsum(quantities, dtype=np.float64)))
        history_mean[target_indices] = (prefix[positions] - prefix[start]) / lengths
        last_q[target_indices] = quantities[ends]
        previous_split.extend(str(dataset.split_lists[p][i]) for i in ends)

    offsets = np.cumsum([0] + [len(x) for x in dataset.seq_lists[:-1]])
    rows = offsets[part] + pos
    dates = frame["demand_dt"].to_numpy()[rows]
    assert np.equal(frame["seq"].to_numpy()[rows], seq).all()
    target_h, truth_h = hashlib.sha256(), hashlib.sha256()
    for n, (p, s, date) in enumerate(zip(part, seq, dates)):
        entity = str(dataset.parts[p])
        target_id = hashlib.sha256(canon([ds, data_sha, entity, int(s), split, str(date)])).hexdigest()
        target_h.update((target_id + "\n").encode())
        truth_h.update(canon([target_id, entity, int(s), split, str(date), float(q[n]), float(gap[n]),
                              int(history_length[n]), float(last_q[n]), float(history_mean[n])]) + b"\n")
    target_sha = target_h.hexdigest()
    truth_sha = truth_h.hexdigest()
    if native is not None:
        assert target_sha == native["target_identity_sha256"]
        assert truth_sha == native["truth_sha256"], (ds, split, "Native truth/history digest differs")
        assert native["metrics"]["quantity_cells"][-1]["count"] == int((q > definition["quantity_boundaries_all_train_rows"][-1]).sum())

    # Independent direct-loader samples check the derived cutoff/cap calculation.
    sample_indices = sorted({0, len(q)-1, int(np.argmin(history_length)), int(np.argmax(history_length))})
    for n in sample_indices:
        sample = dataset[n]
        assert int(sample["mask"].sum()) - 1 == int(history_length[n])
        assert float(sample["values"][-1]) == float(q[n])
        assert float(sample["dts"][-1]) == float(gap[n])
    quantity_bins = np.searchsorted(definition["quantity_boundaries_all_train_rows"], q, side="left")
    per_bin = []
    for bin_number in range(len(definition["quantity_boundaries_all_train_rows"]) + 1):
        active = quantity_bins == bin_number
        per_bin.append({"bin": bin_number, "count": int(active.sum()),
                        "true_quantity": stats(q[active]) if active.any() else None,
                        "recorded_gap": stats(gap[active]) if active.any() else None,
                        "observed_history_length": stats(history_length[active]) if active.any() else None,
                        "left_truncated_targets": int(truncated[active].sum()),
                        "mean_history_quantity": float(history_mean[active].mean()) if active.any() else None,
                        "mean_last_quantity": float(last_q[active].mean()) if active.any() else None})

    unique_dates = np.unique(dates)
    date_widths = {len(str(x)) for x in unique_dates}
    assert date_widths == ({14} if ds == "yellow_trip_hourly" else {8})
    fmt = "%Y%m%d%H%M%S" if ds == "yellow_trip_hourly" else "%Y%m%d"
    parsed = {int(x): dt.datetime.strptime(str(x), fmt) for x in unique_dates}
    months = np.asarray([parsed[int(x)].month for x in dates], dtype=np.int64)
    years = np.asarray([parsed[int(x)].year for x in dates], dtype=np.int64)
    entity_counts = np.bincount(part, minlength=len(dataset.parts))
    tail = q > definition["quantity_boundaries_all_train_rows"][-1]
    tail_series = len(np.unique(part[tail]))
    train_entities = set(frame.filter(pl.col("chronological_split") == "train")["oper_part_no"].to_list())
    first_target = pos == 1
    previous_counts = {s: previous_split.count(s) for s in ("train", "validation", "test")}
    return {"scope": "eligible_next_event_targets_from_original_frozen_loader_index", "split": split,
            "population_identity": identity, "native_target_identity_sha256": target_sha,
            "native_truth_history_sha256": truth_sha,
            "target_dt_sha256": gap_sha,
            "verification": {"original_count_quantity_identity": True,
                             "original_train_validation_target_dt_sha": split != "test",
                             "native_test_target_identity": split == "test",
                             "native_all_target_truth_and_history_digest": native is not None,
                             "direct_loader_history_sample_indices": sample_indices},
            "quantity": stats(q), "quantity_histogram": histogram(q, definition["quantity_boundaries_all_train_rows"]),
            "quantity_bin_target_statistics": per_bin,
            "tail": {"strict_threshold": definition["quantity_boundaries_all_train_rows"][-1],
                     "count": int(tail.sum()), "proportion": float(tail.mean()),
                     "represented_series": tail_series,
                     "mean_quantity": float(q[tail].mean()) if tail.any() else None,
                     "maximum_quantity": float(q[tail].max()) if tail.any() else None,
                     "target_quantity_square_share": float((q[tail] ** 2).sum() / (q ** 2).sum())},
            "recorded_gap": stats(gap), "gap_histogram": histogram(gap, [1,2,4,8,16,32,64]),
            "observed_history_length": stats(history_length),
            "history_histogram": histogram(history_length, [1,3,7,15,31,63,127,255]),
            "history": {"window_units": definition["model"]["time_observation_contract"]["unit"],
                        "lookback": dataset.W, "maximum_observed_history": dataset.max_len - 1,
                        "left_truncated_targets": int(truncated.sum()),
                        "first_target_after_series_start_count": int(first_target.sum()),
                        "previous_event_split_counts": previous_counts,
                        "target_quantity_last_quantity_pearson": corr(q, last_q),
                        "target_log1p_quantity_last_log1p_quantity_pearson": corr(np.log1p(q), np.log1p(last_q)),
                        "target_quantity_history_mean_pearson": corr(q, history_mean),
                        "available_lag_target_counts": {str(k): int((history_length >= k).sum()) for k in (1,2,4,8,16,32,64,128)},
                        "mean_boundary_available_counts": {str(k): int((history_length > k).sum()) for k in (64,128)}},
            "series": {"count": int((entity_counts > 0).sum()),
                       "absent_from_train": sorted(str(dataset.parts[p]) for p in np.unique(part) if str(dataset.parts[p]) not in train_entities),
                       "largest_target_share": float(entity_counts.max() / len(q)),
                       "target_count_per_represented_series": stats(entity_counts[entity_counts > 0]),
                       "_target_counts_for_cross_split_composition": {str(dataset.parts[p]): int(entity_counts[p]) for p in np.unique(part)}},
            "recorded_calendar": {"earliest": min(parsed.values()).isoformat(), "latest": max(parsed.values()).isoformat(),
                                  "different_recorded_dates": len(unique_dates),
                                  "month_counts": {str(m): int((months == m).sum()) for m in range(1,13)},
                                  "year_counts": {str(y): int((years == y).sum()) for y in np.unique(years)},
                                  "interpretation": "per-series chronological split; global calendar ranges may overlap"}}


def tv(a: list[float], b: list[float]) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum() / 2)


def main() -> None:
    contract_path = BUNDLE / "contract.json"
    contract = json.loads(contract_path.read_text())
    c_without_sha = {k: v for k, v in contract.items() if k != "contract_sha256"}
    assert hashlib.sha256(canon(c_without_sha)).hexdigest() == contract["contract_sha256"]
    source = contract["sources"]["history"]
    loader_rel = "data_loader/event_seq_data_module.py"
    loader_path = BUNDLE / source["root"] / loader_rel
    assert sha(loader_path) == source["files"][loader_rel]
    spec = importlib.util.spec_from_file_location("frozen_population_only_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {"schema": "frozen_three_core_population_descriptive_audit_v1", "created_unix": time.time(),
              "contract_sha256": contract["contract_sha256"], "contract_raw_sha256": sha(contract_path),
              "source_revision": source["revision"], "source_closure": source["closure"],
              "loader_source": {"path": str(loader_path), "sha256": sha(loader_path)},
              "remote_calls": 0, "checkpoint_loads": 0, "model_inference_calls": 0, "training_calls": 0,
              "test_is_independent": False, "datasets": {},
              "limitations": ["Marginal/composition changes are descriptive, not evidence that p(target|history) causally changed.",
                              "Same population is shared across seeds; population shift alone cannot explain why one seed reverses and another improves.",
                              "Observed-history features describe the frozen loader, not learned residual/attention behavior.",
                              "No new evaluation, uncertainty test, checkpoint selection or external data is performed."]}
    for ds in DATASETS:
        definition = contract["datasets"][ds]["definition"]
        data_path = BUNDLE / contract["datasets"][ds]["data_path"]
        data_sha = sha(data_path)
        assert data_sha == contract["datasets"][ds]["data_sha256"]
        frame = pl.read_parquet(data_path).sort(["oper_part_no", "seq"])
        assert not frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().any()).item()
        assert set(frame["chronological_split"].unique()) == {"train", "validation", "test"}
        for field in ("seq", "demand_qty", "delta_t"):
            x = frame[field].to_numpy().astype(np.float64)
            assert np.isfinite(x).all() and (x >= 0).all() and np.equal(x, np.round(x)).all()
        if ds == "raf_spare_parts":
            training_path = BUNDLE / contract["datasets"][ds]["training_data_path"]
            assert sha(training_path) == contract["datasets"][ds]["training_data_sha256"]
            training = pl.read_parquet(training_path).sort(["oper_part_no", "seq"])
            assert frame.filter(pl.col("chronological_split") != "test").select(training.columns).equals(training)
        frame = frame.with_columns(pl.lit(0, dtype=pl.Int32).alias("mark"),
                                   pl.col("demand_qty").cast(pl.Float64).alias("scale_residual"))
        dataset_result = {"data": {"path": str(data_path), "sha256": data_sha},
                          "time_contract": definition["model"]["time_observation_contract"],
                          "raw_row_counts": {s: frame.filter(pl.col("chronological_split") == s).height for s in ("train","validation","test")},
                          "full_frame_series_count": frame["oper_part_no"].n_unique(), "splits": {}}
        for split in ("train", "validation", "test"):
            material = frame if split == "test" else frame.filter(pl.col("chronological_split") != "test")
            dataset = module.RMTPPWeekLookbackDataset(material,
                lookback_weeks=definition["loader"]["lookback_weeks"], max_seq_len=definition["loader"]["max_seq_len"],
                mode="all", target_splits={split}, split_col="chronological_split")
            native_path = BUNDLE / "results" / f"{ds}__42__titantpp_cnn_gru54" / ("test_metrics.json" if split == "test" else "validation_replay.json")
            native = json.loads(native_path.read_text()) if split != "train" else None
            statistics = target_statistics(ds, split, material, dataset, definition, data_sha, native)
            statistics["native_receipt"] = {"path": str(native_path), "sha256": sha(native_path)} if native is not None else None
            dataset_result["splits"][split] = statistics
        val, test = (dataset_result["splits"][x] for x in ("validation", "test"))
        va, te = (x["series"]["_target_counts_for_cross_split_composition"] for x in (val, test))
        all_entities = sorted(set(va) | set(te))
        dataset_result["validation_to_test"] = {"quantity_bin_composition_tv": tv(val["quantity_histogram"]["proportions"], test["quantity_histogram"]["proportions"]),
            "gap_bin_composition_tv": tv(val["gap_histogram"]["proportions"], test["gap_histogram"]["proportions"]),
            "history_bin_composition_tv": tv(val["history_histogram"]["proportions"], test["history_histogram"]["proportions"]),
            "series_composition_tv": tv([va.get(p,0)/val["quantity"]["count"] for p in all_entities], [te.get(p,0)/test["quantity"]["count"] for p in all_entities]),
            "series_present_in_test_absent_from_validation": len(set(te) - set(va)),
            "series_present_in_validation_absent_from_test": len(set(va) - set(te)),
            "quantity_mean_percent_change": (test["quantity"]["mean"] / val["quantity"]["mean"] - 1)*100,
            "gap_mean_percent_change": (test["recorded_gap"]["mean"] / val["recorded_gap"]["mean"] - 1)*100,
            "tail_prevalence_percentage_point_change": (test["tail"]["proportion"] - val["tail"]["proportion"])*100,
            "test_to_validation_tail_prevalence_ratio": test["tail"]["proportion"] / val["tail"]["proportion"],
            "tv_definition": "Half absolute difference between normalized category shares; 0 identical shares, 1 disjoint shares; no significance claim"}
        for split in dataset_result["splits"].values():
            del split["series"]["_target_counts_for_cross_split_composition"]
        result["datasets"][ds] = dataset_result
    assert all(sha(Path(result["datasets"][ds]["data"]["path"])) == result["datasets"][ds]["data"]["sha256"] for ds in DATASETS)
    assert sha(loader_path) == source["files"][loader_rel]
    result["script_sha256"] = sha(Path(__file__))
    result["status"] = "PASS_DESCRIPTIVE_POPULATION_IDENTITIES"
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "population_diagnosis.json"
    output.write_bytes(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()+b"\n")
    labels = {"yellow_trip_hourly":"Taxi", "intermittent_frozen_5000":"Intermittent", "raf_spare_parts":"RAF"}
    lines = ["# 동결된 Train·Validation·Test 대상의 구성 진단", "", "**상태: 완료.** 선택 checkpoint나 모델을 읽지 않고, 원래 frozen loader의 적격 next-event target만 CPU로 집계했다. 새 학습·추론·원격 명령은 0회다.", "",
             "Train/Validation의 target 수·identity·quantity SHA 및 target duration SHA가 원계약과 일치한다. Validation/Test 모두 native target digest와 모든 대상의 truth/history digest가 저장된 영수증에 정확히 일치한다. 따라서 이력 길이·직전 수량·이력 평균의 재구성도 native 평가와 일치한다. 직접 loader sample 검사도 통과했다. 첫 관측행을 평가 대상으로 잘못 포함하지 않았다.", "",
             "| 자료·split | 평가 대상/series | 수량 평균·p99·max | 큰 수량 비중 | gap 평균·p99 | 이력 길이 평균·p50 |", "|---|---:|---:|---:|---:|---:|"]
    for ds, data in result["datasets"].items():
        for s, o in data["splits"].items():
            lines.append(f"| {labels[ds]} {s} | {o['quantity']['count']:,}/{o['series']['count']:,} | {o['quantity']['mean']:.3f} · {o['quantity']['quantiles']['0.99']:.3f} · {o['quantity']['maximum']:.0f} | {100*o['tail']['proportion']:.4f}% ({o['tail']['count']:,}) | {o['recorded_gap']['mean']:.3f} · {o['recorded_gap']['quantiles']['0.99']:.3f} | {o['observed_history_length']['mean']:.2f} · {o['observed_history_length']['quantiles']['0.5']:.0f} |")
    lines.extend(["", "**Validation→Test에서 확인된 변화**", ""])
    for ds, data in result["datasets"].items():
        v, t = data["splits"]["validation"], data["splits"]["test"]
        change = data["validation_to_test"]
        lines.append(f"- {labels[ds]}: 수량 평균 {change['quantity_mean_percent_change']:+.2f}%, gap 평균 {change['gap_mean_percent_change']:+.2f}%, 큰 수량 빈도 {change['tail_prevalence_percentage_point_change']:+.4f}%p. 수량 구간 구성 거리(TV) {change['quantity_bin_composition_tv']:.4f}, 이력 길이 구성 거리 {change['history_bin_composition_tv']:.4f}, series별 대상 비중 거리 {change['series_composition_tv']:.4f}. Train에 없던 Test series는 {len(t['series']['absent_from_train'])}개다.")
        lines.append(f"  Validation 기록 기간 {v['recorded_calendar']['earliest']}–{v['recorded_calendar']['latest']}; Test {t['recorded_calendar']['earliest']}–{t['recorded_calendar']['latest']}. 분할은 series별이어서 자료 전체의 기간은 겹칠 수 있다.")
    raf = result["datasets"]["raf_spare_parts"]["splits"]
    inter = result["datasets"]["intermittent_frozen_5000"]["splits"]
    lines.extend(["", "**이력과 대상 구성에서 확인한 진단 단서**", "",
                  f"- RAF 이력 최대값은 Train {raf['train']['observed_history_length']['maximum']:.0f}, Validation {raf['validation']['observed_history_length']['maximum']:.0f}, Test {raf['test']['observed_history_length']['maximum']:.0f}개다. 이력 16개 이상인 대상은 각각 {raf['train']['history']['available_lag_target_counts']['16']}, {raf['validation']['history']['available_lag_target_counts']['16']}, {raf['test']['history']['available_lag_target_counts']['16']}개다. 32·64·128 lag 이용 가능성은 모든 split에서 0이다. 특정 branch가 이 조건에서 실제 활성화되는지는 구조 코드 검증이 별도로 필요하다.",
                  "- RAF Test의 879 series는 Validation에 없고, Validation의 1,693 series는 Test에 없다. 모두 Train에 있던 series다. 따라서 새 entity 문제보다 평가 대상 series 구성과 성숙한 이력의 차이를 먼저 구분해야 한다.",
                  f"- Intermittent Test의 {inter['test']['history']['left_truncated_targets']:,}개 대상은 sequence cap으로 과거를 왼쪽에서 잘라낸다. Train/Validation에서는 이 경우가 0개다. 이력 평균과 target 수량의 기술적 Pearson 상관은 Validation {inter['validation']['history']['target_quantity_history_mean_pearson']:.3f}에서 Test {inter['test']['history']['target_quantity_history_mean_pearson']:.3f}로 달라진다. 이는 loss-free association 요약이며 반전 원인 확정이나 독립 표본 검정이 아니다.",
                  "- Taxi 큰 수량은 Validation/Test 모두 4개 series에서 나온다. 전체 구성이 비교적 비슷하므로 seed 반전을 강한 자료 분포 변화 하나로 설명할 근거는 약하다.",
                  "- 날짜는 파일의 기록 날짜다. 특히 Intermittent 날짜 범위가 미래 연도까지 뻗는다는 사실을 실제 미래 관측 자료나 독립 미래 평가의 증거로 해석하지 않는다."])
    lines.extend(["", "**원인 해석의 한계**", "", "- 구성 차이는 실제로 확인되었지만, 이것만으로 보정 경로의 손해 원인이나 조건부 분포 변화가 증명되지는 않는다.",
                  "- 세 seed의 평가 대상은 같으므로, 자료 구성 변화만으로 특정 seed만 반전하는 이유를 설명할 수 없다. 학습 경로·선택 epoch·모델별 오류 분해와 함께 보아야 한다.",
                  "- 큰 수량의 squared quantity share는 오차 SSE의 비중과 다르다. 모델별 SSE 분해는 저장된 native 결과에서 별도로 확인해야 한다.",
                  "- 이력 길이와 lag의 이용 가능성은 계산 입력의 특성이다. 실제 learned correction의 크기·부호·활성 경로를 직접 관측한 것은 아니다.",
                  "- 불변 Train 통계를 새로 맞추지 않았다. 모든 duration 단위(hour/week/month)와 top-code 없음은 원계약을 그대로 따랐다.",
                  "", "**추가 기록의 최소 범위 — 다음 작업 / 별도 평가 실행 전 계약 필요**", "", "- 기존 선택 모델을 바꾸지 않고, 사전에 고정한 series·기간·이력 길이·수량 구간별 signed error/SSE/MAE/TimeNLL 및 correction 크기·부호를 같은 대상에 함께 기록한다. 모델별 반전이 적은 수의 series나 특정 이력 조건에 집중되는지 확인한다.",
                  "- split별 전체 구성을 공통 고정 가중치로 재집계하되, 원래 전체 Test 수치도 함께 남긴다. 가중치 실험은 성능 개선 주장이 아니라 구성 변화 민감도 진단이다.",
                  "- Test를 이미 보고 정한 후속 변경은 개발 조치다. 독립 일반화 확인에는 이후에 동결된 새 평가 계약과 미노출 자료가 필요하다.", "",
                  f"계약 canonical SHA: `{contract['contract_sha256']}`. Source closure: `{source['closure']}`.",
                  "각 입력·loader·native receipt의 경로와 raw SHA, 모든 분포 통계 및 정체성 검증은 `population_diagnosis.json`에 있다."])
    (OUT / "POPULATION_DIAGNOSIS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status": result["status"], "output": str(output), "sha256":sha(output),
                      "split_counts":{ds:{s:o['quantity']['count'] for s,o in d['splits'].items()} for ds,d in result['datasets'].items()}}))


if __name__ == "__main__":
    main()
