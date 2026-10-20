"""Compare fixed original RAF/Taxi Shared15 Test6 with stored Test42.

The reviewed RAF arithmetic is reused unchanged except for the explicit dataset
and population constants. This module never evaluates or selects a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import inspect
from pathlib import Path

from paper.scripts import compare_titantpp_shared_raw_raf3_test as previous

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_shared_raf_taxi6_test_20261020_v1"
DATASETS = {"raf_spare_parts": 5226, "yellow_trip_hourly": 8327}
DATASET_LABELS = {"raf_spare_parts": "RAF", "yellow_trip_hourly": "Taxi"}
THRESHOLDS = {"raf_spare_parts": 200, "yellow_trip_hourly": 3449}
EPOCHS = {"raf_spare_parts": {42: 9, 52: 5, 62: 19},
          "yellow_trip_hourly": {42: 112, 52: 148, 62: 127}}
SHARED, MLP16, EXTERNAL = previous.SHARED, previous.MLP16, previous.EXTERNAL
SEEDS, GROUPS, METRICS, LABELS = previous.SEEDS, previous.GROUPS, previous.METRICS, previous.LABELS
require, read, sha, canonical, write = previous.require, previous.read, previous.sha, previous.canonical, previous.write
simple_quantity_baselines = previous.simple_quantity_baselines
PREVIOUS_SHA = "5469ceb02be20d2559c51232a7b63dd5917540d5bc317e2903d0ef29141664dd"
require(sha(Path(previous.__file__)) == PREVIOUS_SHA, "Reviewed RAF comparator changed")


def aggregate(records):
    """Require the full two-dataset matrix; preserve all six metrics and signs."""
    require(len(records) == 48 and {r.get("dataset") for r in records} == set(DATASETS),
            "Expected candidate6 plus stored references42 on RAF and Taxi")
    results = {}
    source = inspect.getsource(previous.aggregate)
    for dataset, count in DATASETS.items():
        namespace = dict(previous.__dict__)
        adapted = source.replace('"raf_spare_parts"', repr(dataset)).replace("5226", str(count))
        adapted = adapted.replace("shared_raw_raf3_three_seed_test_v1", "shared_raf_taxi6_dataset_test_v1")
        exec(compile(adapted, __file__, "exec"), namespace)
        rows = [r for r in records if r.get("dataset") == dataset]
        for row in rows:
            if row.get("model") == SHARED:
                require(row.get("selected_epoch") == EPOCHS[dataset].get(row.get("seed")),
                        "Frozen Validation-selected candidate epoch changed")
        results[dataset] = namespace["aggregate"](rows)
    return {"schema": "shared_raf_taxi6_three_seed_test_v1", "split": "test",
        "data_variant": "original_uncapped", "datasets": results, "seed_count": 3,
        "candidate_test_count": 6, "baseline_reference_count": 42,
        "held_out_test_evaluated": True, "checkpoint_reselected": False,
        "historical_baselines_reevaluated": False, "independent_new_test": False,
        "cross_dataset_raw_error_pooled": False}


def markdown(result):
    lines = ["# 원본 RAF·Taxi 공유형 MLP의 고정 checkpoint 3seed Test", "",
        "Validation RMSE로 선택한 checkpoint를 고정했다. 평균±표본SD(ddof1), 낮을수록 좋다.",
        "학습5090/cuDNN92000의 선택checkpoint를5080/cuDNN91900에서 평가했다. "
        "개별 Validation 예측·시간손실과 여섯 지표의 고정 허용오차 재현을 통과한 뒤 Test를 실행했다.", ""]
    for dataset, item in result["datasets"].items():
        body = previous.markdown(item)
        body = body.replace("원본 RAF 공유형", "원본 " + DATASET_LABELS[dataset] + " 공유형")
        body = body.replace("q>200", "q>" + str(THRESHOLDS[dataset]))
        body = body.replace("신규5090/cuDNN92000과 기존5080/cuDNN91900 및 일부 CPU 재현 결과의 차이가 있다.",
            "신규 모델의 학습5090/cuDNN92000과 평가5080/cuDNN91900이 다르며, 기존 비교군에는 일부 CPU 재현 결과가 있다.")
        lines.extend([body, ""])
    lines.extend(["## 평가 범위", "",
        "- 원본 raw9 중 성공 종료한 RAF·Taxi6개만 별도 읽기 전용 회수했다. 전체 raw9 종료·회수를 뜻하지 않는다.",
        "- 기존 CNN·MLP16·외부 모델은 변경·재학습·재추론하지 않았다.",
        "- 5seed 확장·새 구조 탐색·Test에 따른 checkpoint 재선택은 수행하지 않았다.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = args.bundle.resolve()
    receipt_path = bundle / "retrieval/receipt.json"
    collected = read(receipt_path)
    require(collected.get("status") in ("verified", "complete")
        and collected.get("owner_absence_verified") is True
        and collected.get("candidate_test_count") == collected.get("validation_replay_count") == 6
        and collected.get("actual_pipeline_exit0_count") == 1, "Verified Test6 collection required")
    originals = bundle / "retrieval/originals"
    archive, manifest_path = bundle / "retrieval/original.tar.gz", bundle / "retrieval/manifest.json"
    require(sha(archive) == collected["archive_sha256"]
        and sha(manifest_path) == collected["manifest_sha256"], "Evaluation archive/manifest changed")
    inventory = read(manifest_path)["files"]
    require(bool(inventory) and len(inventory) == collected["file_count"], "Empty/incomplete inventory")
    for relative, digest in inventory.items():
        path = (originals / relative).resolve()
        require(path.is_relative_to(originals.resolve()) and not path.is_symlink()
            and sha(path) == digest, "Collected original changed: " + relative)
    c = read(originals / "evaluation_contract.json")
    require(c.get("schema") == "titantpp_shared_raf_taxi6_frozen_v1" and c.get("host") == "5080",
            "Foreign evaluation contract")
    require(len(c.get("jobs", [])) == 6 and {(j["dataset"], j["seed"]) for j in c["jobs"]}
        == {(d, s) for d in DATASETS for s in SEEDS}, "Incomplete candidate job matrix")
    digest = canonical(c)
    pipeline = read(originals / "pipeline_receipt.json")
    pipeline_exit = read(originals / "pipeline_exit.json")
    require(pipeline.get("status") == "complete" and pipeline.get("contract_sha256") == digest
        and pipeline.get("qualification_exit0_count") == 6 and pipeline.get("owned_pids_absent") is True
        and pipeline_exit.get("actual_os_returncode") == 0 and pipeline_exit.get("timeout") is False,
            "Incomplete native pipeline")
    refs_path = originals / c["references"]["path"]
    require(sha(refs_path) == c["references"]["sha256"], "Stored reference registry changed")
    refs = read(refs_path)
    require(refs.get("reference_count") == 42 and refs.get("seed_count") == 3,
            "Expected stored MLP16/external Test42")
    for relative, expected in refs["source_bindings"].items():
        require(sha(ROOT / relative) == expected, "Historical Test reference changed: " + relative)
    records = deepcopy(refs["references"])
    for job in c["jobs"]:
        folder = originals / "results" / job["id"]
        item = read(folder / "test_receipt.json")
        actual = read(folder / "test_exit.json")
        require(actual.get("actual_os_returncode") == 0 and actual.get("timeout") is False
            and actual.get("actual_child_exit_observed") is True,
                "Test worker actual exit0 missing")
        require(item.get("status") == "complete" and item.get("split") == "test"
            and item.get("contract_sha256") == digest and item.get("job_id") == job["id"]
            and item.get("selected_epoch") == job["selected_epoch"] == EPOCHS[job["dataset"]][job["seed"]]
            and item.get("checkpoint_sha256") == job["checkpoint"]["sha256"],
                "Candidate checkpoint/receipt changed")
        records.append({"dataset": job["dataset"], "model": SHARED, "seed": job["seed"],
            "split": "test", "data_variant": "original_uncapped", "selected_epoch": item["selected_epoch"],
            "metrics": item["metrics"], "target_identity": item["target_identity"],
            "simple_quantity_baselines": simple_quantity_baselines(item),
            "receipt_sha256": sha(folder / "test_receipt.json")})
    result = aggregate(records)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report = ROOT / "reports" / NAME
    paths = [report / "THREE_SEED_TEST.json", report / "THREE_SEED_TEST.md"]
    write(paths[0], result)
    write(paths[1], markdown(result))
    final = {"status": "complete", "candidate_test_count": 6, "baseline_reference_count": 42,
        "seed_count": 3, "dataset_count": 2, "validation_replay_count": 6,
        "qualification_exit0_count": 6, "actual_data_worker_exit0_count": 12,
        "actual_native_worker_exit0_count": 18, "actual_pipeline_exit0_count": 1,
        "held_out_test_evaluated": True, "checkpoint_reselected": False,
        "historical_baselines_reevaluated": False, "whole_original_raw9_completed": False,
        "collection_receipt_sha256": sha(receipt_path),
        "result_files": {str(path): sha(path) for path in paths}}
    write(bundle / "final_receipt.json", final)
    print(__import__("json").dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
