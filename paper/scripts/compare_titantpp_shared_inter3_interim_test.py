"""Compare frozen interim Intermittent Shared15 Validation3/Test3 with stored references.

Only local immutable evidence is read. No training, checkpoint reselection,
model construction or model inference is implemented by this comparator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

from paper.scripts import compare_titantpp_shared_raw_raf3_test as previous

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_shared_inter3_interim_test_20261020_v1"
DATASET = "intermittent_frozen_5000"
COUNTS = {"validation": 86285, "test": 88019}
TAIL_THRESHOLD = 187
SHARED, MLP16, EXTERNAL = previous.SHARED, previous.MLP16, previous.EXTERNAL
SEEDS, GROUPS, METRICS, LABELS = previous.SEEDS, previous.GROUPS, previous.METRICS, previous.LABELS
require, read, sha, canonical, write = previous.require, previous.read, previous.sha, previous.canonical, previous.write
simple_quantity_baselines = previous.simple_quantity_baselines
PREVIOUS_SHA = "5469ceb02be20d2559c51232a7b63dd5917540d5bc317e2903d0ef29141664dd"
require(sha(Path(previous.__file__)) == PREVIOUS_SHA, "Reviewed RAF comparator changed")


def aggregate(records, split="test", selected_epochs=None):
    """Reuse reviewed six-metric arithmetic with explicit split/population constants."""
    require(split in COUNTS and len(records) == 24, "Expected candidate3 plus stored references21 for one split")
    for row in records:
        require(row.get("dataset") == DATASET and row.get("split") == split, "Mixed/foreign dataset or split")
        if row.get("model") == SHARED and selected_epochs is not None:
            require(row.get("selected_epoch") == selected_epochs.get(row.get("seed")),
                    "Frozen interim Validation-selected epoch changed")
    namespace = dict(previous.__dict__)
    source = inspect.getsource(previous.aggregate)
    source = source.replace('"raf_spare_parts"', repr(DATASET)).replace("5226", str(COUNTS[split]))
    source = source.replace('"test"', repr(split)).replace("shared_raw_raf3_three_seed_test_v1", "shared_inter3_interim_three_seed_v1")
    exec(compile(source, __file__, "exec"), namespace)
    result = namespace["aggregate"](records)
    result.update(interim_checkpoint_snapshot=True, training_campaign_complete=False,
        snapshot_selection="contained best_state_dict at earliest finite strict Validation raw RMSE minimum",
        held_out_test_evaluated=split == "test", candidate_count=3, baseline_reference_count=21,
        unique_comparison_baseline_count=21, seed_count=3, standard_deviation="sample ddof=1",
        tail_threshold=TAIL_THRESHOLD, checkpoint_reselected=False,
        comparison_primary="metric-specific external-six mean", comparison_secondary=MLP16)
    return result


def markdown(result):
    split = result["split"]
    body = previous.markdown(result).replace("원본 RAF 공유형의 고정 checkpoint Test 비교",
        f"원본 Intermittent 공유형의 중간 checkpoint {split} 비교")
    body = body.replace("q>200", "q>187").replace("Test 비교", f"{split} 비교")
    body = body.replace("신규5090/cuDNN92000과 기존5080/cuDNN91900 및 일부 CPU 재현 결과의 차이가 있다.",
        "공유형은 5090에서 학습 중인 checkpoint의 best_state_dict를 고정해 5080에서 평가했다. 기존 비교군의 Runtime 차이는 남는다.")
    body += ("\n학습 캠페인은 계속 진행 중이다. 이 표는 캡처 시점의 중간 best checkpoint 결과이며 최종 학습 결과가 아니다. "
             "평가는 checkpoint를 변경하지 않았고 Test로 epoch를 재선택하지 않았다.\n"
             "외부6종의 지표별 최선 평균을 1차 비교로, MLP16을 보조 비교로 읽는다. "
             "여섯 지표와 불리한 seed별 절대·상대 차이는 JSON에 보존했다.\n")
    if split == "validation":
        body += ("외부 Validation18은 기존 selected endpoint 집계이며 개별 예측/행 순서 증적은 없다. "
                 "checkpoint·state·history와 원본 모집단 및 큰수량 개수로 연결했다. "
                 "이번 후보의 Validation 재현은 기존 history의 전체 RMSE·MAE·TimeNLL을 고정 허용오차 1e-5로 대조했다.\n")
    return body


def verify_actual_exit(file):
    actual = read(file)
    require(actual.get("actual_os_returncode") == 0 and actual.get("timeout") is False
        and actual.get("actual_child_exit_observed") is True, "Native child actual exit0 missing: " + str(file))
    return actual


def collect_records(originals, contract, refs):
    digest = canonical(contract)
    records = {split: [deepcopy(r) for r in refs["references"] if r.get("split") == split] for split in COUNTS}
    require(all(len(rows) == 21 for rows in records.values()), "Expected stored Validation21 and Test21")
    jobs = contract.get("jobs", [])
    require(len(jobs) == 3 and {(j["dataset"], j["seed"]) for j in jobs} == {(DATASET, s) for s in SEEDS},
            "Incomplete interim candidate matrix")
    for job in jobs:
        folder = originals / "results" / job["id"]
        for split in COUNTS:
            item_path = folder / f"{split}_receipt.json"
            item = read(item_path)
            verify_actual_exit(folder / f"{split}_exit.json")
            expected_sha = job.get("checkpoint", job.get("resume_checkpoint", {})).get("sha256")
            require(item.get("status") == "complete" and item.get("split") == split
                and item.get("contract_sha256") == digest and item.get("job_id") == job["id"]
                and item.get("selected_epoch") == job["selected_epoch"]
                and item.get("checkpoint_sha256") == expected_sha,
                    "Candidate frozen interim checkpoint/receipt differs")
            require(item.get("parameters_unchanged") is True, "Candidate parameters changed during evaluation")
            records[split].append({"dataset": DATASET, "model": SHARED, "seed": job["seed"],
                "split": split, "data_variant": "original_uncapped", "selected_epoch": item["selected_epoch"],
                "metrics": item["metrics"], "target_identity": item["target_identity"],
                "simple_quantity_baselines": simple_quantity_baselines(item), "receipt_sha256": sha(item_path)})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = args.bundle.resolve()
    receipt_path = bundle / "retrieval/receipt.json"
    collected = read(receipt_path)
    require(collected.get("status") in ("verified", "complete")
        and collected.get("owner_absence_verified") is True
        and collected.get("candidate_test_count") == collected.get("validation_replay_count") == 3
        and collected.get("actual_pipeline_exit0_count") == 1, "Verified interim collection required")
    originals = bundle / "retrieval/originals"
    archive, manifest_path = bundle / "retrieval/original.tar.gz", bundle / "retrieval/manifest.json"
    require(sha(archive) == collected["archive_sha256"] and sha(manifest_path) == collected["manifest_sha256"],
            "Evaluation archive/manifest changed")
    inventory = read(manifest_path)["files"]
    require(bool(inventory) and len(inventory) == collected["file_count"], "Empty/incomplete inventory")
    for relative, digest in inventory.items():
        file = (originals / relative).resolve()
        require(file.is_relative_to(originals.resolve()) and not file.is_symlink() and sha(file) == digest,
                "Collected original changed: " + relative)
    contract = read(originals / "evaluation_contract.json")
    require(contract.get("schema") == "titantpp_shared_inter3_interim_frozen_v1" and contract.get("host") == "5080",
            "Foreign interim evaluation contract")
    digest = canonical(contract)
    pipeline, pipeline_exit = read(originals / "pipeline_receipt.json"), read(originals / "pipeline_exit.json")
    require(pipeline.get("status") == "complete" and pipeline.get("contract_sha256") == digest
        and pipeline.get("qualification_exit0_count") == 3 and pipeline.get("owned_pids_absent") is True
        and pipeline_exit.get("actual_os_returncode") == 0 and pipeline_exit.get("timeout") is False,
            "Incomplete native interim pipeline")
    ref_path = originals / contract["references"]["path"]
    require(sha(ref_path) == contract["references"]["sha256"], "Stored reference registry changed")
    refs = read(ref_path)
    require(refs.get("reference_count") == 42 and refs.get("seed_count") == 3, "Expected stored reference42")
    for relative, expected in refs["source_bindings"].items():
        require(sha(ROOT / relative) == expected, "Historical reference changed: " + relative)
    records = collect_records(originals, contract, refs)
    epochs = {j["seed"]: j["selected_epoch"] for j in contract["jobs"]}
    report = ROOT / "reports" / NAME
    paths = []
    for split in COUNTS:
        result = aggregate(records[split], split, epochs)
        result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        for suffix, value in (("json", result), ("md", markdown(result))):
            file = report / f"THREE_SEED_{split.upper()}.{suffix}"
            write(file, value)
            paths.append(file)
    final = {"status": "complete", "candidate_test_count": 3, "candidate_validation_count": 3,
        "baseline_reference_count": 42, "baseline_reference_count_by_split": {"validation": 21, "test": 21},
        "unique_comparison_baseline_count": 21, "seed_count": 3, "dataset_count": 1,
        "validation_replay_count": 3, "qualification_exit0_count": 3,
        "actual_data_worker_exit0_count": 6, "actual_native_worker_exit0_count": 9,
        "actual_pipeline_exit0_count": 1, "held_out_test_evaluated": True,
        "checkpoint_reselected": False, "historical_baselines_reevaluated": False,
        "training_campaign_complete": False, "interim_checkpoint_snapshot": True,
        "whole_original_raw9_completed": False, "collection_receipt_sha256": sha(receipt_path),
        "result_files": {str(p): sha(p) for p in paths}}
    write(bundle / "final_receipt.json", final)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
