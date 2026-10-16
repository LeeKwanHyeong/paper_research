"""Build a Validation-only review from one saved A100 observation; no network or inference."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
B = ROOT / "search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1"
EXPECTED_CONTRACT = "46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae"
EXPECTED_SOURCE = "40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd"
DATASETS = {"yellow_trip_hourly": "Taxi", "intermittent_frozen_5000": "Intermittent", "raf_spare_parts": "RAF"}
ARMS = {"titantpp_history_mlp_width16": "MLP16", "titantpp_cnn_mlp16": "CNN+MLP16", "titantpp_gru54": "GRU54", "titantpp_cnn_gru54": "CNN+GRU54"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint_checks(e):
    assert e["evaluation_scope"] == "validation_only" and e["held_out_test_evaluated"] is False
    assert e["time_metric"] == "recorded_positive_integer_time_nll"
    cells = e["quantity_cells"]
    assert sum(x["count"] for x in cells) == e["count"]
    assert e["tail"] == cells[-1] or all(e["tail"][k] == cells[-1][k] for k in e["tail"])
    for cell in [e, *cells]:
        assert cell["count"] > 0
        assert all(math.isfinite(cell[k]) for k in ("qty_rmse", "qty_mae", "time_nll", "qty_sse"))
        assert math.isclose(cell["qty_rmse"], math.sqrt(cell["qty_sse"] / cell["count"]), rel_tol=1e-10)
    assert math.isclose(sum(x["qty_sse"] for x in cells), e["qty_sse"], rel_tol=1e-10)
    for key in ("qty_mae", "time_nll"):
        assert math.isclose(sum(x[key] * x["count"] for x in cells) / e["count"], e[key], rel_tol=1e-10, abs_tol=1e-12)


def row(dataset, arm, ep, saved, selected, parameters, source):
    endpoint_checks(selected)
    return {"dataset": dataset, "arm": arm, "seed": 42, "split": "validation", "selected_epoch": ep,
            "saved_epoch": saved, "parameter_count": parameters, "count": selected["count"],
            "qty_rmse": selected["qty_rmse"], "qty_mae": selected["qty_mae"], "time_nll": selected["time_nll"],
            "tail_threshold_exclusive": selected["quantity_boundaries"][-1], "tail_count": selected["tail"]["count"],
            "tail_qty_rmse": selected["tail"]["qty_rmse"], "tail_qty_mae": selected["tail"]["qty_mae"],
            "tail_time_nll": selected["tail"]["time_nll"], "source": source}


def main():
    latest_path = B / "hourly_monitor/latest.json"
    frozen_reference = OUT / "observation_reference.json"
    if not frozen_reference.exists():
        write_json("observation_reference.json", {"latest_path": str(latest_path), "latest_sha256_at_capture": sha(latest_path), "latest": read(latest_path)})
    latest = read(frozen_reference)["latest"]
    snapshot_path = Path(latest["snapshot"])
    assert sha(snapshot_path) == latest["snapshot_sha256"]
    assert latest["pod_id"] == "okbl76hxfr7a9d" and latest["observe_py_invocations"] == 1
    contract = read(B / "execution_contract.json")
    assert canonical(contract) == EXPECTED_CONTRACT
    assert contract["source"]["files_sha256"] == EXPECTED_SOURCE
    snapshot = read(snapshot_path)
    analysis = read(Path(latest["analysis"]))
    provider_path = Path(latest["path"]) / "provider_manager_records.json"
    provider = read(provider_path)
    allocation = read(B / "control/allocation.json")
    qualification = snapshot["files"]["qualification/receipt.json"]
    assert qualification["contract_sha256"] == EXPECTED_CONTRACT and qualification["status"] == "passed"
    assert qualification["held_out_test_evaluated"] is False
    baselines, records, bindings = {}, [], []
    for dataset, ref in contract["baseline_replays"].items():
        source = ROOT / ref["origin_replay"]
        checkpoint = ROOT / ref["origin_checkpoint"]
        assert sha(source) == ref["origin_replay_sha256"] and sha(checkpoint) == ref["checkpoint_sha256"]
        endpoints = read(source)
        assert endpoints["evaluation_scope"] == "validation_only" and endpoints["held_out_test_evaluated"] is False
        assert endpoints["best_epoch"] == ref["selected_epoch"]
        replay = qualification["baselines"][dataset]
        assert replay["status"] == "passed" and replay["full_validation_replay"] is True
        assert replay["checkpoint_sha256"] == ref["checkpoint_sha256"]
        for key in ("qty_rmse", "qty_mae", "time_nll"):
            assert math.isclose(endpoints["selected"][key], replay["metrics"][key], rel_tol=1e-5, abs_tol=1e-5)
        baselines[dataset] = row(dataset, "titantpp_history_mlp_width16", ref["selected_epoch"],
                                endpoints["completed_epochs"], endpoints["selected"], endpoints["parameter_count"], ref["origin_replay"])
        records.append(baselines[dataset])
        bindings.append({"dataset": dataset, "role": "baseline", "endpoint_path": str(source), "endpoint_sha256": sha(source),
                         "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                         "A100_baseline_replay_metrics": replay["metrics"], "A100_baseline_diagnostic_sha256": replay["diagnostic_sha256"]})
    completed, pending, selected_records = [], [], []
    by_job = {r["job"]["id"]: r for r in analysis["rows"]}
    for run in snapshot["runs"]:
        job = run["job"]
        assert job["seed"] == 42 and job["dataset"] in DATASETS
        observed = by_job[job["id"]]
        if observed["state"] != "completed":
            assert observed["state"] == "training"
            pending.append(observed)
            continue
        manifest, endpoints = run["terminal_manifest"], run["endpoints"]
        assert manifest["scientific_success"] is True and manifest["contract_sha256"] == EXPECTED_CONTRACT
        assert manifest["held_out_test_evaluated"] is False and endpoints["held_out_test_evaluated"] is False
        assert run["small_record_SHA_passed"] is True and all(run["small_record_checks"].values())
        assert observed["terminal_verified"] and observed["selected_last_validation_replay_verified"]
        prefix = "runs/" + job["arm"] + "/count_only_log_regression/seed_42/"
        relative = "run/" + job["id"] + "/" + prefix + "endpoint_replays.json"
        assert snapshot["file_sha256"][relative] == manifest["files"][prefix + "endpoint_replays.json"]
        assert endpoints["evaluation_scope"] == "validation_only" and endpoints["status"] == "complete"
        earliest_min = min(run["history"]["history"], key=lambda h: (h["val_qty_rmse"], h["epoch"]))
        assert endpoints["best_epoch"] == earliest_min["epoch"]
        assert math.isclose(earliest_min["val_qty_rmse"], endpoints["selected"]["qty_rmse"], rel_tol=1e-5, abs_tol=1e-5)
        endpoint_checks(endpoints["last"])
        base = baselines[job["dataset"]]
        record = row(job["dataset"], job["arm"], endpoints["best_epoch"], endpoints["completed_epochs"],
                     endpoints["selected"], endpoints["parameter_count"], str(snapshot_path.relative_to(ROOT)) + "#" + job["id"])
        assert record["count"] == base["count"] and record["tail_count"] == base["tail_count"]
        assert record["tail_threshold_exclusive"] == base["tail_threshold_exclusive"]
        completed.append(record)
        records.append(record)
        selected_records.append({"job": job, "selected": endpoints["selected"], "last": endpoints["last"], "terminal_manifest": manifest})
        bindings.append({"job": job, "role": "completed_candidate", "terminal_manifest_sha256": snapshot["file_sha256"]["run/" + job["id"] + "/terminal_manifest.json"],
                         "endpoint_sha256": manifest["files"][prefix + "endpoint_replays.json"],
                         "selected_checkpoint_declared_sha256": manifest["files"][prefix + "best_val_qty_rmse_model.pt"],
                         "remote_small_record_SHA_verified": True, "binary_downloaded_in_this_review": False})
    assert len(completed) == latest["status_counts"]["completed"] == 6 and len(pending) == 3
    deltas = []
    for r in completed:
        base = baselines[r["dataset"]]
        delta = {"dataset": r["dataset"], "arm": r["arm"], "seed": 42, "split": "validation"}
        for key in ("qty_rmse", "qty_mae", "tail_qty_rmse", "tail_qty_mae", "time_nll", "tail_time_nll"):
            delta[key + "_delta"] = r[key] - base[key]
            if "time_nll" not in key:
                delta[key + "_change_percent"] = 100 * (r[key] / base[key] - 1)
        deltas.append(delta)
    write_csv("validation_completed_and_baseline.csv", records)
    write_csv("validation_paired_changes.csv", deltas)
    write_json("completed_selected_records.json", selected_records)
    write_json("in_progress_observation.json", pending)
    write_json("verification.json", {"status": "passed", "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "observed_kst": latest["observed_kst"], "snapshot_path": str(snapshot_path), "snapshot_sha256": sha(snapshot_path),
        "analysis_path": latest["analysis"], "analysis_sha256": sha(Path(latest["analysis"])),
        "provider_records_path": str(provider_path), "provider_records_sha256": sha(provider_path),
        "frozen_observation_reference": str(frozen_reference), "frozen_observation_reference_sha256": sha(frozen_reference),
        "allocation_sha256": sha(B / "control/allocation.json"),
        "contract_canonical_sha256": canonical(contract), "source_closure_sha256": EXPECTED_SOURCE,
        "status_counts": latest["status_counts"], "source_bindings": bindings,
        "checks": ["local snapshot SHA", "frozen contract identity", "baseline checkpoint and endpoint SHA", "A100 baseline replay parity",
                   "remote completed small-record SHA", "selected and last full Validation replay", "earliest minimum Validation RMSE epoch",
                   "quantity-cell counts, SSE and count-weighted MAE/Time NLL", "same population counts and frozen tail boundary"],
        "baseline_tail_metrics_runtime": "original selected checkpoint Validation replay; A100 baseline parity passed for overall metrics",
        "candidate_binary_original_retrieval": "pending; manager retrieves after campaign terminal", "candidate_CPU_binary_reinference": "not performed",
        "new_network_calls": 0, "new_training_or_evaluation": False, "baseline_reselected": False})
    text = ["# A100 CNN·GRU 완료 조건 검토", "", f"실제 관측: **{latest['observed_kst']}**. 완료6·진행3·실패0·대기0·미확정0, 현재3병렬입니다.", "",
        "기존 MLP16 seed42의 동일 선택 checkpoint를 기준으로 **완료된 여섯 조건의 전체 Validation**을 비교했습니다. 현재 A100의 Test는 접근하지 않았습니다. 기준선은 원래 GPU의 Validation replay이고 동일 checkpoint의 A100 full Validation replay가 허용오차 안에서 통과했습니다. 기준선의 tail 값은 원래 replay에서 읽었습니다.", "",
        "**완료 결과를 비교한다 — 완료**", "", "모든 지표는 낮을수록 좋습니다. tail은 Train에서 고정한 raw 수량 경계 초과(Taxi3449·Intermittent187·RAF200)입니다. 선택 epoch는 최초 최소 전체 Validation raw RMSE 기준입니다.", "",
        "| 데이터 | 모델 | 선택 / 저장 epoch | 전체 RMSE | 전체 MAE | 전체 Time NLL | tail RMSE | tail MAE | tail Time NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for r in [x for x in records if x["dataset"] == dataset]:
            text.append(f"| {DATASETS[dataset]} | {ARMS[r['arm']]} | {r['selected_epoch']} / {r['saved_epoch']} | {r['qty_rmse']:.6f} | {r['qty_mae']:.6f} | {r['time_nll']:.6f} | {r['tail_qty_rmse']:.6f} | {r['tail_qty_mae']:.6f} | {r['tail_time_nll']:.6g} |")
    text += ["", "| 데이터 | 완료 후보 | 전체 RMSE 변화 | 전체 MAE 변화 | tail RMSE 변화 | 전체 Time NLL 차이 |",
             "|---|---|---:|---:|---:|---:|"]
    for r in deltas:
        text.append(f"| {DATASETS[r['dataset']]} | {ARMS[r['arm']]} | {r['qty_rmse_change_percent']:+.2f}% | {r['qty_mae_change_percent']:+.2f}% | {r['tail_qty_rmse_change_percent']:+.2f}% | {r['time_nll_delta']:+.6f} |")
    text += ["", "- Taxi CNN+MLP16은 전체·큰 수량 RMSE와 MAE가 개선됐지만 전체 Time NLL은 악화됐습니다. GRU54는 전체 RMSE·MAE·Time NLL과 tail RMSE가 악화됐으며 tail MAE는 개선됐습니다.",
             "- RAF CNN+MLP16은 표의 전체·tail 지표가 모두 개선됐습니다. GRU54와 CNN+GRU54는 전체 RMSE가 소폭 악화되는 대신 MAE·Time NLL과 tail RMSE가 개선되는 교환 관계입니다.",
             "- Intermittent CNN+MLP16은 전체·tail RMSE와 Time NLL이 악화됐고 전체 MAE는 거의 같습니다. 다른 두 후보는 진행 중이므로 최종 결론에 넣지 않았습니다.",
             "- 단일seed·이종GPU 탐색입니다. 3seed 안정성, 통계적 우월성, 모든 데이터의 개선 또는 논문 효율 측정으로 해석하지 않습니다. CNN은 Encoder1 Q/K/V의 인과적 국소 경로를 추가합니다. GRU54는 중간 MLP 보정 전체를 대체하며 참조 범위·활성화·잔차 크기도 변경하므로 GRU 연산 하나의 효과로 단정하지 않습니다.", "",
             "**남은 세 조건의 학습을 기다린다 — 진행 중**", "", "아래 성능은 best-so-far이며 완료 결과와 구분합니다. ETA는 각 fit의 최근 최대10개 실제 완료 epoch 중앙값, 현재3병렬 속도 유지와 best 추가 개선 없음이라는 가정입니다. endpoint replay·회수·CPU 감사는 포함하지 않습니다.", "",
             "| 데이터 | 후보 | 저장 / best epoch | best RMSE | best MAE | best Time NLL | 초/epoch 중앙값 | 조건부 종료 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in pending:
        metric = r["validation_best_same_epoch"]
        text.append(f"| {DATASETS[r['job']['dataset']]} | {ARMS[r['job']['arm']]} | {r['saved_epoch']} / {r['best_epoch']} | {metric['qty_rmse']:.6f} | {metric['qty_mae']:.6f} | {metric['time_nll']:.6f} | {r['recent_epoch_seconds_median']:.1f} | {r['eta']['conditional_finish_kst']} |")
    manager_latest = provider["records"]["monitor/latest.json"]
    deadline = datetime.fromtimestamp(allocation["deadline_unix"], timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
    text += ["", f"- 승인 총상한$100과 별도로 실제 작업상한은 **${allocation['work_cap_usd']:.8f}**, 회수·정리 여유$20입니다. 관측된 경과시간×배정요율 비용은 **${manager_latest['own_elapsed_rate_cost_usd']:.2f}**이며 청구서 금액이 아닙니다.",
             f"- 현재 로컬 마감은 **{deadline}**입니다. 두 Intermittent의 현재 속도 ETA는 마감보다 늦으므로, 9조건이 이번 임대에서 모두 완료될지는 미확정입니다. Taxi 슬롯 종료 이후 속도 변화도 아직 관측되지 않았습니다. 자동 예산 확대·재개·추가 임대는 하지 않습니다.",
             f"- 회수 관리자 PID{provider['local_manager_process']['pid']}의 실제 명령이 일치하고 최근 관측까지 활동했습니다. 완료6조건의 terminal·소형 기록 SHA·selected/last Validation replay를 확인했습니다. 신규 checkpoint binary 원본 회수, CPU 재추론 감사, 최종 비용·Pod 삭제는 아직 완료되지 않았습니다.",
             "- 사용자 요청에 따라 기존 관리자를 유지하며 남은 학습과 원본 회수를 기다립니다. 마감 시 미완료 조건이 있으면 원본·중단 이력을 보존하고 후속 재개 여부를 별도 계약으로 검토합니다.", "",
             "**원본 근거와 검증 범위를 보존한다 — 완료**", "", "[정밀 수치 CSV](validation_completed_and_baseline.csv), [기준선 대비 seed42 변화](validation_paired_changes.csv), [완료 selected/last 기록](completed_selected_records.json), [진행 조건 원본 관측](in_progress_observation.json), [SHA·계약·모집단 검증](verification.json)을 연결했습니다. 보고서 생성은 저장된 단일 SSH 관측과 로컬 원본만 읽었고, 새 네트워크 호출·학습·평가는 수행하지 않았습니다."]
    (OUT / "README.md").write_text("\n".join(text) + "\n")
    print(json.dumps({"status": "passed", "completed": len(completed), "in_progress": len(pending), "observed_kst": latest["observed_kst"], "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
