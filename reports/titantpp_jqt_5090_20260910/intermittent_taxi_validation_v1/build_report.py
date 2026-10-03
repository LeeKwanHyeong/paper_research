"""Extract two completed datasets from saved JSON evidence, without model execution."""

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SNAPSHOT = ROOT / "monitor/20260910T105909Z_snapshot.json"
ARMS = ("joint", "quantity_only", "time_only")
DATASETS = ((1, "intermittent_frozen_5000", "Intermittent"),
            (2, "yellow_trip_hourly", "Taxi"))
EXPOSURE = ("epoch", "global_step", "train_batch_order_sha256", "train_count",
            "train_batches", "validation_count", "validation_batches")
KST = timezone(timedelta(hours=9))


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def comparison(joint, single, relative=True):
    return {"J": joint, "single_task": single, "single_minus_J": single - joint,
            "change_percent": (single / joint - 1) * 100 if relative else None}


def change_text(value):
    delta = value["single_minus_J"]
    direction = "악화" if delta > 0 else "개선" if delta < 0 else "동일"
    if value["change_percent"] is None:
        return f"{delta:+.6f} · {direction}"
    return f"{value['change_percent']:+.2f}% · {direction}"


def main():
    raw = SNAPSHOT.read_bytes()
    envelope = json.loads(raw)
    files = envelope["snapshot"]["files"]
    suite = files["suite_contract.json"]["json"]
    status = files["suite_run/status.json"]["json"]
    assert all(envelope["identity_checks"].values())
    assert suite == json.loads((ROOT / "remote_receipts/suite_contract.json").read_text())
    assert status["held_out_test_evaluated"] is False
    datasets = []
    for index, dataset_id, label in DATASETS:
        prefix = f"suite_run/{index:02d}_{dataset_id}/"
        contract = files[f"contracts/{dataset_id}.json"]["json"]
        job = next(job for job in status["jobs"] if job["dataset_id"] == dataset_id)
        manifest = files[prefix + "wrapper_manifest.json"]["json"]
        paired = files[prefix + "paired_comparison_epoch_000120.json"]["json"]
        summaries = {arm: files[prefix + arm + "/summary.json"]["json"] for arm in ARMS}
        histories = {arm: files[prefix + arm + "/history.json"]["json"]["history"] for arm in ARMS}
        train_n = contract["populations"]["train"]["target_count"]
        val_n = contract["populations"]["validation"]["target_count"]
        train_batches = math.ceil(train_n / 128)
        val_batches = math.ceil(val_n / 128)
        assert job["status"] == "complete" and job["returncode"] == 0
        assert job["recomputed_equal_exposure_verified"] is True
        assert contract["seed"] == 42 and contract["epochs"] == 120
        assert manifest["evaluation_scope"] == "validation_only"
        assert manifest["held_out_materialized"] is False
        assert manifest["contract"] == contract
        assert paired["equal_training_exposure_verified"] is True
        assert paired["global_step"] == train_batches * 120
        assert job["wrapper_manifest_sha256"] == files[prefix + "wrapper_manifest.json"]["sha256"]
        assert job["paired_comparison_sha256"] == files[prefix + "paired_comparison_epoch_000120.json"]["sha256"]
        for arm in ARMS:
            summary, history = summaries[arm], histories[arm]
            assert summary["status"] == "complete"
            assert summary["epochs_completed"] == summary["epochs_budget"] == len(history) == 120
            assert summary["global_step"] == train_batches * 120
            assert summary["history"] == history and finite(history)
            assert summary["evaluation_scope"] == "validation_only"
            assert summary["held_out_test_evaluated"] is False
            assert summary["initial_state_sha256"] == manifest["initial_state_sha256"]
            assert job["arms"][arm]["summary_sha256"] == files[prefix + arm + "/summary.json"]["sha256"]
            assert files[prefix + arm + "/contract.json"]["json"]["identity"]["contract"] == contract
            for epoch, row in enumerate(history, 1):
                assert (row["epoch"], row["global_step"], row["train_count"], row["train_batches"],
                        row["validation_count"], row["validation_batches"]) == (
                    epoch, epoch * train_batches, train_n, train_batches, val_n, val_batches)
                assert all(row[field] == histories["joint"][epoch - 1][field] for field in EXPOSURE)
            for metric, selector in summary["selectors"].items():
                if selector["applicable"]:
                    best = min(history, key=lambda row: (row[metric], row["epoch"]))
                    assert (selector["best_value"], selector["best_epoch"], selector["global_step"]) == (
                        best[metric], best["epoch"], best["global_step"])
                else:
                    assert all(row[metric] is None for row in history)
                    assert selector["best_epoch"] is None and selector["best_value"] is None
        selected = {}
        for task, other, metric in (("quantity", "quantity_only", "raw_quantity_rmse"),
                                     ("time", "time_only", "legacy_time_loss")):
            pair = paired["paired_comparisons"][task]
            for key, arm in (("joint", "joint"), ("single_task", other)):
                assert all(pair[key][field] == value for field, value in summaries[arm]["selectors"][metric].items())
            j_epoch = pair["joint"]["best_epoch"]
            s_epoch = pair["single_task"]["best_epoch"]
            j_row, s_row = histories["joint"][j_epoch - 1], histories[other][s_epoch - 1]
            value = comparison(j_row[metric], s_row[metric], task == "quantity")
            assert value["single_minus_J"] == pair["single_task_minus_joint"]
            selected[task] = {"J_epoch": j_epoch, "single_epoch": s_epoch, "metric": metric,
                              "comparison": value, "J_state_sha256": pair["joint"]["state_sha256"],
                              "single_state_sha256": pair["single_task"]["state_sha256"]}
            if task == "quantity":
                selected[task]["mae_at_rmse_selector"] = comparison(j_row["quantity_mae"], s_row["quantity_mae"])
        same_epoch = []
        for i in range(120):
            j, q, t = (histories[arm][i] for arm in ARMS)
            same_epoch.append({"epoch": i + 1, "global_step_per_arm": j["global_step"],
                               "raw_quantity_rmse": comparison(j["raw_quantity_rmse"], q["raw_quantity_rmse"]),
                               "quantity_mae": comparison(j["quantity_mae"], q["quantity_mae"]),
                               "legacy_time_loss": comparison(j["legacy_time_loss"], t["legacy_time_loss"], False)})
        clipping = {}
        for arm, history in histories.items():
            count = sum(row["train_clipped_batch_count"] for row in history)
            clipping[arm] = {"count": count, "total_batches": train_batches * 120,
                             "percent": count / (train_batches * 120) * 100,
                             "first_10_epoch_percent": sum(row["train_clipped_batch_count"] for row in history[:10]) / (train_batches * 10) * 100,
                             "last_10_epoch_percent": sum(row["train_clipped_batch_count"] for row in history[-10:]) / (train_batches * 10) * 100,
                             "by_epoch": [row["train_clipped_batch_count"] for row in history]}
        datasets.append({"dataset_id": dataset_id, "label": label,
                         "finished_at_kst": datetime.fromtimestamp(job["finished_at_unix"], KST).isoformat(),
                         "train_targets_per_epoch": train_n, "validation_targets": val_n,
                         "train_batches_per_epoch": train_batches, "validation_batches": val_batches,
                         "steps_per_arm": train_batches * 120, "audit_status": "passed",
                         "initial_state_sha256": manifest["initial_state_sha256"],
                         "selected_checkpoint_comparison": selected,
                         "same_epoch_comparison": same_epoch, "clipping": clipping,
                         "evidence_sha256": {path: value["sha256"] for path, value in files.items()
                                             if path.startswith(prefix) and "sha256" in value}})
    result = {"assessment": "Share with caveats", "scope": "seed42 validation-only, static B J/Q/T, two completed datasets",
              "source_revision": suite["source"]["revision"], "source_snapshot": str(SNAPSHOT),
              "source_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
              "observed_at_kst": datetime.fromisoformat(envelope["snapshot"]["observed_at_utc"]).astimezone(KST).isoformat(),
              "independent_review": "monitor_schema_review: primary selector/receipt audits from completed checks, plus independent epoch120 and clipping recomputation",
              "datasets": datasets, "new_model_evaluation": False, "held_out_evaluated": False,
              "metric_notes": {"quantity": "original quantity scale, lower is better; percentage=(single/J-1)*100",
                               "time": "legacy clamped loss, not normalized NLL; report absolute delta only",
                               "selected_mae": "MAE at each raw-RMSE-selected checkpoint, not MAE-selected",
                               "same_epoch": "all 120 paired boundaries retained; final epoch120 is separate from primary selectors"}}
    (OUT / "validation_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Intermittent·Taxi J/Q/T 검증 결과", "",
             "**종합 판단 — 조건과 한계를 명시하면 공유 가능(Share with caveats)**", "",
             "두 데이터셋·6개 학습의 완료 증적과 비교 계산은 검증을 통과했다. 사전 지정된 최적 validation selector 비교에서는 두 데이터셋 모두 Q의 수량 RMSE와 T의 시간 loss가 J보다 나빴다. Intermittent의 수량 MAE만 1.07% 좋아졌다. 다만 **Taxi는 동일 최종 epoch120에서 Q·T가 J보다 좋으므로, 단독 학습이 언제나 나쁘다고 해석해서는 안 된다.**", "",
             "대상은 이번 5090에서 재학습한 static B의 J(공동)·Q(수량 전용)·T(시간 전용)다. 과거 B 결과표나 dual-timescale·RMTPP·THP 결과를 비교 분모에 섞지 않았다. 증적 기준 시각은 **2026-09-10 19:59:09 KST**다.", "",
             "**학습 계약과 완료 증적 확인 — 완료**", "",
             "공통 조건: seed42, arm당120 epoch, batch128, AdamW lr0.001·weight decay0.01·clip1, 수량 log-MSE 가중치1, legacy time cap300. 데이터셋 안에서 동일 초기화·batch 순서·표본 노출을 검증했다. 동일 step은 동일 계산 비용이나 각각 최적으로 조정한 학습률을 뜻하지 않는다.", "",
             "| 데이터셋 | 완료 arm | train/epoch | validation | step/arm | 종료 KST |",
             "|---|---:|---:|---:|---:|---|"]
    for d in datasets:
        lines.append(f"| {d['label']} | 3/3 | {d['train_targets_per_epoch']:,} | {d['validation_targets']:,} | {d['steps_per_arm']:,} | {d['finished_at_kst'][11:19]} |")
    lines += ["", "source revision은 `76426fd981e624144947fd582f794c2832e9fe05`다. 완료 receipt와 summary·wrapper·paired comparison SHA, history 일치, 전체120 epoch의 step·표본·순서, 활성 selector의 strict earliest minimum을 확인했다. 비활성 지표는 null로 보존했다. 원본 표본에서 성능을 다시 계산하거나 checkpoint를 다시 실행한 검증은 아니다.", "",
              "**최적 validation checkpoint 비교 — 완료, 사전 지정된 주 비교**", "",
              "수량은 J와 Q 각각 raw-RMSE selector로 고른 checkpoint를 비교한다. MAE도 그 동일 checkpoint에서 읽는다. 시간은 J와 T 각각 시간 selector를 적용한다. 수량·시간의 J checkpoint가 다르므로 하나의 모델이 동시에 달성한 성적으로 합치지 않는다.", "",
              "| 데이터셋 | 지표 | J | Q 또는 T | 단독 학습의 J 대비 변화 | 선택 epoch J / 단독 |",
              "|---|---|---:|---:|---|---|"]
    for d in datasets:
        q = d["selected_checkpoint_comparison"]["quantity"]
        t = d["selected_checkpoint_comparison"]["time"]
        for metric, arm, value, selector in (("수량 Raw RMSE", "Q", q["comparison"], q),
                                              ("수량 MAE", "Q", q["mae_at_rmse_selector"], q),
                                              ("Legacy time loss", "T", t["comparison"], t)):
            lines.append(f"| {d['label']} | {metric} | {value['J']:.6f} | {arm} {value['single_task']:.6f} | {change_text(value)} | {selector['J_epoch']} / {selector['single_epoch']} |")
    lines += ["", "**동일 최종 epoch120 비교 — 완료, 보조 비교**", "",
              "두 학습을 같은 최종 step에서 비교한다. 성능을 본 뒤 주 selector를 바꾼 결과가 아니다. 모든120 epoch의 대응 수치는 JSON에 보존했다.", "",
              "| 데이터셋 | 지표 | J epoch120 | Q/T epoch120 | 단독 학습의 J 대비 변화 |",
              "|---|---|---:|---:|---|"]
    for d in datasets:
        row = d["same_epoch_comparison"][-1]
        for metric, label, arm in (("raw_quantity_rmse", "수량 Raw RMSE", "Q"),
                                    ("quantity_mae", "수량 MAE", "Q"),
                                    ("legacy_time_loss", "Legacy time loss", "T")):
            value = row[metric]
            lines.append(f"| {d['label']} | {label} | {value['J']:.6f} | {arm} {value['single_task']:.6f} | {change_text(value)} |")
    lines += ["", "Intermittent는 최종epoch에서 Q의 RMSE·MAE와 T의 시간 loss가 모두 J보다 나쁘다. 최적 수량 checkpoint에서의 MAE 개선은 최종epoch에서는 유지되지 않는다. Taxi는 최종epoch에서 Q의 RMSE·MAE와 T의 시간 loss가 모두 J보다 좋다. 그러나 Taxi의 J·T 시간 loss는 각각의 최적값보다 크게 높아져, 최적 성능 비교와 후반 학습 안정성을 함께 살펴야 한다.", "",
              "**학습 중 clipping 집계 — 완료, 원인 해석에는 제한**", "",
              "| 데이터셋 | J 횟수·비율 | Q 횟수·비율 | T 횟수·비율 | 전체 batch/arm |",
              "|---|---:|---:|---:|---:|"]
    for d in datasets:
        cells = [f"{d['clipping'][arm]['count']:,} · {d['clipping'][arm]['percent']:.4f}%" for arm in ARMS]
        lines.append(f"| {d['label']} | {' | '.join(cells)} | {d['steps_per_arm']:,} |")
    lines += ["", "Intermittent의 J·T clipping은 거의 초기6 epoch에 집중된다(J epoch108의1건 예외). Taxi는 세 arm 모두120개 epoch 전체에서 clipping이 발생한다. Taxi 마지막10 epoch의 비율은 J95.57%, Q3.33%, T92.63%다. Taxi T도 후반 clipping이 높기 때문에 공동 학습만의 문제로 확정할 수 없다.", "",
              "clipping은 실제 clip_grad_norm_ 호출 전 norm이 임계값1을 초과한 batch 횟수다. 각 arm의 학습 경로가 달라지므로 횟수 차이는 시간 gradient의 인과 효과나 실제 AdamW update 억제량이 아니다. 선택 checkpoint의 과거 train-only gradient probe와 이번 전체 학습 이력도 구분한다.", "",
              "**해석 검토와 남은 한계 — 검토 완료**", "",
              "- 중요한 해석 보정: 기존의 ‘단독 학습의 이득 없음’은 최적 selector의 주 비교 지표에 한정한다. Taxi의 최종epoch 이득과 Intermittent의 선택 checkpoint MAE 이득을 함께 제시해야 한다.",
              "- 확인된 사실: 두 데이터셋 모두 주 selector 기준 수량 RMSE·시간 loss의 분리 이득은 없다. Taxi의 최종epoch에서는 비교 방향이 바뀐다.",
              "- 해석: 현재 B에서 학습 간섭이 공통의 핵심 병목이고 완전 분리가 해결책이라는 주장은 지지되지 않았다. 그렇다고 간섭의 부재나 모든 분리 구조의 열세를 입증한 것은 아니다.",
              "- 미확인: Body/tail MAE, 수량 구간·이력 길이별 오차, 구간별 task gradient 방향, 여러 seed의 재현성, held-out 성능, 별도 최적화 조건에서의 수렴 성능이다.",
              "- Q와 T를 결합한 시스템의 용량·학습/추론 비용·동시 성능과 benchmark 우월성은 검증하지 않았다. J의 수량/시간 최적 checkpoint도 서로 다르다.",
              "- validation은 checkpoint 선택과 기술적 비교에 사용했다. 표본수는 독립 반복실험 수가 아니며 통계적 유의성·보편적 우월성 주장은 하지 않는다. 시간 지표는 정규화된 Time NLL이 아니다.", "",
              "**남은 작업 순서**", "",
              "**Instacart 완료 결과를 같은 기준으로 통합 — 외부 작업 대기**", "",
              "- 승인된 기존 학습을 시간별 scheduler로 관찰하고 완료 증적이 확보되면 주 selector와 동일 최종step 비교를 추가한다.", "",
              "**두 데이터셋의 오차·후반 안정성 원인 확인 범위 확정 — 다음 작업**", "",
              "- Intermittent는 수량 구간·이력 길이별 RMSE/MAE 차이, Taxi는 후기 시간 loss와 clipping의 동반 변화에 초점을 둔다. 필요한 표본별 예측과 학습 중 gradient가 없으므로 원인을 완료된 것처럼 보고하지 않는다.", "",
              "이번 추출의 차단 요소는 없다. 새로운 평가 실행은 이 보고서에 포함하지 않았다. 기존 학습·원본 결과·scheduler는 변경하지 않았다.", "",
              "**재현 및 원본 증적**", "",
              "- [검증 결과 JSON](validation_results.json): 전체120 epoch 대응 수치, 지표 정의, clipping, 증적 SHA.",
              "- [추출·검증 코드](build_report.py): 표준 라이브러리만 사용하며 원격·모델·원본 데이터에 접근하지 않는다.",
              "- [19:59 원격 스냅샷](../monitor/20260910T105909Z_snapshot.json), [기존 완료 감사](../monitor/20260910T105909Z_check.json), [동결 실행 계약](../remote_receipts/suite_contract.json).", "",
              "재계산 명령:", "", "```bash", f"python3 {OUT / 'build_report.py'}", "```", ""]
    (OUT / "validation_report.md").write_text("\n".join(lines))
    print(json.dumps({"assessment": result["assessment"], "completed_arms": 6,
                      "report": str(OUT / "validation_report.md"),
                      "datasets": [{"dataset": d["label"], "audit": d["audit_status"],
                                    "final_epoch": d["same_epoch_comparison"][-1]} for d in datasets]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
