"""Join already analysed local evidence; never run models or modify experiments."""
from pathlib import Path
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = {"yellow_trip_hourly": "Taxi", "raf_spare_parts": "RAF", "intermittent_frozen_5000": "Intermittent"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence_names = ["INTERNAL_ANALYSIS.json", "EXTERNAL_ANALYSIS.json", "INPUT_CONTEXT.json"]
    source = {name: json.loads((HERE / name).read_text()) for name in evidence_names}
    internal, external, context = [source[name] for name in evidence_names]
    assert internal["status"] == external["status"] == "complete"
    assert context["status"] == "ANALYZED"
    assert not internal["new_training"] and not internal["new_inference"]
    assert not external["new_training"] and not external["new_inference"]
    assert not context["new_training"] and not context["model_inference"]
    assert external["CNN_conditions_per_split"] == 9 and external["external_conditions_per_split"] == 54
    assert len(internal["training_histories"]) == 18

    internal_rows = []
    for dataset in NAMES:
        val = next(r for r in internal["summaries"] if r["dataset"] == dataset and r["split"] == "validation")
        test = next(r for r in internal["summaries"] if r["dataset"] == dataset and r["split"] == "test")
        decomposition = next(r for r in internal["reversal_decomposition"] if r["dataset"] == dataset)
        internal_rows.append({
            "dataset": dataset,
            "validation_FULL_minus_CNN_mean_RMSE": val["mean_RMSE_gap_FULL_minus_CNN"],
            "test_FULL_minus_CNN_mean_RMSE": test["mean_RMSE_gap_FULL_minus_CNN"],
            "validation_FULL_minus_CNN_mean_MAE": val["mean_MAE_gap_FULL_minus_CNN"],
            "test_FULL_minus_CNN_mean_MAE": test["mean_MAE_gap_FULL_minus_CNN"],
            "FULL_validation_RMSE_wins": val["FULL_RMSE_wins"],
            "FULL_test_RMSE_wins": test["FULL_RMSE_wins"],
            "test_tail_signed_share_of_mean_MSE_gap": test["tail_signed_share_of_mean_MSE_gap"],
            "test_mean_MSE_gap": test["mean_MSE_gap_FULL_minus_CNN"],
            "test_tail_mean_MSE_gap_contribution": test["bins"][-1]["MSE_gap_population_contribution"],
            "decomposition": decomposition,
            "selected_neighborhood_pm5_mean_median_RMSE": {
                arm: statistics.mean(r["neighborhood_pm5"]["RMSE_median"] for r in internal["training_histories"] if r["dataset"] == dataset and r["arm"] == arm)
                for arm in ["FULL", "CNN"]
            },
        })

    external_rows = []
    for row in external["comparisons"]:
        external_rows.append({k: row[k] for k in ["dataset", "external_arm", "metric_direction_reversals", "validation", "test", "decomposition_mean_of_seed_mse"]})

    population = []
    for row in context["rows"]:
        val, test = [row["profiles"][s] for s in ["validation", "test"]]
        population.append({
            "dataset": row["dataset"], "Train_fixed_thresholds": row["thresholds"],
            "validation_count": val["all_rows_count"], "test_count": test["all_rows_count"],
            "validation_mean_quantity": val["quantity"]["mean"], "test_mean_quantity": test["quantity"]["mean"],
            "test_validation_second_moment_ratio": row["Test_Validation_raw_quantity_second_moment_ratio"],
            "validation_tail_share": val["quantity_cells"][-1]["share"],
            "test_tail_share": test["quantity_cells"][-1]["share"],
            "validation_positive_gap_mean": val["positive_stored_delta_t"]["mean"],
            "test_positive_gap_mean": test["positive_stored_delta_t"]["mean"],
            "time_unit": row["time_unit"],
            "entity_overlap": row["Validation_Test_entity_intersection"],
            "validation_only_entities": row["Validation_only_entities"], "test_only_entities": row["Test_only_entities"],
            "per_entity_chronology_violations": row["per_entity_chronology_violations"],
            "global_calendar_ranges_overlap": row["global_date_ranges_may_overlap_across_entities"],
        })

    hypotheses = [
        {"hypothesis": "선택 checkpoint 또는 저장 결과 연결 오류", "finding": "대상 18개 fit의 최초 strict finite Validation raw RMSE 최소와 실제 선택 binary/tensor·평가 연결이 일치한다.", "status": "점검한 증적에서 오류 발견 없음", "causal_established": False},
        {"hypothesis": "큰 수량의 개수 비중이 바뀌어 반전", "finding": "내부 비교의 대칭 분해에서 구간 비중 항은 Taxi 1.98%, RAF 0.62%, Intermittent -34.20%; Taxi 외부 S2P2 대비는 0.47%다.", "status": "넓은 수량 구간 비중 변화만으로 설명되지 않음", "causal_established": False},
        {"hypothesis": "기간·개체·관측 이력·수량 조건 변화에 대한 민감성", "finding": "입력 분포 변화와 구간 내부 상대 오류 변화가 함께 관측된다. CNN/FULL 원시행 예측이 없어 같은 개체·시점 오류와 직접 연결할 수 없다.", "status": "가능한 설명, 기전 미확정", "causal_established": False},
        {"hypothesis": "log1p 학습 목적과 raw RMSE 선택·평가의 가중 차이", "finding": "대상 18/18 fit의 raw RMSE 선택 epoch와 Val log MSE 최소 epoch가 다르다. Train raw RMSE와 구간별 고정 모델 오류는 없다.", "status": "목적·선택 차이는 확인, 실패 원인으로서 기여는 미확정", "causal_established": False},
        {"hypothesis": "GRU 추가 또는 활성 파라미터 증가가 과적합 유발", "finding": "FULL의 Test 평균 악화는 확인됐지만 동일 활성 용량 대조가 없고 고정 모델 Train raw 오류도 없다. 순위만으로 과적합이나 GRU 기전을 입증할 수 없다.", "status": "미확정", "causal_established": False},
        {"hypothesis": "Validation 최솟값 선택의 불안정성", "finding": "Intermittent 선택점에서는 FULL 우세지만 ±5epoch 중앙값의 평균은 FULL 2.1467, CNN 2.0690이다. 최솟값이 이웃보다 좋은 것은 구조적으로 예상되며 선택 과적합 증거는 아니다.", "status": "선택 주변 변동성 확인, Test 실패 원인 미확정", "causal_established": False},
    ]

    output = {
        "schema": "stored_evidence_rank_reversal_diagnosis_v1", "status": "complete",
        "material_passport": {"origin_skill": "academic-research-suite / experiment-agent validate", "verification_status": "ANALYZED", "human_read_attested": False},
        "scope": "기존 로그·누적 구간 오류·외부의 저장 예측·고정 입력만 분석",
        "new_training": False, "new_inference": False, "remote_calls": 0,
        "checkpoint_deserialization": False, "checkpoint_reselection": False, "manuscript_changed": False,
        "test_is_new_independent_evaluation": False, "causal_mechanism_established": False,
        "source131_closure": internal["source131_closure"],
        "internal_comparisons": internal_rows, "external_comparisons": external_rows,
        "input_context": population, "hypothesis_ledger": hypotheses,
        "sources": {str(HERE / name): sha(HERE / name) for name in evidence_names},
        "source_report_hashes": {str(HERE / name): sha(HERE / name) for name in ["INTERNAL_ANALYSIS.md", "EXTERNAL_ANALYSIS.md", "INPUT_CONTEXT.md"]},
        "script_sha256": sha(Path(__file__)),
        "limitations": [
            "오류 분해는 산술적인 위치 설명이며 인과적 원인 기여율이 아니다.",
            "평균 편향은 예측-정답의 평균이다. 모든 행의 과소·과대 예측이나 건수 비율은 아니다.",
            "구간 내부의 값·이력·개체·기간 분포 변화는 넓은 수량 구간 비중 항에 포함되지 않는다.",
            "CNN/FULL의 원시행 예측이 없어 날짜·개체·개별 이력 조건별 같은 행 오류 분석은 불가능하다.",
            "외부 Validation 원시 truth 지문·signed error가 없다. 선택 tensor·대상 수·구간 수·고정 입력 계약까지만 확인했다.",
            "Train 로그는 업데이트 중 train mode의 온라인 log1p MSE이며 고정 선택 모델의 Train raw RMSE가 아니다.",
            "활성 파라미터 차이와 GRU 구조 효과를 구분할 동일 활성 용량 대조가 없다.",
            "외부 비교는 공통 head를 적용한 과거 encoder 캠페인과 source131 CNN의 비교이며 같은 Runtime 재학습 대조가 아니다.",
            "Intermittent 42/52는 5090/cuDNN92000, 62는 5080/cuDNN91900이다. 같은 seed의 구조는 같은 host이며 cross-seed 시간은 합치지 않는다.",
            "Taxi·Intermittent는 개체별 시간 순 분할이며 모든 개체에 공통인 달력 경계의 미래 기간 평가가 아니다. 개체별 순서 위반 0만으로 모든 정보 누출 가능성을 배제하지 않는다.",
            "3seed의 기술 통계·사후 진단이다. 유의성·새 독립 Test 검증·구조의 보편적 우월성으로 확대하지 않는다.",
            "이전 대표 FULL·보정27·Instacart9와 새 구성요소 FULL/CNN을 합치거나 좋은 값으로 교체하지 않았다.",
        ],
        "next_steps": [
            {"status": "다음 작업", "subject": "근본 원인 구분에 필요한 최소 자료의 수집 계약을 정한다", "details": "추가 학습 없이 고정 선택 모델의 Train raw 오차와 날짜·개체·이력별 같은 행 오류를 확인할 자료 범위를 정한다. 기존 저장 증적 분석은 완료됐다."},
            {"status": "승인 필요", "subject": "추가 진단이 필요하면 학습 없이 고정 모델 추론 범위를 정한다", "details": "선택 checkpoint를 유지한 Train/Validation/Test 행별 예측 저장 및 동일 행 날짜·개체·이력별 오류 비교는 현재 실행하지 않았다. 재추론 범위와 증적 보존 계약을 별도로 확정한다."},
            {"status": "다음 작업", "subject": "확인된 오류 양상과 미확정 원인을 논문에 구분해 반영한다", "details": "현재 결과는 구간별 오류·seed 예외와 한계로 기록한다. 추가 자료를 얻더라도 인과적 기전이 확인되지 않으면 그 미확정을 보존한다. 원고 수정은 후속 작업이다."},
        ],
    }

    lines = [
        "# Validation→Test 순위 반전: 기존 증적 진단",
        "",
        "**오류가 커진 구간과 예측 방향은 확인했다. 과적합·GRU 기전·분포 변화 중 어느 것이 근본 원인인지는 기존 저장 자료만으로 확정하지 못했다.** 두 비교는 서로 다른 결론을 갖는다.",
        "",
        "## Material Passport와 현재 기준선",
        "",
        "Origin Skill: academic-research-suite / experiment-agent validate. Verification Status: **ANALYZED**. 사용자에게 읽음 확인을 받았다는 표시를 하지 않는다.",
        "",
        "대상은 source131의 FULL/CNN 18개 fit, 두 split의 36개 결과와 기존 외부 6개×3데이터셋×3seed의 54개 모델이다. 외부 Test54의 저장 예측을 재집계했다. 새 학습·추론·SSH·checkpoint 역직렬화·모델/분할/선택 변경·원고 수정은 모두 0이다. 원래 여섯 평가 지표와 다른 캠페인의 결과는 기존 파일에 그대로 보존했다.",
        "",
        "여기서 CNN은 TitanTPP의 인과적 attention·정적 prototype·persistent token에 CNN 보정만 적용한 구조다. 독립적인 단순 CNN 모델이 아니다. FULL은 같은 구성에 GRU 보정을 추가한 CNN+GRU다. 외부는 공통 평가 head를 적용한 encoder 비교군이다.",
        "",
        "## 1. CNN+GRU 대 CNN — 완료",
        "",
        "차이=FULL−CNN. 음수는 FULL이 좋다. 모두 같은 seed의 Validation 선택 checkpoint를 유지한 결과다.",
        "",
        "| 데이터셋 | Validation RMSE 차이 | Test RMSE 차이 | Validation MAE 차이 | Test MAE 차이 | FULL RMSE 승리 Val→Test | 큰 수량의 Test MSE 격차 기여 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in internal_rows:
        lines.append(f"| {NAMES[row['dataset']]} | {row['validation_FULL_minus_CNN_mean_RMSE']:+.4f} | {row['test_FULL_minus_CNN_mean_RMSE']:+.4f} | {row['validation_FULL_minus_CNN_mean_MAE']:+.4f} | {row['test_FULL_minus_CNN_mean_MAE']:+.4f} | {row['FULL_validation_RMSE_wins']}/3→{row['FULL_test_RMSE_wins']}/3 | {100*row['test_tail_signed_share_of_mean_MSE_gap']:.2f}% |")
    lines += [
        "",
        "이 기여율은 3seed 평균 MSE 차이 중 해당 구간의 누적 제곱오차 차이가 차지하는 signed 비율이다. RMSE 개선율이나 오류 건수 비율이 아니다. 평균 RMSE를 제곱하거나 구간 RMSE를 더해서 계산하지 않았다.",
        "",
        "- **Taxi:** Test `q>3449`의 평균 편향(예측−정답)은 FULL −925.57, CNN +68.11이다. FULL은 이 구간에서 세 seed 모두 순 과소 예측이며 구간 MSE도 세 seed 모두 CNN보다 크다. 전체 격차 +7054.45 중 +5929.33이 이 구간에서 발생한다. 다만 seed62는 다른 구간의 개선이 이를 상쇄해 전체 RMSE는 FULL이 8.621 낮다. Validation에서도 큰 수량은 FULL이 불리했지만 다른 구간의 이득으로 전체 평균이 좋았다.",
        "- **RAF:** `q>200` 구간은 Validation에서 FULL에 유리했으나 Test에서는 불리해졌다. Test의 평균 MSE 격차 +23.626 중 +18.407이 큰 수량 구간이다. FULL과 CNN 모두 이 구간에서 과소 예측하며, 전체 차이는 작고 seed별 방향도 다르다. FULL이 모든 seed에서 실패했다는 결론은 아니다.",
        "- **Intermittent:** `q>187`이 전체 격차에서 차지하는 비중은 4.94%다. `(46,187]` 구간의 기여 +0.971은 전체 +1.683의 약 58%다. 이 구간 평균 편향은 FULL +6.77, CNN +3.70으로 FULL의 순 과대 예측이 더 크다. Taxi의 큰 수량 과소 예측과 같은 양상으로 설명할 수 없다.",
        "",
        "## 2. CNN 대 외부 비교군 — 완료",
        "",
        "차이=CNN−외부. 음수는 CNN이 좋다. 전체 6개 비교는 별도 외부 분석 표에 모두 있으며, 여기서는 평균 순위가 바뀐 기준과 유지된 경우를 구분한다.",
        "",
        "| 데이터셋·비교 | Validation RMSE 차이 | Test RMSE 차이 | Validation MAE 차이 | Test MAE 차이 | 해석 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    interpretation = {
        ("yellow_trip_hourly", "s2p2_matched_head"): "평균 반전; seed42·62 반전, 52 우세 유지",
        ("yellow_trip_hourly", "attnhp_matched_head"): "평균 반전; seed42·62 반전",
        ("raf_spare_parts", "s2p2_matched_head"): "RMSE는 CNN 개선 방향으로 세 seed 반전",
        ("intermittent_frozen_5000", "s2p2_matched_head"): "CNN의 RMSE·MAE 우세 유지",
    }
    for row in external["comparisons"]:
        key = row["dataset"], row["external_arm"]
        if key not in interpretation:
            continue
        # Metrics in these summaries retain the exact source schema.
        val, test = row["validation"], row["test"]
        rm = "overall_qty_rmse"
        ma = "overall_qty_mae"
        lines.append(f"| {NAMES[key[0]]} / {key[1]} | {val[rm]['mean_difference']:+.4f} | {test[rm]['mean_difference']:+.4f} | {val[ma]['mean_difference']:+.4f} | {test[ma]['mean_difference']:+.4f} | {interpretation[key]} |")
    lines += [
        "",
        "Taxi는 다른 네 외부 모델보다 Test 평균 RMSE·MAE가 여전히 낮다. 개별 seed의 반전도 외부 전체 표에 보존했다. RAF는 Test 평균 RMSE가 외부 여섯 모델보다 낮지만, MAE는 RMTPP·NHP보다 높다. Intermittent는 두 split에서 외부 여섯 모델보다 평균 RMSE·MAE가 낮다. 따라서 ‘CNN도 모든 데이터셋에서 Validation→Test 우위를 잃었다’는 설명은 맞지 않는다.",
        "",
        "**Taxi 대 S2P2의 반전은 `q>1562` 두 구간에 집중된다.** 평균 seed MSE 차이의 Validation→Test 변화 +9388.830 중 89.86%가 이 구간에서 발생한다. `q>3449`만으로는 40.8%다. 이는 앞 표의 FULL−CNN **Test 격차 84.05%**와 비교 대상·분모가 다르다. Taxi CNN의 큰 수량 편향은 seed42 +348.7, seed52 −240.0, seed62 +95.6이다. FULL처럼 세 seed 공통의 순 과소 예측으로 외부 대비 실패를 설명할 수 없다.",
        "",
        "## 3. 입력 변화와 학습 로그에서 확인한 사실 — 완료",
        "",
        "| 데이터셋 | 큰 수량 비중 Validation→Test | 수량 평균 Validation→Test | Test 수량 제곱평균의 변화 | 입력 맥락 |",
        "|---|---|---|---:|---|",
    ]
    for row in population:
        note = {"yellow_trip_hourly": "같은 131개 개체; 양의 gap 평균 1.779→1.846시간", "raf_spare_parts": "평가 개체 구성이 달라짐; gap 평균은 약 10.08로 비슷", "intermittent_frozen_5000": "같은 5000개 개체; 양의 gap 평균 2.767→3.624"}[row["dataset"]]
        lines.append(f"| {NAMES[row['dataset']]} | {100*row['validation_tail_share']:.4f}%→{100*row['test_tail_share']:.4f}% | {row['validation_mean_quantity']:.3f}→{row['test_mean_quantity']:.3f} | {100*(row['test_validation_second_moment_ratio']-1):+.2f}% | {note} |")
    lines += [
        "",
        "수량 제곱평균은 정답 규모의 맥락 지표이며 모델 MSE가 아니다. RAF는 큰 수량의 비중이 거의 같아도 수량 제곱평균과 평가 개체 구성이 달라졌다. Intermittent는 수량이 낮아지고 gap이 길어졌다. 이런 관측은 입력 조건 변화가 있음을 보여주지만, 그 변화가 특정 모델 오류를 얼마나 유발했는지는 확인하지 못했다.",
        "",
        "- **선택·저장 오류:** FULL/CNN 18/18의 선택 epoch가 최초 strict finite Validation raw RMSE 최소와 일치한다. 모든 저장 학습 epoch는 finite이며 Val/Test의 binary·tensor·대상 지문 연결도 일치한다. 확인한 증적에서 선택·저장 오류를 발견하지 못했다.",
        "- **학습 목적과 선택 기준:** 18/18에서 Validation log MSE 최소 epoch와 raw RMSE 최소 epoch가 다르다. 학습은 log1p 수량 MSE, 선택·주 평가는 raw RMSE여서 큰 수량에 대한 가중이 다르다. 이 차이를 확인했지만 반전의 인과적 원인으로 입증한 것은 아니다.",
        "- **학습 곡선:** 선택 이후 Train 온라인 log MSE는 감소하고 Validation raw RMSE는 악화한 fit이 18/18이다. 마지막 모델 대신 선택 모델로 Test를 평가했다. 온라인 train mode 손실과 고정 eval mode의 raw RMSE를 비교해 과적합을 확정할 수는 없다.",
        "- **Intermittent 선택 주변:** 선택점의 평균 RMSE는 FULL 1.7005, CNN 1.7230이지만, 각 선택점 ±5epoch Validation RMSE 중앙값의 평균은 FULL 2.1467, CNN 2.0690이다. 작은 선택점 우위가 주변에서는 유지되지 않았다. 최솟값과 이웃의 차이 자체는 예상되는 현상이며, 이것을 선택 과적합의 증명으로 보거나 checkpoint를 바꾸지 않았다.",
        "",
        "## 4. 무엇이 설명됐고 무엇이 아직 남았는가",
        "",
        "구간별 누적 SSE를 전체 대상 수로 나눈 값의 합으로 MSE 차이를 복원했다. 이어 동일한 Train 고정 구간을 사용해 `D=Σp·d`의 변화를 대칭식 `ΣΔp·(dT+dV)/2 + Σ(pT+pV)/2·Δd`로 분해했다. 이 식은 넓은 수량 구간 비중 변화와 구간 안의 상대 오류 변화의 **산술 분해**다.",
        "",
        "| 비교 | 구간 비중 변화 항의 signed 기여 | 의미 |",
        "|---|---:|---|",
        "| FULL−CNN / Taxi | 1.98% | 큰 수량이 조금 더 많아진 것만으로 반전 설명 불가 |",
        "| FULL−CNN / RAF | 0.62% | 같은 수량 구간 내부에서 상대 오류가 달라짐 |",
        "| FULL−CNN / Intermittent | −34.20% | 비중 변화는 반전을 완화하는 방향; 구간 내 악화가 이를 넘음 |",
        "| CNN−S2P2 / Taxi | 0.47% | 구간 내 상대 오류 변화가 핵심 |",
        "",
        "작은 비중 항은 ‘분포 변화가 없다’는 뜻이 아니다. 구간 내부의 수량·이력·개체·기간 변화는 분리하지 못했다. GRU가 과적합을 일으켰다는 결론, 또는 데이터 변화가 유일한 원인이라는 결론 모두 현재는 미확정이다.",
        "",
        "## 5. 분석과 논문 서술에 필요한 한계",
        "",
    ]
    lines += [f"{i}. {value}" for i, value in enumerate(output["limitations"], 1)]
    lines += [
        "",
        "**통계 해석:** 평균·표본 SD(ddof=1)와 seed별 결과는 각 분석 JSON에 있다. 여기의 순위 변화는 기술 통계다. 가설 검정·신뢰구간·다중 비교 보정을 수행하지 않았으며, 통계적 유의성이나 모집단의 확정적 효과를 주장하지 않는다. 기존 Test를 이미 본 뒤의 진단이므로 추가 모델 선택 근거로 사용하지 않았다. 11개 통계·인과 해석 오류 점검은 FINAL_QA에 별도로 기록한다.",
        "",
        "## 6. 남은 작업 순서",
        "",
    ]
    for step in output["next_steps"]:
        lines += [f"**{step['subject']} — {step['status']}**", "", f"- {step['details']}", ""]
    lines += [
        "## 증적과 재현",
        "",
        f"- [내부 비교·구간 오류·18개 학습 로그]({HERE / 'INTERNAL_ANALYSIS.md'}), [수치와 원본 SHA]({HERE / 'INTERNAL_ANALYSIS.json'})",
        f"- [외부 6개 비교·저장 예측 재집계]({HERE / 'EXTERNAL_ANALYSIS.md'}), [수치와 원본 SHA]({HERE / 'EXTERNAL_ANALYSIS.json'})",
        f"- [고정 입력의 분포 맥락]({HERE / 'INPUT_CONTEXT.md'}), [수치와 입력 SHA]({HERE / 'INPUT_CONTEXT.json'})",
        f"- [독립 QA와 11개 해석 오류 점검]({HERE / 'FINAL_QA.md'})",
        "",
        "위 링크는 이 새 보고서 디렉터리의 파일이다. DIAGNOSIS.json에는 세 입력 분석 파일의 절대경로와 raw SHA, 가설별 확인 상태, 실행하지 않은 작업을 기록했다. 기존 캠페인·학습·claim·Runtime·원고는 변경하지 않았다.",
    ]
    (HERE / "DIAGNOSIS.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (HERE / "DIAGNOSIS.md").write_text("\n".join(lines) + "\n")
    for name in ["DIAGNOSIS.json", "DIAGNOSIS.md"]:
        print(name, sha(HERE / name))


if __name__ == "__main__":
    main()
