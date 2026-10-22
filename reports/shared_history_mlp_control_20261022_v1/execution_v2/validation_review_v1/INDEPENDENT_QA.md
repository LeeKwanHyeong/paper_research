# 표 산술·출처검증

## Review Result: PASS

Validation 원예측 27개, 911,187행을 프로젝트 집계 함수를 사용하지 않고 독립 재계산했습니다. 18개 데이터·모델·범위 행의 54개 평균·표본SD, 162개 seed별 절대·상대차이, 54개 평균차이·평균비, 모든 seed 악화 위치가 REPORT JSON/Markdown 및 CSV와 일치합니다.

- 방법: Python 표준 라이브러리의 gzip/json, math.fsum, statistics.mean/stdev(ddof=1).
- 원예측 SHA, Validation split, 동일 target 순서·수량·history length, Train에서 고정한 strict q > cutoff를 확인했습니다.
- receipt와 독립 계산의 최대 절대차이: 1.4210854715202004e-14.
- REPORT record와 독립 계산의 최대 절대차이: 2.984279490192421e-13. 합산 순서의 부동소수점 반올림 수준이며 방향·악화 seed는 모두 일치합니다.
- 원본 54개 receipt/prediction SHA와 검토 산출물 7개 SHA는 INDEPENDENT_QA.json에 기록했습니다. REPORT의 증거 SHA 83개도 실제 파일과 일치합니다.
- ANALYSIS_RUN 실제 종료코드 0과 analysis.py SHA를 연결했습니다. 소스 코드의 보안·구현 전반 검토는 이 판정 범위가 아닙니다.
- 원격 호출·새 모델 추론·Test 성능 열람: 모두 0.

## 수용 기준

- [x] contract_raw_sha
- [x] prediction_sha_27
- [x] validation_only_911187_rows
- [x] target_identity_order_quantity_history_equal
- [x] train_frozen_strict_tail_cutoff
- [x] receipt_metrics_162
- [x] report_record_metrics_162
- [x] summary_cells_54_mean_and_sample_sd
- [x] paired_seed_deltas_162
- [x] mean_ratio_deltas_54
- [x] worsened_seed_sets_54
- [x] markdown_summary_cells_54
- [x] markdown_worsened_cells_54
- [x] report_evidence_sha_83
- [x] analysis_run_exit0_and_script_sha

## CRITICAL

없음.

## HIGH

없음.

## MEDIUM

없음.

## LOW

없음.

## 남은 해석 한계

- TimeNLL은 저장된 개별 NLL의 평균 검증이며 분포 파라미터에서 새 likelihood를 계산하지 않았습니다.
- 같은 Validation으로 checkpoint를 선택하고 구조를 비교했습니다. held-out 일반화·통계적 유의성·동등성·독창성을 보증하지 않습니다.
- 3seed와 작은 큰 수량 집단의 한계를 유지해야 합니다. 극소 TimeNLL 상대비를 실질적인 효과 크기로 쓰면 안 됩니다.
- B와 단일 MLP의 총 보정 파라미터 9600개는 같지만 함수 집합과 활성 부분의 용량은 다릅니다.
- 진행 중인 비용 결과와 전체 실험 완료 여부는 포함하지 않습니다.

## 변경 파일

- INDEPENDENT_QA.json
- INDEPENDENT_QA.md

기존 REPORT·analysis.py·실험 원본은 변경하지 않았습니다.
