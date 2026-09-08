# Instacart BOUNDED-QK 표본별 Raw RMSE 진단

## 결론

이 진단은 실행할 가치가 있었고, Raw RMSE 악화의 위치를 분명히 좁혔다.
BOUNDED-QK는 B보다 **전체 MAE, log-space MSE, 잔차의 중심화 RMSE를 개선**한다.
그러나 평균 예측을 `0.143628`만큼 더 낮추면서 과소예측 편향을 키웠고, 이
level shift가 중심화 오차 개선을 상쇄해 Raw RMSE를 악화시켰다. 따라서 이
후보에는 유용한 Backbone 신호가 있지만, 그 신호를 유지하면서 수량 수준을
보존하는 장치가 부족하다.

악화는 소수 series에만 생긴 현상이 아니다. 가장 큰 net harm을 보인 25개
series가 전체 per-row gross positive harm에서 차지하는 비중은 `1.27%`였다.
표본 조건으로는 모델이 본 이력이 `2–3`개인 구간과 실제 수량 `p50–p90` 구간이
가장 큰 양의 ΔSE를 만들었다. 반대로 `<=p50`과 이력 `8–15`개 구간에서는
BOUNDED-QK가 분명히 개선됐다.

## 범위와 비교 계약

- 데이터: Instacart seed 42 validation `503,733`개 target
- 주 비교: `BOUNDED-QK − B`
- 보조 비교: `BOUNDED-QK − FULL`, `FULL − B`
- 세 checkpoint 모두 validation Raw RMSE의 가장 이른 최솟값
- 모델 학습, checkpoint 재선택, calibration fitting, held-out test 접근 없음
- 동일 dataset instance와 동일 batch를 세 frozen model에 입력
- 표본 identity: `(series_id, target_position, target_seq)`
- 수량 구간 경계 `[8, 20, 25, 35]`는 train split에서 이미 정해진 값을 재사용

원본 paired prediction은 크기 때문에 Git에 넣지 않았다. 로컬 ignored artifact
`search_artifacts/hard_lmm_bounded_qk_error_diagnostic_20260908/paired_validation_predictions.parquet`
에 보관하며, SHA-256은
`98fa0e2fd514a60c59ee4aa66bb795efd32b097c11ebb009daae9a1af4fc2f0f`이다.

## 전체 결과

| 모델 | MAE | Raw RMSE | Bias | Centered RMSE | Log MSE |
|---|---:|---:|---:|---:|---:|
| B | 3.993781 | 5.872217 | -0.849631 | 5.810427 | 0.244596 |
| FULL | 3.995619 | 5.877257 | -0.878694 | 5.811200 | 0.244326 |
| BOUNDED-QK | **3.988698** | 5.886228 | -0.993258 | **5.801820** | **0.243203** |

BOUNDED-QK와 B의 차이는 다음과 같다.

- MAE: `-0.005083`, 상대 `-0.127%`
- Raw RMSE: `+0.014011`, 상대 `+0.239%`
- Log MSE: `-0.001393`, 상대 `-0.569%`
- Bias: `-0.143628`만큼 더 음수
- Centered MSE: `-0.099941`
- Bias 제곱: `+0.264690`
- Net MSE: `+0.164748 = -0.099941 + 0.264690`

Bias 제곱 증가는 net MSE 악화의 `160.66%`에 해당하고, centered MSE 개선이 그중
`60.66%`를 상쇄했다. 이 분해는 두 개의 deterministic series fold에서 모두 같은
방향이었다. FULL은 B보다 centered RMSE도 소폭 나빴지만, BOUNDED-QK는 FULL과
B 둘보다 centered RMSE가 낮았다. Q/K residual을 제한한 변경 자체는 오차의
산포를 줄이는 효과가 있었다고 해석할 수 있다.

## 실제 수량 구간

| Train 기준 구간 | 표본 수 | ΔMAE | ΔRMSE | Net ΔSE |
|---|---:|---:|---:|---:|
| `<=p50` | 247,651 | **-0.092440** | **-0.103153** | **-194,866.95** |
| `p50–p90` | 202,534 | +0.078612 | +0.084943 | +160,922.32 |
| `p90–p95` | 27,322 | +0.113477 | +0.122659 | +60,957.30 |
| `p95–p99` | 20,190 | +0.069947 | +0.090975 | +48,267.74 |
| `>p99` | 6,036 | **-0.016898** | +0.026031 | +7,708.85 |

낮아진 예측 수준은 `<=p50`에서 이득이지만, 이미 과소예측 중인 `p50–p99`에서
손실이다. `p50–p90`은 per-row gross positive harm의 `39.02%`를 차지하며,
구간별 net ΔSE 기준으로도 가장 큰 악화 구간이다. `>p99`에서는 MAE가 조금
좋아져도 큰 오차의 제곱 때문에 RMSE는 나빠진다.

## 이력 길이와 최근 상태

| 모델이 본 이력 | 표본 수 | ΔMAE | ΔRMSE | Net ΔSE |
|---|---:|---:|---:|---:|
| `1` | 261 | +0.091662 | +0.092222 | +510.78 |
| `2–3` | 130,279 | +0.014139 | +0.094642 | **+147,721.34** |
| `4–7` | 205,041 | **-0.006538** | +0.005548 | +13,124.94 |
| `8–15` | 140,970 | **-0.021608** | **-0.050033** | **-83,636.54** |
| `16–31` | 26,298 | +0.001928 | +0.016283 | +4,966.45 |
| `32–63` | 884 | **-0.102534** | +0.024807 | +302.28 |

이력 `2–3`개에서 BOUNDED-QK의 평균 prediction shift는 `-0.557328`이고, 이
구간 하나가 전체 net ΔSE보다 큰 악화를 만든다. 이력 `8–15`개 구간의 개선이
그 손실을 크게 상쇄한다. `p50–p90` 내부에서도 이력 `2–3`과 `4–7` 구간이 각각
`+61,146.72`, `+73,483.58`의 net ΔSE를 만들었다.

`recent3 mean log quantity − history mean log quantity = 0` 구간의 net ΔSE는
`+148,219.84`다. 다만 이 값은 별도 수요 패턴으로 해석하면 안 된다. 이력이
3개 이하이면 recent-3과 전체 이력이 같으므로 이 신호가 정의상 0이 된다.
따라서 이 결과는 최근 변화 자체보다 **짧은 이력에서 residual 크기와 level을
안정화하지 못한 문제**를 주로 반영한다.

## 오차 집중도

- BOUNDED-QK가 더 나쁜 표본: `49.91%`
- 더 좋은 표본: `50.09%`
- 전체 gross positive harm: `1,103,174.15`
- 전체 gross negative help: `1,020,184.90`
- Net ΔSE: `82,989.26`
- 가장 해로운 `1%` 표본이 gross positive harm의 `25.13%`를 차지
- 가장 해로운 `5%` 표본이 gross positive harm의 `57.28%`를 차지
- net harm 상위 25개 series의 표본은 전체의 `0.0248%`, gross positive harm은
  전체의 `1.27%`

극단 표본의 영향은 존재하지만, 결과를 소수 사용자 series의 이상치만으로
설명할 수는 없다. 같은 방향의 작은 변화가 넓은 series에서 반복되고 있다.

## 설명용 bias 반사실

Validation target으로 BOUNDED-QK의 bias를 B와 같게 만드는 상수 shift는
`+0.143628`이다. 이 값을 같은 validation prediction에 더하면 Raw RMSE는
`5.863701`이 된다. 이 수치는 과소예측 level shift가 관측된 RMSE 격차를
설명한다는 확인용 계산이다. Validation target으로 값을 구하고 같은 validation에
적용했으므로 성능 결과, calibration 후보의 성공, 일반화 증거로 사용할 수 없다.

기존 Frozen-B global affine calibration은 train-only series-disjoint audit에서 세
데이터셋 모두 gate를 통과하지 못했다. Instacart는 OOF RMSE 개선이 `0.3516%`에
그쳤고 body MAE가 `2.1846%` 악화됐다. 따라서 이 진단을 기존 affine 보정의
재시도 근거로 해석하지 않는다.

## 다음 Backbone 설계에 주는 조건

BOUNDED-QK 전체를 폐기할 근거는 없다. centered RMSE와 log MSE 개선은 기존 B와
FULL에서 없던 유용한 신호다. 다음 공통 Backbone 후보는 다음 두 조건을 동시에
만족해야 한다.

1. BOUNDED-QK가 만든 centered residual 개선을 보존한다.
2. 특히 이력 `2–3`개에서 causal residual이 수량 prediction level을 과도하게
   낮추지 않도록 history-confidence gate 또는 level-preserving residual 제약을
   둔다.

실제 target 수량 구간은 추론 시 알 수 없으므로 `p50–p90` 여부를 gate 입력으로
사용해서는 안 된다. 먼저 frozen checkpoint의 train prediction을 series-disjoint
두 fold로 나눠 `-0.143628` 수준의 하향 이동이 train에서도 재현되는지 확인해야
한다. 재현될 때만 동일한 공통 gate/제약을 구현한다. 단순 output affine 보정은
Backbone 기여와 분리된 control로만 취급한다.

## 증적

- `analysis.json`: 전체 집계, 정확한 MSE·bias 분해, fold 일관성, oracle 경계
- `inference_audit.json`: source/data/checkpoint SHA, 동일 표본 정렬, frozen state 감사
- `target_strata.csv`: 수량 구간 결과
- `history_strata.csv`: 이력 길이 결과
- `p50_p90_by_history.csv`: 주 악화 구간과 이력 길이 교차표
- `p50_p90_by_recent_signal.csv`: 주 악화 구간과 최근 편차 교차표
- `top_residual_delta.csv`, `top_series.csv`: 오차 집중도
- 전체 per-series 표는 로컬 ignored
  `search_artifacts/hard_lmm_bounded_qk_error_diagnostic_20260908/analysis_detail/series_contributions.csv`
  에 보관한다.

계약은 `paper/contracts/hard_lmm_bounded_qk_error_diagnostic_v1.json`, 실행기는
`paper/scripts/run_hard_lmm_bounded_qk_error_diagnostic.py`, 분석기는
`paper/scripts/analyze_hard_lmm_bounded_qk_error_diagnostic.py`이다.
