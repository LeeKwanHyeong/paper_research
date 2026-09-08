# LPHC-QKV 5090 screening 결과

상태: **Instacart seed42에서 후보 중단**. Taxi·Intermittent seed42, 추가 seed, normalized time fit, held-out test는 실행하지 않았다.

## 판정

LPHC-QKV는 epoch 80에서 조기 종료됐고 validation raw RMSE의 가장 이른 최솟값인 epoch 40을 선택했다. B 대비 전체·body MAE는 소폭 개선됐지만 사전 고정한 핵심 조건인 raw RMSE strict 개선에 실패했다.

| 지표 | B | FULL | LPHC-QKV | B 대비 | 판정 |
|---|---:|---:|---:|---:|---|
| Raw RMSE | 5.872216930 | 5.877257164 | 5.872889737 | +0.0115% | 실패 |
| 전체 MAE | 3.993781078 | 3.995618642 | 3.992162749 | -0.0405% | 통과 |
| Body MAE | 3.438601503 | 3.428296373 | 3.431271518 | -0.2132% | 통과 |
| >p99 MAE | 21.917794765 | 22.190151583 | 22.093013990 | +0.7994% | 통과 |

Raw RMSE 차이는 절대 `+0.000672807`로 매우 작지만, 결과 확인 전에 strict 감소를 채택 조건으로 고정했으므로 통과로 바꾸지 않는다.

## 기전 감사

원래 캠페인의 기전 감사기는 `python -s <absolute script>`에서 repository root를 import path에 넣지 않아 실행 오류가 발생했다. 학습이나 checkpoint 선택을 반복하지 않고, 동일 commit·동일 frozen checkpoint를 검증된 source root에서 module로 실행해 감사를 완료했다.

7개 기전 조건 중 다음 네 개가 실패했다.

- centered MSE: `33.766448461` (B `33.761059326`)
- 이력 2–3 raw MSE: `36.229186986` (B `35.320082612`, +2.574%)
- 절대 bias: `0.851108346` (B `0.849630703`)
- raw RMSE strict 개선 실패

이력 8–15 raw MSE는 `34.762990950`로 B `35.450350643`보다 1.939% 개선됐고, 전체 MAE와 평균 prediction shift 조건도 통과했다. 따라서 전이 residual은 긴 이력 일부에는 유용했지만, 설계 목표였던 짧은 이력 병목과 centered error를 해결하지 못했다.

## 실행 계약

- source: `667f9810d3c183b0217d0d511f8031a7d2b7451f`
- screening contract: `a4ba58bb69f40496de20869643843798a13af17ca169ee4587815c55b8d1061d`
- CUDA qualification과 세 데이터셋 full-data e1 통과
- fresh B/LPHC 공통 상태·초기 출력·RNG bitwise identity 검증
- 추가 파라미터: 384개
- 비용 gate 통과

| 길이 | B 대비 step | B 대비 peak memory |
|---:|---:|---:|
| 8 | 1.423× | 1.056× |
| 64 | 1.455× | 1.074× |
| 256 | 1.164× | 1.058× |

## 결론과 다음 순서

LPHC-QKV는 공통 Backbone 후보로 채택하지 않는다. 이 계열을 Taxi·Intermittent나 추가 seed로 확장하지 않으며, `paper_research/master`에도 최종 Backbone으로 병합하지 않는다. 다음 Backbone 검토는 긴 이력 이득을 유지하면서 이력 2–3에서 residual을 억제하거나 방향을 교정할 근거부터 다시 확인해야 한다. 결과를 보정하기 위한 dataset별 계수나 사후 threshold 변경은 사용하지 않는다.
