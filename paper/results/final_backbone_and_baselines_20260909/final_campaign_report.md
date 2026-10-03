# 마지막 Backbone 평가 종료: TitanTPP(B) 유지

2026-09-10 07:06 KST 최종 확인. 후보 source `ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c`.

신규 두 시간척도 memory 후보는 Instacart seed42에서 고정 기준을 통과했다. Taxi에서는 수량 지표가 모두 개선됐지만 기존 시간 loss가 크게 악화되어 공통 성능 기준을 통과하지 못했다. 사전 계약에 따라 Intermittent full screening을 시작하지 않고 기존 TitanTPP(B)를 유지한다.

## Taxi: 123 epoch 종료, 선택 epoch 83

| 지표 ↓ | B | 신규 후보 | 변화 |
|---|---:|---:|---:|
| Raw RMSE | 88.194997 | 78.255427 | -11.27% |
| 전체 MAE | 28.674020 | 25.048308 | -12.64% |
| Body MAE | 18.997478 | 15.738603 | -17.15% |
| >p99 MAE | 332.769744 | 303.451314 | -8.81% |
| 기존 시간 loss | 1.473391 | 10.825870 | +9.3525 |

시간 기준은 B+0.01 이하(1.4833906265)였으며 실제값은10.8258699686이다. Raw-RMSE selector로 선택한 동일 epoch83의 모든 지표를 사용했다. 결과를 본 뒤 selector·loss·기준을 바꾸지 않았다.

## 검증과 해석

- 실행 계약과 성능 판정은 구분된다. Taxi의 CUDA·finite·처리 건수·저장·복원 감사는 passed이고 성능 gate의 시간 항목이 failed다.
- Taxi validation8,268건, source/data/split/대상 identity와 quantity SHA, 선택·종료 epoch, 실제 checkpoint byte SHA와 strict 모델·AdamW 복원 증적을 확인했다.
- 시간 head는 B와 같은 legacy_clamped_rmtpp(scale3, intercept cap300, w_max10/3)이다. 같은 점수 정의이며 정상화된 Time NLL 비교가 아니다.
- 저장된 수량 구간별 시간 loss를 표본 수로 가중하면10.8258699686이 재현된다. <=p50 구간(4,364건)의 시간 loss19.739087541이 전체 시간 loss에10.41865를 기여한다. 같은 epoch train 시간 loss는 약1.260으로 validation 시간 일반화 격차가 관측됐다. 이는 원인 규명을 완료했다는 뜻이 아니다.
- Instacart는112epoch/best72에서 모든 gate를 통과했다. RMSE 개선0.0708%, MAE 개선0.2074%로 작으며 단일 seed validation 결과다.
- 이 결과는 수량 개선이 관측된 후보의 미채택 기록이다. 공통 시간·수량 개선 또는 통계적 우위는 입증되지 않았다.

## 종료 상태

- 5080: 빠진 RMTPP·THP 기준선4개 완료,4/4 감사 통과. 기존 Instacart와 B를 합친9행 비교표도 완료.
- 5090: `stopped_seed42_gate_failed`, `final_model=TitanTPP(B)`, GPU compute없음, campaign tmux정상종료.
- Intermittent 후보 full screening·추가seed·held-out은 실행하지 않았다. e1 Intermittent 실행 증적만 있다.
- 후보 소스와 모든 결과를 보존한다. 신규 모델 탐색·master/develop 병합·push를 실행하지 않았다.
- 시간별 자동 점검은 두 큐 종료 확인 후 PAUSED로 전환 완료했다.

[기준선 9행 비교표](seed42_validation_comparison.md) · [최종 판정 JSON](final_campaign_decision.json) · [시간별 실행 기록](README.md)

## 후속 원인 진단 — 판정 변경 없음

전체 학습 이력을 확인하니 기존 B도 후반에는 시간 loss가 커졌다(epoch83:8.0080). B는epoch45를, 후보는epoch83을 수량 RMSE 최적으로 선택했다. 최종 시간 loss 증가의98.975%는 이력64개 이하 집단에서 발생한다. 초기 관측 구간을 제외한 유사 희소 집단은train에도 있어, validation 데이터만의 새 현상으로 단정할 수 없다. 후보epoch45에는RMSE·전체MAE·시간loss가 B의 선택 결과보다 모두 좋았다는 기록도 있다. 이 관측은 checkpoint 재선택이나 gate 통과 판정이 아니다. [학습 궤적·수식·입력 집단 진단](taxi_time_loss_diagnostic/report.md)에 확정 사실과 남은 원인 가설을 분리했다.
