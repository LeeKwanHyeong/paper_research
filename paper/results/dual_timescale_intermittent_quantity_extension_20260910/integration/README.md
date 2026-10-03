# Dual-timescale 후보 최종 validation 비교

**상태: 완료.** Intermittent는 2026-09-10 09:50:14 KST에 77 epoch로 종료했고, 최적 epoch37의 실행 감사가 통과했다.

동일 seed42 validation 표본과 최초 strict raw-RMSE 최적 checkpoint 기준이다. Body는 train p95 이하, tail은 train p99 초과다. 실행 감사 통과와 성능 기준 통과를 구분한다.

| 데이터셋 | 모델 | 최적/종료 epoch | RMSE | 전체 MAE | Body MAE | tail MAE | legacy time loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Taxi | TitanTPP(B) | 45/85 | 88.194997 | 28.674020 | 18.997478 | 332.769744 | 1.473391 |
| Taxi | RMTPP | 46/86 | 93.286440 | 28.196301 | 17.127993 | 362.707309 | 1.357873 |
| Taxi | THP | 193/233 | 100.023882 | 31.557175 | 19.141899 | 392.097536 | 1.361875 |
| Taxi | Dual-timescale candidate | 83/123 | 78.255427 | 25.048308 | 15.738603 | 303.451314 | 10.825870 |
| Instacart | TitanTPP(B) | 72/112 | 5.872217 | 3.993781 | 3.438602 | 21.917795 | 3.215523 |
| Instacart | RMTPP | 40/80 | 5.836961 | 3.985841 | 3.440235 | 21.888547 | 3.217361 |
| Instacart | THP | 50/90 | 5.852342 | 3.991003 | 3.454441 | 21.773342 | 3.222556 |
| Instacart | Dual-timescale candidate | 72/112 | 5.868059 | 3.985497 | 3.433833 | 21.781940 | 3.214805 |
| Intermittent | TitanTPP(B) | 77/117 | 1.499555 | 0.604340 | 0.465790 | 5.148108 | -3.483794 |
| Intermittent | RMTPP | 35/75 | 1.833607 | 0.669522 | 0.482428 | 7.982565 | -3.139204 |
| Intermittent | THP | 104/144 | 1.649723 | 0.598333 | 0.431940 | 6.593146 | -3.556539 |
| Intermittent | Dual-timescale candidate | 37/77 | 1.580844 | 0.667658 | 0.502126 | 5.826921 | -3.168242 |

## B 대비 후보 변화

음수는 개선, 양수는 악화다.

| 데이터셋 | RMSE | 전체 MAE | Body MAE | tail MAE |
| --- | --- | --- | --- | --- |
| Taxi | -11.27% | -12.64% | -17.15% | -8.81% |
| Instacart | -0.07% | -0.21% | -0.14% | -0.62% |
| Intermittent | +5.42% | +10.48% | +7.80% | +13.19% |

## 판단과 남은 작업

- Taxi: 수량 지표는 개선했으나 legacy time loss 10.825870은 B 1.473391 대비 기존 허용 기준을 충족하지 못한다. 원래 판정을 유지한다.
- Instacart: B보다 소폭 개선했으나 RMSE는 RMTPP·THP보다 높다. 기존 CUDA 결과를 유지하며 CPU 진단 재현값으로 대체하지 않았다.
- Intermittent: B 대비 전체·Body·tail 수량 오차 모두 악화했다. RMSE와 tail MAE는 RMTPP·THP보다 낮으나, 전체 MAE는 THP보다 높고 Body MAE는 두 모델보다 높다.
- 세 데이터셋 공통 개선 근거가 없으므로 B 유지가 타당하다. 단일 seed validation 결과이며 최종 후보 채택이나 통계적 유의성을 선언하지 않는다.
- 다음 작업: Intermittent 오차를 수량 구간·이력 길이별로 B와 대조할 진단 계획을 구체화한다. Body와 tail 모두 악화했으므로 tail만의 문제로 가정하지 않는다. Instacart 원인을 일반화하거나 모델 수정 대상을 아직 확정하지 않는다.
- 새로운 진단 추론·모델/loss 변경·재학습·추가 seed·held-out 평가는 실행하지 않았다. 기존 캠페인 판정은 보존했다.

원본 경로·SHA·전체 비교와 검증 항목은 [comparison.json](comparison.json), 서버 실행 감사는 [audit.json](../remote/audit.json)에 기록했다.

Scheduler **Intermittent 수량 평가 점검**은 종료 결과 통합 후 **PAUSED**로 확인했다. [확인 기록](automation_terminal_receipt.json)
