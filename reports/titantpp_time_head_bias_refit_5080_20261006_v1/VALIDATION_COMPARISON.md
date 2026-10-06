# 수량 경로 동결·시간 보정값2개 재적합 Validation 결과

**6조건 학습·원본 회수를 마친다 — 완료**
- 5080의 Taxi·RAF × seed42/52/62 6조건을 완료했습니다. 2026-10-06 09:57:54 KST 실제 관측에서 controller/worker/GPU 소유PID 부재, supervisor exit0, 모든 terminal scientific_success와 SHA를 확인했습니다.
- Train에서 b_t/w_raw 두 scalar만 다시 학습했습니다. 원래 정수 시간 likelihood와 단위를 유지하며, 원본 E0를 포함한 full Validation TimeNLL로 선택했습니다. Test는 평가하지 않았습니다.
- 전체 Train/Validation 수량 prediction tensor byteSHA 및 나머지 모든 weight/buffer SHA는 6조건 모두 before/after 정확히 같습니다. 이 동일성은 각각의 5080 원본 E0 재현과 새 보정값 사이의 검증이며, A100과 RTX의 bitwise 동등성 주장과 구분합니다.
- 새 head selected/last·이력·캐시와 원래 selected checkpoint를 119개 파일 원본 manifest/archive로 회수해 SHA를 검증했습니다. CPU binary 재추론 감사는 별도 미완료입니다.

**같은 수량 checkpoint에서 시간 손해가 줄었는지 비교한다 — 완료**

| 데이터 | seed | 원래 수량 선택 epoch | 새 head 선택/마지막 epoch | TimeNLL E0 → 선택 | 변화 |
|---|---:|---:|---:|---:|---:|
| Taxi | 42 | 126 | 0/10 | 1.365979 → 1.365979 | +0.000% |
| Taxi | 52 | 157 | 0/10 | 2.163854 → 2.163854 | +0.000% |
| Taxi | 62 | 54 | 0/10 | 0.875252 → 0.875252 | +0.000% |
| RAF | 42 | 9 | 8/18 | 3.548010 → 3.510662 | -1.053% |
| RAF | 52 | 5 | 4/14 | 3.479681 → 3.439869 | -1.144% |
| RAF | 62 | 20 | 0/10 | 3.611548 → 3.611548 | +0.000% |

E0는 개선이 없어 원래 보정값을 유지한 경우입니다. 4조건은 E0, RAF seed52는 새 head E4, RAF seed42는 E8을 선택했습니다. 실패나 미시작은 0개입니다.

| 데이터 | TimeNLL 원본 mean±sample SD | 재적합 mean±sample SD | 평균 변화 | 수량 RMSE | 수량 MAE | 큰 수량 RMSE | 큰 수량 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taxi | 1.468362±0.650374 | 1.468362±0.650374 | +0.000% | 79.0628±2.6668 | 25.2797±0.7099 | 392.1247±24.7344 | 303.9220±29.0428 |
| RAF | 3.546413±0.065948 | 3.520693±0.086278 | -0.725% | 34.1841±0.2334 | 9.2062±0.0741 | 328.3920±7.8560 | 281.5753±8.7931 |

수량 4열은 원본과 재적합에서 동일합니다. 큰 수량은 원래 Train 기준 Taxi3449·RAF200 초과이며, seed당 Taxi79/RAF50 Validation target입니다. 표준편차는 seed3개의 표본 표준편차(ddof1)이며 통계적 유의성 주장으로 쓰지 않습니다.

| 데이터 | seed | 큰 수량 TimeNLL 원본 → 선택 |
|---|---:|---:|
| Taxi | 42 | 9.05632e-123 → 9.05632e-123 |
| Taxi | 52 | 4.10182e-19 → 4.10182e-19 |
| Taxi | 62 | 2.80381e-32 → 2.80381e-32 |
| RAF | 42 | 3.51547 → 3.47733 |
| RAF | 52 | 3.26486 → 3.24388 |
| RAF | 62 | 3.70553 → 3.70553 |

**확인한 개선 범위에 맞춰 다음 실험을 정한다 — 다음 작업**
- 확인한 사실: 두 scalar 방식은 수량을 정확히 보존했습니다. Taxi 시간 NLL은 세 seed 모두 개선되지 않았고, RAF는 두 seed에서 소폭 개선돼 평균0.725% 감소했습니다.
- 해석: 이번 Train 재적합·LR·patience 조건의 두 scalar만으로 Taxi 시간 손해를 해결하지 못했습니다. 시간 예측 개선 자체가 불가능하다거나 gradient 충돌이 확정됐다고 해석하지 않습니다. 원래 Train을 본 encoder의 사후 재적합이며 독립 calibration이 아닙니다.
- 다음 작업: 130개 시간 출력부 전체 재적합의 계약과 데이터별 과적합 방지 gate를 검토합니다. 이 단계에서 고정된 시간 projection도 조절할 수 있지만 자동으로 시작하지 않았습니다.
- 외부 작업 대기: 5090의 원래 Intermittent seed62 완료·원본 회수 후 같은 동결 시간 보정 단계를 연결할 수 있습니다. 이번 작업에서 5090을 변경하거나 새 명령을 실행하지 않았습니다.
- 다음 작업: 같은 범위의 S2P2 재적합 대조와, 모델/계약 확정 후 별도 승인된 Test·독립 평가로 외부 비교군 우위를 검증합니다. 이번 두 scalar 실험만으로 S2P2 우월성·CNN/GRU 독립 기여를 주장하지 않습니다.

**운영·증적을 보존한다 — 완료**
- 첫 native 회차는 입력 경로 준비 오류로 실제 fit0에서 중단됐습니다. claim/failure/원본 계약을 그대로 보존하고, source cwd를 복원한 별도 attempt2만 실행했습니다. 최초 전체 lease 마감16:15:04 KST를 유지했습니다.
- 로컬 검증: runtime42 + operations22 = 64 tests, 추가22 subtests 통과. actual native 대표 Taxi52 검증은 real optimizer update0이며, 각6 fit은 자체 E0/원본/수량 검증 후 Train optimizer를 실행했습니다.
- 현재 계약 SHA: `c94bcfe1699282a4cd25b36fed3f28ec0a2d30517b3de026fc04c109acbce8f7`.
- 회수 archive SHA: `4b31019b26c500cded324ca1812bfb0e505c976e791caaa0fddfb3b6e9003689`.
- 실제 비교 split과 원본: `analysis/Validation_refit_comparison.json`, 요약: `analysis/Validation_refit_summary.json`, 회수 증적: `analysis/completion_receipt.json`, 실제 관측: `hourly_monitor/terminal_cache.json`.
- 기존 시간별 자동화에 새 head 관측·회수를 연결했습니다. 새 head는 terminal cache를 재사용해 반복 SSH를 하지 않습니다. 원래 CNN+GRU54 9조건의 최종 회수·종료 확인은 별도 gate이므로 자동화는 ACTIVE로 유지합니다.
