# 5080 MLP Gate seed42 최종 validation 결과

승인된 MLP Gate 네 조건은 **2026-09-29 23:25:25 KST**에 학습·selected/last 재평가를 모두 완료했습니다. 회수 파일83개 SHA, 원격 동결 소스111/114개, 소유 프로세스 종료를 확인했고 신규 selected/last checkpoint8개의 CPU strict loading·tensor SHA·학습 이력·optimizer·노출·배치 prefix·선택·구간 재평가 집계 감사를 마쳤습니다.

**결론:** 현재 seed42에서는 조건부 Gate가 기존 이력 MLP보다 수량 예측을 일관되게 개선하지 못했습니다. Intermittent에서는 두 Gate 모두 기존 MLP보다 MAE·RMSE가 높습니다. Taxi에서는 상수 계수가 기존 MLP의 수량 지표를 개선하지만 시간 NLL이 나빠졌습니다. 조건부 Gate는 Taxi 수량 지표에서도 개선되지 않았습니다.

평가 범위는 validation만입니다. 각 모델의 strict 최초 수량 RMSE 최소 checkpoint에서 MAE·RMSE·시간·구간 지표를 함께 가져왔습니다. 수량 MAE 최소 epoch로 재선택하지 않았습니다. seed42 탐색이며 3seed 평균·통계적 우월성·held-out 일반화 주장이 아닙니다. 기존 core54조건과 아래 Gate조건은 별도입니다.

## Intermittent: 같은 seed42 selected checkpoint

| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ | 선택 / 종료 epoch |
|---|---:|---:|---:|---:|
| 기존 Full | 0.777970 | 1.792506 | 0.308359 | 5 / 45 |
| 기존 이력 MLP | 0.638480 | 1.578487 | 0.346835 | 10 / 50 |
| MLP + 조건부 Gate | 0.734496 | 1.631734 | 0.316678 | 5 / 45 |
| MLP + 상수 계수 | 0.708235 | 1.649104 | 0.363903 | 4 / 44 |

기존 이력 MLP 대비 변화율은 양수가 오차 증가, 음수가 감소입니다.
- 조건부: MAE +15.04%, RMSE +3.37%, 시간 NLL 차이 -0.030157.
- 상수: MAE +10.93%, RMSE +4.47%, 시간 NLL 차이 +0.017068.
- 조건부 대 상수: MAE +3.71%, RMSE -1.05%, 시간 NLL 차이 -0.047225.

## Taxi: 같은 seed42 selected checkpoint

| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ | 선택 / 종료 epoch |
|---|---:|---:|---:|---:|
| 기존 Full | 25.040551 | 77.616782 | 1.244313 | 175 / 215 |
| 기존 이력 MLP | 26.499368 | 82.483934 | 0.842735 | 60 / 100 |
| MLP + 조건부 Gate | 27.722204 | 86.737720 | 0.678927 | 27 / 67 |
| MLP + 상수 계수 | 26.122403 | 80.208327 | 1.304540 | 134 / 174 |

기존 이력 MLP 대비 변화율은 양수가 오차 증가, 음수가 감소입니다.
- 조건부: MAE +4.61%, RMSE +5.16%, 시간 NLL 차이 -0.163808.
- 상수: MAE -1.42%, RMSE -2.76%, 시간 NLL 차이 +0.461806.
- 조건부 대 상수: MAE +6.12%, RMSE +8.14%, 시간 NLL 차이 -0.625614.

## 선택 이후와 구간별 반례

| 조건 | Last MAE | Last RMSE | Last 시간 NLL | Selected tail MAE / RMSE |
|---|---:|---:|---:|---:|
| mlp_A | 1.014753 | 3.149529 | 0.373104 | 5.212183 / 6.928635 |
| mlp_B | 0.771280 | 2.072524 | 0.401178 | 5.846174 / 7.474653 |
| mlp_C | 29.904683 | 91.890910 | 0.925708 | 363.680086 / 449.618606 |
| mlp_D | 26.035714 | 81.565672 | 1.593910 | 337.497024 / 409.626939 |

Intermittent 조건부는 5epoch가 마지막까지 best였고 45epoch에서 patience40으로 종료됐습니다. 이후 학습이 선택 checkpoint의 수량 성능을 갱신하지 못했습니다. Taxi 상수의 수량 개선을 시간 예측이나 모든 구간의 개선으로 일반화하지 않습니다. 아래는 기존 MLP 대비 악화된 구간·지표의 전체 목록이며 원값·표본수·빈 구간·전체 셀은 comparison.json에 보존합니다.

- mlp_A: body(qty_mae, qty_rmse); quantity_cells:0(qty_mae, qty_rmse); quantity_cells:1(qty_mae); quantity_cells:2(qty_mae, qty_rmse, time_nll); quantity_cells:3(qty_mae, qty_rmse); history_cells:0(qty_mae, qty_rmse); history_cells:1(qty_mae, qty_rmse); history_cells:2(qty_mae, qty_rmse); additional_history_cells:0(qty_mae, qty_rmse); additional_history_cells:1(qty_mae, qty_rmse); additional_history_cells:2(qty_mae, qty_rmse).
- mlp_B: body(qty_mae, qty_rmse, time_nll); tail(qty_mae); quantity_cells:0(qty_mae, qty_rmse); quantity_cells:1(time_nll); quantity_cells:2(qty_mae, qty_rmse, time_nll); quantity_cells:3(qty_mae, qty_rmse, time_nll); quantity_cells:4(qty_mae); history_cells:0(qty_mae, qty_rmse, time_nll); history_cells:1(qty_mae, qty_rmse); history_cells:2(qty_mae); additional_history_cells:0(qty_mae, qty_rmse, time_nll); additional_history_cells:1(qty_mae, qty_rmse); additional_history_cells:2(qty_mae).
- mlp_C: body(qty_mae, qty_rmse); tail(qty_mae, qty_rmse, time_nll); quantity_cells:0(qty_mae); quantity_cells:1(qty_mae); quantity_cells:2(qty_mae, qty_rmse, time_nll); quantity_cells:3(qty_mae, qty_rmse, time_nll); quantity_cells:4(qty_mae, qty_rmse, time_nll); history_cells:0(qty_mae); history_cells:1(qty_mae); history_cells:2(qty_mae, qty_rmse); additional_history_cells:0(qty_mae); additional_history_cells:1(qty_mae); additional_history_cells:2(qty_mae, qty_rmse).
- mlp_D: body(time_nll); tail(time_nll); quantity_cells:0(time_nll); quantity_cells:1(qty_mae, time_nll); quantity_cells:2(time_nll); quantity_cells:3(time_nll); quantity_cells:4(time_nll); history_cells:0(qty_mae, qty_rmse, time_nll); history_cells:1(qty_mae, qty_rmse, time_nll); history_cells:2(time_nll); additional_history_cells:0(qty_mae, qty_rmse, time_nll); additional_history_cells:1(qty_mae, qty_rmse, time_nll); additional_history_cells:2(time_nll).

수량 구간은 train에서 고정한 경계를 사용합니다. 이력 기본 bin0/1/2는 ≤64, 65–128, >128입니다. additional_history_cells는 계약의 추가 경계이며 기본 구간과 별도로 보존합니다. 같은 셀의 수량과 시간 점수를 분리해 해석합니다.

## 기존 Full Gate A/B와 보류 C/D — 별도 결과

| 조건 | Selected MAE | Selected RMSE | 시간 NLL | 선택 / 종료 epoch |
|---|---:|---:|---:|---:|
| full_A | 0.731695 | 1.782706 | 0.304817 | 41 / 81 |
| full_B | 0.668810 | 1.699489 | 0.365952 | 57 / 97 |

두 조건은 Intermittent seed42입니다. Full A는 변화 경로 조건부 Gate, Full B는 변화 경로 상수 계수이며 MLP 보정 출력 전체에 Gate를 적용한 새 실험과 다릅니다. Full Taxi C/D는 사용자 전환 승인으로 **미시작 보류**이며 성공·실패 조건으로 계수하지 않습니다. Full A의 기존 CPU감사와24파일 회수, Full B의17파일 회수·CPU감사를 재사용했습니다. Full B attempt0의73epoch ACK timeout과 승인된74epoch 복구 이력은 보존했습니다.

## 실제 시간·메모리·비용

| 조건 | 파라미터 | fit 초 | 평균 초/epoch | peak allocated MiB | 종료 KST |
|---|---:|---:|---:|---:|---|
| mlp_A | 96,008 | 5676.07 | 125.95 | 1879.04 | 2026-09-29 20:59:24 KST |
| mlp_B | 96,004 | 5521.88 | 125.31 | 1878.54 | 2026-09-29 22:31:45 KST |
| mlp_C | 96,008 | 891.03 | 13.28 | 1879.04 | 2026-09-29 22:46:47 KST |
| mlp_D | 96,004 | 2312.67 | 13.28 | 1878.54 | 2026-09-29 23:25:25 KST |

MLP 네 fit 실측 합계는 14401.646초입니다. fit은 train_one 구간이며 별도 endpoint replay는 제외합니다. epoch 시간은 train iteration부터 validation/checkpoint 저장까지입니다. 파라미터 증가는 plain MLP96003 대비 상수+1, 조건부+5입니다. 전용 속도·추론비용 시험이 아니며 조기 종료 epoch도 다릅니다.

기존 Full Gate 최초 시작부터 새 큐 종료까지 14.356시간의 벽시계 시간을 기록했습니다. 준비 실패·대기·절전·전환을 포함하며 순수 GPU 연산시간이 아닙니다. 원래 마감 2026-10-02 09:04:05 KST 이내에 끝났습니다. Full A fit11505.963초는 Mac 백업 ACK 대기를 포함합니다. Full B summary3075.468초는 복구 이후만이며 원래1~73epoch9432.632초와 복구74~97epoch3067.489초의 완료epoch 기록합12500.121초를 따로 보존합니다. 시간을 추정으로 차감하거나 보정하지 않았습니다.

개인5080의 추가 cloud 임대료는0달러이며 전기료는 미측정입니다. 원래 v1 CUDA 경로 실패, retry1 데이터 복제 누락, Full B ACK timeout, 서버 자체 실행 배포 중 문법 실패도 별도 이력으로 유지합니다. 새 MLP 네 fit의 실행 실패는0입니다.

## 감사 범위와 남은 작업

**5080 승인 Gate 실행·회수·보고 — 완료**

- MLP4조건/8 endpoint 역할, 기존 Full A/B와 보류 C/D를 구분했습니다. 동결 모델·계약·native 공통 초기화, optimizer 및 train 전체 노출, 기준선 batch prefix, RNG/shuffle 존재, 최초 strict 선택·조기 종료, replay 집계와 same-epoch MAE/RMSE/time을 확인했습니다.
- 서버 학습은 Mac ACK 없이 완료됐고 소유 프로세스 종료를 확인했습니다. 원본 서버 파일과 실패 기록은 수정하지 않았습니다. 새로운 학습·CUDA qualification·prediction replay·held-out 열람은 하지 않았습니다.

**5090 core 종료·최종 취합 — 진행 중**

- 별도 core의03:26KST 관측에서 신규35/36조건·70replays 감사완료, 재사용18조건 감사완료입니다. 마지막 Instacart seed62 정적 검색 제거는77epoch였으며 이 관측 이후 상태는 다음 heartbeat에서 확인합니다. 이 Gate 보고서를 core전체 완료로 해석하지 않습니다.
- 마지막 조건이 종료되면 전체54조건/108역할의 원본 회수·최종감사·3seed 결과/비용 보고를 마치고 heartbeat를 삭제합니다. 추가 seed·모델·튜닝·held-out·예산 연장은 실행하지 않습니다.

근거: [기계 판독 비교](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_mlp_gate_execution_20260929_v1/comparison.json), [신규 MLP terminal 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_autonomous_5080_20260929_v1/retrieved/mlp_terminal/terminal_audit.json), [원본 회수 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_autonomous_5080_20260929_v1/retrieved/mlp_terminal/retrieval_receipt.json).
