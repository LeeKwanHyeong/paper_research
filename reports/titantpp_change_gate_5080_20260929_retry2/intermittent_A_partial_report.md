# 5080 Gate 중간 결과 — Intermittent seed42 조건부 A 완료

2026-09-29 KST. validation만 사용했다. 완료1/4조건이며 seed42 결과다. 3seed 평균·일반화 또는 전체 성능 기준 통과로 해석하지 않는다.

| 모델 | 수량 MAE ↓ | 수량 RMSE ↓ | 시간 NLL ↓ |
|---|---:|---:|---:|
| 기존 Full | 0.777970 | 1.792506 | 0.308359 |
| 기존 이력 MLP | 0.638480 | 1.578487 | 0.346835 |
| Full + 조건부 Gate | 0.731695 | 1.782706 | 0.304817 |

각 모델은 자기 strict 최초 validation 수량 RMSE 최솟값의 동일 checkpoint에서 MAE·RMSE·시간·구간을 계산했다. MAE 최소 checkpoint로 다시 선택하지 않았다. 수량 전체86285, tail1141, 이력128초과32741표본이다. Full 대비 전체 MAE는5.95%, RMSE는0.55% 낮다. MLP 대비 MAE는14.60%, RMSE는12.94% 높다. 상수 계수 결과가 없어 조건부 조절의 필요성은 아직 판단할 수 없다.

| 모델 | tail MAE ↓ | tail RMSE ↓ | 이력128초과 MAE ↓ | 이력128초과 RMSE ↓ |
|---|---:|---:|---:|---:|
| 기존 Full | 7.777805 | 9.171339 | 0.170805 | 0.363756 |
| 기존 이력 MLP | 5.274322 | 7.499710 | 0.163851 | 0.539866 |
| Full + 조건부 Gate | 6.983056 | 9.378631 | 0.207383 | 0.553992 |

Gate의 tail은 Full 대비 MAE가 낮아졌지만 RMSE는 높아졌다. 긴 이력은 두 수량 지표 모두 Full보다 나쁘다. RMSE는 큰 오차에 더 큰 가중치를 주므로 MAE와 함께 해석한다. 위 요약치만으로 개별 극단 오차 표본이나 원인을 확정하지 않는다. [3seed 및 seed별 MAE·RMSE 재확인](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_core_ablation_execution_20260928_v1/completed_5080_20260929/mae_rmse_review.md)에 전체 모델 비교와 반례를 기록했다.

**A 학습·회수·감사 — 완료**
- 81epoch 종료,41epoch 선택,249237optimizer steps. selected/last2replays와24파일 로컬 SHA 검증 완료. checkpoint byte/tensor SHA·CPU strict loading·optimizer/RNG/shuffle 상태 존재·노출·기존 Full/MLP와 겹치는 배치 prefix·검증 순서·strict선택·최초 허용 조기 종료를 감사했다. 새 GPU 실행이나 예측 재평가는 하지 않았다.
- train_one 실측 11505.963초(약3.20시간). epoch 백업/ACK 대기 포함, 별도 사전 검증과 endpoint replay는 제외한다. 전기요금은 미측정이며 RunPod사용료는0달러다.
- 원본 v1 CUDA경로 및 retry1 데이터 누락 준비 실패는 별도 증적에 남아 있으며 두 실패 모두 본학습 전이다.
- 기계판독 감사: [terminal_A_20260929T044755Z.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_change_gate_5080_20260929_retry2/verification/terminal_A_20260929T044755Z.json). 원본은 retrieved/A에 보존했다.

**상수 계수와 Taxi 비교 — 진행 중 / 다음 작업**
- 기존 supervisor가 B Intermittent 상수 계수를 배정했고 첫 전체 epoch와 로컬 백업 SHA/ACK를 확인했다.
- B→C Taxi 조건부→D Taxi 상수 순차 실행,각조건회수·재평가감사 후 실패이력 포함 최종 비교/시간 보고가 남아 있다. 원래 모델별36시간과2026-10-02 09:04:05.376781KST 공통한도를 유지한다.
