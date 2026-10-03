# 5080·5090·RunPod 종료 예상과 스케줄러 검토

관측 기준: 2026-09-30 19:16~19:17 KST. 예상 시간은 학습 진행과 완료 epoch 기록에 따른 조건부 계산이다. 학습 설정·실행 프로세스·자동화 prompt는 변경하지 않았다.

## 현재 상태와 종료 예상

| 대상 | 확인된 상태 | 현재 fit 종료: 이후 best 갱신이 없을 때 | 전체 큐 예상 |
|---|---|---|---|
| 5080 추가 TPP | 8/12조건 완료. Intermittent S2P2 seed52: 114epoch 완료, best93 | 133epoch, 9/30 19:37경 | 완료된 Intermittent seed42의 epoch 수·시간을 다음 조건에 적용하면 10/1 00:35경. 남은 미시작 조건이 각각 75~150epoch에서 종료하는 시나리오는 10/1 00:05~04:33 |
| 5090 추가 TPP | 0/6조건 완료. Instacart S2P2 seed42: 98epoch 완료, best72 | 112epoch, 9/30 20:12경 | 낮은 신뢰도의 시나리오: 10/2 09:39~10/3 04:58. 미시작 5조건이 각각 100~150epoch에서 종료하고 AttNHP 상대 속도가 다른 데이터의 관측과 비슷하다는 가정 |
| RunPod RTX4090 정규화 | seed42 pilot 1/3조건 완료: Taxi169epoch/best129. Intermittent34epoch/best12, Instacart 미시작 | Intermittent52epoch, 9/30 20:08경 | Instacart의 해당 GPU·모델 실측이 없어 전체 종료 ETA는 미확정. 전체 절대 마감은 10/2 00:22:09이며 종료 예상과 다름 |

최근 완료 20epoch의 평균 시간은 5080 63.376초, 5090 249.411초, RunPod Intermittent 173.610초다. 현재 fit의 예상은 저장된 마지막 epoch 완료시각부터 남은 epoch 시간을 계산했다. 새 best가 생기면 patience 종료시점이 늦어질 수 있다. 위 시각에는 아직 수행하지 않은 selected/last 재평가, 원본 회수 및 최종 checkpoint CPU 감사 시간을 포함하지 않는다.

5080은 현재 조건 이후 Intermittent AttNHP52, S2P2 62, AttNHP62가 남았다. 같은 데이터의 완료 seed42는 S2P2 100epoch(평균 65.599초/epoch), AttNHP 76epoch(74.477초/epoch)였다. 이를 재사용한 계산은 다른 seed의 실제 종료 epoch를 보장하지 않는다.

5090에서 아직 Instacart AttNHP 시간은 측정되지 않았다. 전체 큐 범위는 S2P2 실측 249.411초/epoch와 5080 Taxi·Intermittent에서 관측한 AttNHP/S2P2 비율 1.135~1.154를 대입한 참고 시나리오다. 데이터·GPU가 다른 상대 속도 대입이라는 한계가 있다. 추가 TPP 공통 절대 마감은 10/5 12:16:19 KST다.

`estimates.json`에 남긴 RunPod Instacart 55~75epoch·5~10분/epoch 시나리오는 실제 4090 측정이 아니다. 이를 관측에 근거한 전체 ETA로 채택하지 않는다. 현재 Intermittent 종료 후 Instacart의 실제 완료 epoch 시간이 생겨야 전체 ETA를 갱신할 수 있다.

## RunPod의 seed52·62 확대 여부

현재 동결된 확대 검사는 seed42 각 데이터의 최대 관측 epoch 시간에 300epoch, 1.2배 여유, 조건별 600초를 적용해 남은 여섯 조건의 시간을 예약한다. 성능이 좋고 나쁨으로 확대 대상을 고르지 않는다.

이미 기록된 Taxi·Intermittent만 계산해도 seed52·62의 네 조건 예약은 합계 **51.989시간**이다. Instacart 두 조건은 아직 이 합계에 포함되지 않았다. 관측 시점의 절대 마감까지 남은 시간은 **29.072시간**이다. 최대 관측 epoch 시간은 이후 감소하지 않고 남은 시간은 줄어드므로, 현 코드·계약을 유지하면 확대 검사를 통과할 수 없다.

따라서 기존 큐가 정상 완료하면 seed42 세 조건을 마친 뒤 `pilot_complete_budget_limited`로 나머지 여섯 조건을 미시작으로 기록하고 회수·Pod 정리로 진행하는 경로다. 이 예약은 최대300epoch를 감당할 여유의 검사이며, 실제로 앞으로 52시간 학습한다는 예측도 아니고 현재 학습을 즉시 중단한다는 의미도 아니다. 예산·시간 상한·학습 설정은 변경하지 않았다.

## 실제 저장된 Scheduler 검토

대상: `titantpp-tpp`, **TitanTPP·추가 TPP 시간별 비교**, ACTIVE, 1시간 간격. 실제 저장된 `automation.toml`과 앱의 view 조회를 확인했다. 이번 요청은 점검·보고로 해석하여 prompt를 수정하지 않았다.

현재 prompt는 관측 대상과 금지 범위가 명확하다. 서버별 한 회차 한 번 관측, 같은 데이터·seed의 plain History MLP 비교, 동일 RMSE 선택 epoch의 MAE·RMSE·시간 NLL 보고, 진행 중 best와 최종 replay·binary 감사 구분, 불리한 결과 보존, 중복 실행 금지, 의미 있는 변화만 통지하는 규칙이 포함되어 있다. RunPod의 기존 관리자와 서버 자체 실행도 구분한다.

다음 세 부분은 보완할 가치가 있다.

1. **ETA 산정 규칙을 추가한다 — 다음 작업.** 현재 조건의 no-new-best 종료와 전체 큐 예상, 재평가·회수·최종 감사를 구분한다. 최근 완료 epoch 실측과 가정을 함께 기록하고, 미시작 모델·데이터의 측정이 없으면 전체 ETA를 미확정 또는 낮은 신뢰도의 시나리오로 표시한다. 절대 마감을 완료 예상으로 쓰지 않는다.
2. **RunPod 확대 판정을 최신 근거로 명확히 한다 — 다음 작업.** 기존 3seed 확대는 조건부 승인임을 유지하고, 현재 최대300epoch 예약상 seed52·62 확대가 불가능한 근거와 정상 pilot 완료 후 정리 경로를 명시한다. 관측값은 최신 증거를 우선하며, 승인되지 않은 한도 연장이나 계산 규칙 변경은 하지 않는다.
3. **중단과 정리 완료의 차이를 명시한다 — 다음 작업.** 추가 TPP 프로세스를 변경하지 않는다는 원칙과 승인된 RunPod 소유 리소스의 terminal 정리를 구분한다. `stop_receipt`만 있고 `cleanup_receipt`가 없으면 원본 회수·SHA 확인과 최종 delete/목록 부재 확인이 남은 상태로 처리한다. 현재 관리자의 비용·기한 중단 분기는 stop 후 종료할 수 있으므로 저장 비용 정리까지 끝났다고 표현하지 않는다.

위 보완은 관측과 보고 규칙에 관한 것이며 현재 5080·5090·RunPod 학습을 중지하거나 다시 시작할 이유는 아니다.

## 증거

- 수치와 계산: [estimates.json](estimates.json)
- 추가 TPP 관측: `search_artifacts/titantpp_additional_tpp_20260930_v1/hourly_comparison/20260930T101656540915Z/`
- RunPod 관측: `search_artifacts/titantpp_active_branch_norm_runpod4090_20260930_v1_retry1/monitor/20260930T101739819282Z/eta_snapshot.json`
- RunPod 확대 검사: `search_artifacts/titantpp_active_branch_norm_runpod4090_20260930_v1_retry1/source/paper/scripts/run_active_branch_norm.py`
- RunPod 정리 경로: 같은 사본의 `control/manage.py`
- 실제 자동화: `/Users/igwanhyeong/.codex/automations/titantpp-tpp/automation.toml`
