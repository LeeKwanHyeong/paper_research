# RTX4090 활성 분기 수 정규화 — 최종 종료 보고

작성 기준: 2026-09-30T21:16:36.661201+09:00. validation-only, seed42. **Taxi·Intermittent 2조건 완료, Instacart 1조건은 2epoch 후 시간·비용 예약 검사로 중단, seed52·62 6조건은 사용자 보류로 미시작**이다. 캠페인 전체 성공으로 표시하지 않는다.

## 결과와 판단

동일한 raw validation RMSE 선택 checkpoint에서 MAE·RMSE·기록된 양의 정수 시간 NLL을 함께 비교했다. 세 지표 모두 낮을수록 좋다. 기존 plain History MLP의 감사 완료 seed42 결과를 재사용했다.

| 데이터 | 모델/상태 | 선택/완료 epoch | MAE | RMSE | 시간 NLL |
|---|---|---:|---:|---:|---:|
| Taxi | 기존 plain MLP | 60/기존완료 | 26.499368 | 82.483934 | 0.842735 |
| Taxi | 정규화 완료 | 129/169 | 25.361965 | 79.297432 | 1.171106 |
| Intermittent | 기존 plain MLP | 10/기존완료 | 0.638480 | 1.578487 | 0.346835 |
| Intermittent | 정규화 완료 | 12/52 | 0.712465 | 1.707505 | 0.357442 |
| Instacart | 기존 plain MLP | 30/기존완료 | 3.991502 | 5.878730 | 2.803218 |
| Instacart | 정규화 중단/잠정 | 1/2 | 4.070557 | 6.054036 | 2.801180 |

- Taxi: 정규화가 MAE4.29%·RMSE3.86%를 낮췄으나 시간 NLL은 0.842735→1.171106으로 악화됐다.
- Intermittent: 정규화의 MAE는11.59%, RMSE는8.17% 높아졌고 시간 NLL도 0.346835→0.357442로 악화됐다.
- Instacart: 표의 값은 2epoch 중 best인 epoch1의 학습 중 validation 기록이다. 최종 selected/last replay가 없고 최소40epoch도 충족하지 못했으므로 완료 성능·수렴 결과·실패한 방법의 성능으로 판단하지 않는다.
- 현재 증거는 기존 MLP를 일관되게 대체할 정규화 이득을 뒷받침하지 않는다. Taxi의 수량 개선과 시간 점수 악화, Intermittent 반례를 함께 보존한다. Instacart 효과는 미확인이다. seed42 한 개이며 기존 GPU와4090의 차이를 포함하므로 3seed 우월성으로 일반화하지 않는다.

## Instacart 중단 원인과 원본 보존
실제 worker 로그의 오류는 `Full-epoch projected300-epoch cost exceeds condition budget`다. epoch1 598.918초, epoch2 606.759초를 기록했다. 동결 규칙 `max(첫2epoch 시간) × 298 × 1.2 + 600`은 잔여 60.438시간을 예약해야 했지만, 당시 절대 마감까지 27.919시간만 남아 검사를 통과하지 못했다.
실패 기록 시각은 2026-09-30T20:27:08.142239+09:00다. 실제 지출 상한 소진·NaN·OOM이 아니라 최대300epoch 수행 여유를 확인하는 보수적 시간 검사다. early stopping으로 더 일찍 끝날 가능성을 이유로 규칙을 우회하거나 마감을 연장하지 않았다. 추가 자동 retry/resume은 수행하지 않았다.
Instacart last_epoch_state.pt에 epoch2 모델·optimizer31114steps·history·Python/numpy/torch/CUDA RNG·shuffle·epoch1 best_state가 보존되어 CPU strict load 및 tensor SHA 감사를 통과했다. standalone selected checkpoint와 endpoint replay는 생성되지 않았다. 이를 임의로 성공 산출물로 만들지 않았다.

## 무결성·선택·노출 감사 — 완료
- 원본 archive SHA 재검증 및 파일251개, 동결source104개, 원본 계약 canonical SHA 확인을 완료했다. binary checkpoint5개(Taxi2,Intermittent2,Instacart1)를 CPU로 감사했다.
 - 완료2조건의 모델 strict loading/tensor SHA·qualified 초기화·RMSE 최초 최소 선택·patience40 종료·optimizer·RNG·shuffle·전체 train/validation 노출·공통 batch prefix·selected/last 재평가 및 구간별 합산 검증이 통과했다. 재평가를 새로 실행하지 않았다.
- 마지막 epoch 저장 시점 exposure와 학습 종료의 추가 validation exposure는 구분했다. 추가 validation 한 항목을 메모리에서 제외한 prefix가 과거 epoch receipt SHA와 정확히 같고, 최종 파일은 terminal manifest SHA와 일치한다. 원본 파일은 수정하지 않았다.
- 원본 실패 로그, supervisor/worker claim, 최초 Pod의 본학습 전 배포 실패와 준비 정정 이력을 보존했다. seed52·62 run/claim은 없다.
- 감사 통과는 저장된 증적 무결성 통과이며 Instacart 학습 완료를 의미하지 않는다. GPU 재학습·재평가·추가 업데이트0회다.

## 실측 시간·메모리와 구간별 반례

| 데이터 | epoch 기록합(초) | train_one elapsed(초) | peak allocated MiB | optimizer steps |
|---|---:|---:|---:|---:|
| Taxi | 3195.508 | 3205.265 | 1869.74 | 50700 |
| Intermittent | 8835.179 | 8851.361 | 1869.74 | 160004 |
| Instacart | 1205.677 | 최종 summary 없음 | 최종 summary 없음 | 31114 |

완료 조건 selected/last의 수량·이력 구간별 count/MAE/RMSE/시간 NLL과 tail/body 값은 comparison.json의 endpoint_replays에 원본 그대로 포함한다. 단일 평균으로 구간 반례를 제거하지 않는다. 원래 시간은 겹침·대기·준비 시간을 추정하여 보정하지 않았다. 합성60updates는 과학적 epoch에 포함하지 않았다.

## 비용과 Pod 정리 — 완료 / 청구 측정 한계
Pod z7o7zma8bu0gl8은 2026-09-30T20:28:24.888966+09:00에 원본 회수·SHA확인 후 stop/delete와 목록부재 확인을 마쳤다. 첫 준비 실패 Pod bgz64ve7xq10mp도 정리됐으며, 2026-09-30T21:15:44.296921+09:00 읽기 전용 provider 재조회에서 두 소유 Pod 모두 부재를 확인했다.
초기 잔액$30 대비 현재 잔액은 $26.963337, 잔액 감소는 **$3.036663**다. 최초 실패 비용 예약$0.18을 포함한 GPU·설정 저장료 기반 보수적 계산은 약 **$3.0475**다. 계정 잔액 차감과 추정치를 구분하며, 개별 Pod 최종 itemized 청구액이라고 주장하지 않는다. 자동충전은 하지 않았다. 사용 종료 Pod를 유휴 보유하지 않는다.

## 남은 작업 순서

**추가 TPP 비교군 18조건을 마친다 — 진행 중**
- 5080/5090 기존 큐는 계속한다. 21:11 KST 관측에서5080 9/12,5090 1/6조건 완료다. RunPod 실패가 두 서버의 학습을 중단시키지 않는다.

**seed42 결과와 추가 비교군을 대조하고 RAF 후속 배분을 정한다 — 다음 작업**
- 정규화 seed52·62는 사용자 보류를 유지한다. RAF는 계획 단계이며 새 실행 예약이 아니다. 이 실패를 근거로 자동 예산 연장·재임대·재학습하지 않는다.
