# A100 원본 회수와 RTX 미완료 조건 재개

2026-10-05 사용자 “진행하자” 승인에 따른 수동 이관이다. 학습 원본과 완료 결과를 보존하고 미완료 두 조건만 이어 학습한다. Instacart 및 Test 평가는 포함하지 않는다.

**원본 회수와 실제 잔여 조건 확정 — 완료**

- 원본 428개 파일과 archive SHA를 검증했다. 동결117소스·worker4가 부모 계약과 일치한다.
- Taxi CNN+GRU54는 166epoch/best126으로 완료됐다. 전체 9조건 중 7조건 terminal scientific_success, 소형 파일 SHA, selected/last 전체 Validation 재평가를 확인했다.
- A100 소유 Pod `okbl76hxfr7a9d`는 원본 회수 후 정지·삭제했고 provider 목록 부재를 확인했다.
- 첫 회수는 REST Pod 조회에 SSH주소가 없어 정지했다. 기존 pinned CLI의 `ssh info` 조회로 보완한 수동 회수 시도는 기존30분 제한 안에 완료했다. 첫 시도 증거를 보존했다.

| 완료 조건 (seed42) | 저장/best epoch | Validation RMSE | MAE | Time NLL | 큰 수량 RMSE |
|---|---:|---:|---:|---:|---:|
| yellow_trip_hourly / titantpp_cnn_mlp16 | 187/147 | 76.487614 | 24.816514 | 1.525345 | 350.269038 |
| raf_spare_parts / titantpp_cnn_mlp16 | 56/16 | 33.856291 | 9.194703 | 3.455938 | 320.960506 |
| intermittent_frozen_5000 / titantpp_cnn_mlp16 | 47/7 | 1.720114 | 0.713877 | 0.401519 | 8.774515 |
| yellow_trip_hourly / titantpp_gru54 | 134/94 | 85.117724 | 26.973401 | 1.090884 | 415.824081 |
| raf_spare_parts / titantpp_gru54 | 48/8 | 34.151962 | 9.117516 | 3.429004 | 325.963033 |
| yellow_trip_hourly / titantpp_cnn_gru54 | 166/126 | 78.319802 | 25.231538 | 1.365979 | 370.079442 |
| raf_spare_parts / titantpp_cnn_gru54 | 49/9 | 34.128845 | 9.284153 | 3.548010 | 327.325327 |

전체 Validation 수량 raw RMSE의 최초 최소 epoch를 선택했다. 표의 모든 지표는 해당 epoch의 selected replay 값이다. 큰 수량 경계는 Train 기준 Taxi3449·Intermittent187·RAF200 초과이다. 단일seed·이종GPU 탐색이며 3seed 확정 결론이나 논문 효율 측정이 아니다.

**재개 환경 검증 — 완료**

- 5080: Intermittent GRU54 원본24/best10, AdamW73,848 step; 다음 epoch25.
- 5090: Intermittent CNN+GRU54 원본12/best12, AdamW36,924 step; 다음 epoch13.
- 기존 min40/max300/patience40·손실·출력·선택 기준을 유지한다. optimizer·CPU/CUDA/Python/NumPy RNG·sampler generator를 복원하고 exposure/timing의 완료epoch prefix를 이어 기록한다.
- GPU runtime·native 초기화·반복 결정성·인과성·메모리80%상한·selected/last 전체 Validation 재평가를 통과한 receipt가 있어야 training permit을 발급한다.
- 과거 A100 selected replay에만 사전 검증한 abs/rel1e-5 허용오차를 적용하며 원본 best/history를 고치지 않는다. 새 RTX epoch의 endpoint는 원래 엄격한 허용오차를 유지한다.
- 원본 sample_data/.keep 경로 표식을 보완했다. 최초 import 실패 로그를 보존했고 동결117개 소스 바이트는 바꾸지 않았다.

**미완료 두 조건 병렬 학습 — 진행 중**

- 서버당 하나의 supervisor·worker·exclusive claim을 사용한다. 수동 재개1회만 허용하며 실패·중단 시 claim을 지우거나 자동 재시작하지 않는다.
- 조건별36시간, 전체168시간의 RTX 재개 마감과 별도 소유 process group timeout을 적용한다. 이는 종료 ETA가 아니다.
- ETA는 이관 이후 각 RTX fit에서 완료한 최근10epoch 중앙값만 사용한다. A100 epoch 속도를 섞지 않는다.

**회수·비용·감사 증거**

- 보수 비용 추정은 약 $25.82이며 청구서나 계정 잔액 차이가 아니다. GPU running segment와 전체 생존기간의 보수 저장료를 계산했다.
- archive SHA: `21eb9b095b9ff62784235cbd7e5ae7a78423bf545f69c88a2fad3d6bfaa73f92`
- contract canonical SHA: `46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae` (부모); `6581c156867c7ad1b317d1af6fb3dfffc29bcf55def953f6264ff6b2de4f6a02` (운영).
- binary 내용을 CPU에서 검사했으나 CPU 재추론 감사는 미완료다.
- 승인·계약·원본 목록·qualification·launch·hourly snapshot은 해당 search_artifacts bundle에 남긴다.

두 서버에서 원본 selected/last checkpoint의 전체 Validation 6개 지표가 허용오차 안에서 일치했다. 초기화·결정성·인과성·메모리 검증도 통과한 뒤, 전용 supervisor와 학습 worker를 시작했다.

5090의 기존 NumPy 2.1.3에서는 Train 시간 평균에 4 ULP 차이가 있었다. 전용 환경에 NumPy 2.4.4를 설치해 원본 통계와 정확히 일치시켰다. 공용 ai_env는 변경하지 않았다. 최초 실패 증거와 5080의 계약·qualification을 보존하고, 5090에만 별도 운영 계약을 적용했다.

| 서버 | 실제 관측 시각(KST) | 조건(seed42) | 저장/best epoch | 재개 epoch | worker/GPU PID | 상태 |
|---|---|---|---:|---:|---:|---|
| 5080 | 10/5 14:19:11 | Intermittent GRU54 | 24/10 | 25부터 | 6624 | 진행 중·첫 재개 checkpoint 저장 전 |
| 5090 | 10/5 14:18:35 | Intermittent CNN+GRU54 | 12/12 | 13부터 | 2694725 | 진행 중·첫 재개 checkpoint 저장 전 |

실제 fit와 GPU 프로세스, 90초 이내 progress, 계약·claim·원본 이력·작업용 checkpoint SHA가 일치한다. 이 관측 시점에는 RTX에서 새로 완료한 epoch가 0개여서 학습 ETA는 미확정이다. 이관 전 A100의 7조건 완료와 현재 2조건 진행을 합쳐 canonical 9조건을 한 번씩 집계한다.

기존 시간별 자동화 `titantpp-cnn-gru-a100-seed42`를 이 두 서버의 재개 관측으로 갱신했다. 종료가 확인된 A100에 다시 접속하지 않는다.

**잔여 두 조건의 학습·종료 검증 — 진행 중**

- 대상: 5080 Intermittent GRU54, 5090 Intermittent CNN+GRU54.
- 최초 재개 epoch 저장 이후 각 서버의 실제 완료 epoch 시간으로 조건부 ETA를 계산한다. 자동 재시작이나 checkpoint 재선택은 하지 않는다.
- 완료 조건: terminal scientific_success, 소형 기록 SHA, selected/last 전체 Validation 재평가와 소유 프로세스 종료가 확인된 상태다.

**최종 원본 회수와 결과 연결 — 다음 작업**

- RTX의 종료 checkpoint·이력·원본 manifest를 SHA와 함께 회수하고, 완료된 A100 7조건과 연결한다.
- CPU 재추론 감사는 별도 미완료 항목이다. 이 실행에서는 Test 평가를 하지 않는다.


재개 상태 복원·중복 실행 방지·최초 checkpoint 저장 전 관측을 포함한 로컬 검사 63개와 6개 하위 검사가 통과했다.

**주요 증거**

- [회수 SHA 기록](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery/ssh_info_attempt2/retrieval_receipt.json)
- [실제 7완료·2잔여 목록](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery_inventory.json)
- [5080 qualification](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/qualification/5080/receipt.json)
- [5090 qualification](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/qualification/5090/receipt.json)
- [실제 학습 PID 확인](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/initial_launch_confirmation.json)
- [Pod 삭제·목록 부재](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery/cleanup_receipt.json)
- [보수 비용 추정](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery/cost_estimate.json)
