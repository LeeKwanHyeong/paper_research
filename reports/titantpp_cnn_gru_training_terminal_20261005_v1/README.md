# CNN·GRU seed42 학습 종료 관측 — RTX 원본 회수 미완료

**학습과 서버 종료 검증 — 완료**

- Taxi·Intermittent·RAF × CNN+MLP16·GRU54·CNN+GRU54, seed42의 고유9조건이 모두 완료됐다. 실패·진행·대기·미확정·미시작은 모두0이다.
- A100 원본7조건을 재사용하고 RTX 재개2조건의 scientific_success·소형 기록 SHA·selected/last full Validation·실제 소유 프로세스 종료를 확인했다. 단일seed·이종GPU 탐색이다.
- 5080은18:41:17 KST 관측을 캐시로 재사용했다. 5090의 신규 실제 관측은19:36:28 KST다. 현재 각 서버의 학습 worker 수는0이며 학습 ETA는 해당 없다.

| 데이터 | 모델 | 저장/best | 실제 완료(KST) | 전체 RMSE | MAE | Time NLL | 큰 수량 RMSE |
|---|---|---:|---|---:|---:|---:|---:|
| yellow_trip_hourly | titantpp_cnn_mlp16 | 187/147 | 2026-10-04 22:15:28 KST | 76.487614 | 24.816514 | 1.525345 | 350.269038 |
| raf_spare_parts | titantpp_cnn_mlp16 | 56/16 | 2026-10-04 19:21:19 KST | 33.856291 | 9.194703 | 3.455938 | 320.960506 |
| intermittent_frozen_5000 | titantpp_cnn_mlp16 | 47/7 | 2026-10-05 03:32:10 KST | 1.720114 | 0.713877 | 0.401519 | 8.774515 |
| yellow_trip_hourly | titantpp_gru54 | 134/94 | 2026-10-05 01:42:46 KST | 85.117724 | 26.973401 | 1.090884 | 415.824081 |
| raf_spare_parts | titantpp_gru54 | 48/8 | 2026-10-04 23:01:55 KST | 34.151962 | 9.117516 | 3.429004 | 325.963033 |
| yellow_trip_hourly | titantpp_cnn_gru54 | 166/126 | 2026-10-05 10:19:58 KST | 78.319802 | 25.231538 | 1.365979 | 370.079442 |
| raf_spare_parts | titantpp_cnn_gru54 | 49/9 | 2026-10-05 04:20:06 KST | 34.128845 | 9.284153 | 3.548010 | 327.325327 |
| intermittent_frozen_5000 | titantpp_gru54 | 50/10 | 2026-10-05 17:51:39 KST | 1.773139 | 0.690563 | 0.280780 | 9.512158 |
| intermittent_frozen_5000 | titantpp_cnn_gru54 | 52/12 | 2026-10-05 19:33:05 KST | 1.768369 | 0.705254 | 0.296848 | 8.712390 |

표는 해당 선택 epoch의 전체 Validation replay 값이다. RTX replay와 과거 A100 history의 미세한 수치 차이는 허용오차 안에 있으며 원본 선택·history를 변경하지 않았다. 두 값을 summary.json에 따로 보존했다. 큰 수량 경계는 Train 기준 Taxi3449·Intermittent187·RAF200 초과이다. Test와 예측을 열람하거나 평가하지 않았다.

**불리한 마지막 checkpoint 결과 — 보존**

| 모델 | 마지막 epoch | RMSE | MAE | Time NLL | 큰 수량 RMSE |
|---|---:|---:|---:|---:|---:|
| titantpp_gru54 | 50 | 2.990045 | 0.923164 | 0.264357 | 21.549596 |
| titantpp_cnn_gru54 | 52 | 2.353597 | 0.791555 | 0.310605 | 14.700833 |

**원본과 비용 증거 — 현재 기준선**

- A100 원본428파일 회수와 archive SHA검증, 소유 Pod삭제·목록부재는 완료다. 보수비용 약$25.82는 청구서나 계정잔액 차이가 아니다.
- archive SHA: 21eb9b095b9ff62784235cbd7e5ae7a78423bf545f69c88a2fad3d6bfaa73f92
- [원본 회수 receipt](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery/ssh_info_attempt2/retrieval_receipt.json), [삭제·목록부재](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery/cleanup_receipt.json), [비용 추정](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1/recovery/cost_estimate.json)를 재사용한다.
- RTX remote terminal manifest에 best/last binary SHA가 있으나 실제 로컬 원본 회수 receipt는 아직 없다. 원격 기록된SHA와 로컬 파일회수를 구분한다.

**RTX 종료 원본을 회수하고 SHA를 확인한다 — 다음 작업**

- 대상: 5080·5090 전용 재개 root의 종료 checkpoint·이력·terminal manifest다. 두 서버는 독립 회수가 가능하다.
- heartbeat의 한 번 관측 범위에서 추가SSH·원격회수는 실행하지 않았다. 로컬 파일과 최종 manifest SHA가 일치해야 회수완료로 판정한다.

**최종 증거를 연결하고 모니터를 종료한다 — 다음 작업**

- 새 RTX원본 회수·두 서버terminal·기존 A100회수/비용/삭제·기존 Intermittent Test완료 증거를 확인한 뒤 이 통합 모니터만삭제한다. 현재 RTX회수가 미완료라 유지한다.
- CPU binary 재추론감사는 별도 미완료다. 전체 연구감사 완료로 표현하지 않는다.

[기계 판독 가능한 요약](summary.json)


**원본 회수와 모니터 종료 — 완료**
- RTX 새 원본26개를 최종 manifest SHA로 검증하여 로컬로 회수했다. A100 회수·비용·삭제 및 기존 Intermittent Test 완료와 함께 종료 gate가 충족되어 이 학습 모니터를 삭제했다.
- 신규 CNN·GRU Test는 사용자의 후속 명시 승인에 따른 별도 동결 평가다. 이 문서의 표는 학습 종료 Validation 기록으로 유지한다. CPU 재추론 감사는 별도 미완료다.
