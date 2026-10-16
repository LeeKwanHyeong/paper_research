# CNN+GRU54 3seed·시간 출력부 보정 최종 관측

회차 2026-10-06 13:45:38 KST. 신규6조건과 재사용seed42 3조건을 합쳐 원래9/9조건 완료입니다. bias2/full130은 각각 별도6조건으로 집계했습니다. 현재 대상의 실패·진행·대기·미확정은 모두0입니다.

**세 stage의 실제 종료와 회수 — 완료**

| 대상 | 실제 terminal 관측 KST | 완료/실패/진행/대기/미확정 | 원본 회수·SHA |
|---|---|---|---|
| 원래 5080 | 2026-10-06 04:37:49 | 4/0/0/0/0 | 새 selected/last·이력 archive SHA 검증 완료 |
| 원래 5090 | 2026-10-06 13:46:27 | 2/0/0/0/0 | 새 selected/last·이력 archive SHA 검증 완료 |
| 5080 bias2 | 2026-10-06 09:57:54 | 6/0/0/0/0 | 119개·archive SHA 검증 완료 |
| 5080 full130 | 2026-10-06 10:22:28 | 6/0/0/0/0 | 119개·archive SHA 검증 완료 |

5080 및 두 head stage는 원래 실제 시각을 보존한 terminal cache입니다. 5090은 이번 회차의 읽기 전용 SSH1회로 실제 종료를 확인했습니다. 각 소유 감독/worker/GPU PID 부재, supervisor exit0와 동결 source·operation·GPU UUID를 확인했습니다. 모든 학습이 종료돼 학습 ETA는 더 이상 적용하지 않습니다.

**원래9조건의 선택 epoch별 full Validation — 완료**

| 데이터 | Seed | 신규/재사용 | 실제 저장/best epoch | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰 수량 RMSE | 큰 수량 MAE | 큰 수량 TimeNLL |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Taxi | 42 | 재사용 | 166/126 | 78.319802 | 25.231538 | 1.365979 | 370.079442 | 276.911825 | 9.05606e-123 |
| Taxi | 52 | 신규 | 197/157 | 76.846316 | 24.595004 | 2.163854 | 387.421508 | 300.212940 | 4.10182e-19 |
| Taxi | 62 | 신규 | 94/54 | 82.022244 | 26.012447 | 0.875252 | 418.872949 | 334.641055 | 2.80381e-32 |
| Intermittent | 42 | 재사용 | 52/12 | 1.768369 | 0.705254 | 0.296848 | 8.712390 | 6.737986 | 2.43978e-14 |
| Intermittent | 52 | 신규 | 51/11 | 1.676304 | 0.688762 | 0.287827 | 8.518481 | 6.433376 | 0.000268649 |
| Intermittent | 62 | 신규 | 53/13 | 1.744248 | 0.712499 | 0.322681 | 8.811847 | 6.918856 | 8.5242e-06 |
| RAF | 42 | 재사용 | 49/9 | 34.128845 | 9.284153 | 3.548010 | 327.325327 | 279.662009 | 3.51547 |
| RAF | 52 | 신규 | 45/5 | 34.440158 | 9.136572 | 3.479681 | 336.726860 | 291.167499 | 3.26486 |
| RAF | 62 | 신규 | 60/20 | 33.983366 | 9.197878 | 3.611548 | 321.123869 | 273.896349 | 3.70553 |

seed42 Taxi166/best126·RAF49/best9·Intermittent52/best12는 보존 원본을 재사용했고 재학습하지 않았습니다. Intermittent42는 A100 E1–12→5090 E13–52이며 best12의 원래 상태를 유지했습니다. 신규 seed62 Intermittent는53/best13으로 종료했습니다. 최초 최소 finite full Validation raw 수량 RMSE 선택 epoch의 RMSE/MAE/TimeNLL을 같은 epoch에서 보고했습니다.

**세 데이터의3seed 평균과 변동 — 완료 / Validation 개발 결과**

| 데이터 | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰 수량 RMSE | 큰 수량 MAE | 큰 수량 TimeNLL |
|---|---:|---:|---:|---:|---:|---:|
| Taxi | 79.0628 ± 2.66675 | 25.2797 ± 0.709946 | 1.46836 ± 0.650374 | 392.125 ± 24.7344 | 303.922 ± 29.0428 | 1.36727e-19 ± 2.36819e-19 |
| Intermittent | 1.72964 ± 0.0477394 | 0.702172 ± 0.0121648 | 0.302452 ± 0.0180902 | 8.68091 ± 0.149195 | 6.69674 ± 0.245354 | 9.23909e-05 ± 0.000152703 |
| RAF | 34.1841 ± 0.233359 | 9.2062 ± 0.0741413 | 3.54641 ± 0.0659478 | 328.392 ± 7.856 | 281.575 ± 8.7931 | 3.49529 ± 0.221023 |

각 데이터seed42/52/62의 평균±표본표준편차(ddof1)입니다. 큰 수량은 원래 Train 기준Taxi3449·Intermittent187·RAF200 초과이며 모집단 전체 full Validation입니다. 원래 고정 Train N/n 가중 진단과 혼합하지 않았습니다.

**시간 출력부6조건씩의 같은 원본 checkpoint 비교 — 완료 / Validation 개발 결과**

| 단계 | 데이터·seed | 원본 selected epoch | 새 head 저장/best | 전체 TimeNLL | 큰 수량 TimeNLL |
|---|---|---:|---|---:|---:|
| bias2 | Taxi·52 | 157 | 10/0 | 2.163854 | 4.10182e-19 |
| bias2 | Taxi·62 | 54 | 10/0 | 0.875252 | 2.80381e-32 |
| bias2 | Taxi·42 | 126 | 10/0 | 1.365979 | 9.05632e-123 |
| bias2 | RAF·52 | 5 | 14/4 | 3.439869 | 3.24388 |
| bias2 | RAF·62 | 20 | 10/0 | 3.611548 | 3.70553 |
| bias2 | RAF·42 | 9 | 18/8 | 3.510662 | 3.47733 |
| full130 | Taxi·52 | 157 | 10/0 | 2.163854 | 4.10182e-19 |
| full130 | Taxi·62 | 54 | 10/0 | 0.875252 | 2.80381e-32 |
| full130 | Taxi·42 | 126 | 10/0 | 1.365979 | 9.05632e-123 |
| full130 | RAF·52 | 5 | 12/2 | 3.460704 | 3.29741 |
| full130 | RAF·62 | 20 | 17/7 | 3.563462 | 3.49167 |
| full130 | RAF·42 | 9 | 11/1 | 3.505588 | 3.26997 |

시간 보정은 각 원래 수량-selected checkpoint E0에서 시작했습니다. bias2/full130은 별도 stage이며 후자에서 전자의 선택 head를 이어 학습하지 않았습니다. full Train/Validation 수량 예측 byteSHA와 동결 weight/buffer SHA를 동일하게 보존했습니다. Taxi의 세 seed는 양쪽 모두E0를 선택해 평균TimeNLL1.468361564가 유지됐습니다. RAF 평균은원본3.546413112→bias2 3.520692913→full130 3.509917798, full130 원본 대비1.0291% 감소입니다. RAF52는 bias2 값이 full130보다 좋았다는 결과를 보존합니다.

원래9조건 표의 재사용seed42 보존 endpoint와 head stage의 E0 native replay에는 계약대로 보존된 작은 이종GPU 수치 차이가 있습니다. 두 계보를 따로 대조했으며 원래 결과를 덮어쓰거나 tolerance를 확대하지 않았습니다. head stage 안의 동일성은 같은6 checkpoint·같은 native E0와의 byteSHA 비교입니다.

**원본 회수와 Intermittent 진단 연결 — 완료**

원래9조건 scientific success·선택/마지막 전체 Validation·소형기록 SHA·checkpoint 원본 archive SHA 및 실제 양 서버 종료가 검증됐습니다. 원래 finalizer는validation_complete_originals_SHA_retrieved이고 양 head finalizer는already_finalized입니다. 로컬 Intermittent 링크는verified_original9_final_Validation_and_binary_SHA_linked이며 완료 receipt·summary·endpoint/history SHA 연결을 통과했습니다.

| 회수 대상 | archive SHA256 |
|---|---|
| 원래 5080 | f0ef6c6107735202778603ca6fa796e2e35b5fe0f81ea646e373c8c9f010faaa |
| 원래 5090 | 4d245c709e83ede1e5a984afdb3e64232348972cf925182107fd0e39c269a8ed |
| bias2 | 4b31019b26c500cded324ca1812bfb0e505c976e791caaa0fddfb3b6e9003689 |
| full130 | 1dbf6f033948813b17b8c004f8e43beffc0d276af258b30734199d7b54eb7348 |

bias2 attempt1은 데이터 상대경로 준비 실패로 optimizer update0에서 중단된 증거를 보존했습니다. 현재 attempt2의 수치적 학습 실패로 중복 집계하지 않았습니다. 기존 GRUcontrol33은 사용자 중단:완료2·부분중단2·미시작29이며 이번 원래9조건의 실패·대기로 합치지 않습니다. 삭제된A100 Pod·manager/provider와 이전 캠페인을 재조회하지 않았습니다.

**통합 시간별 관측의 종료 조건 — 완료**

원래9조건 완료/원본SHA/양서버terminal, bias2 6조건 원본SHA/5080terminal, full130 6조건 원본SHA/5080terminal 및 Intermittent 최종 연결의 모든 gate를 충족했습니다. 이 보고서 작성 후 자동화titantpp-gru-3seed만 삭제 완료했습니다. 앱 도구의 deleteStatus=deleted 응답을 automation_delete_receipt.json과 이 회차 receipt.json에 기록했습니다. 다른 자동화/PAUSEDmonitor/서버작업/Pod/채팅archive는 변경하지 않습니다.

**후속 작업을 목적에 맞춰 정렬한다**

1. **외부·강한 내부 비교군과 최종3seed 결과를 연결한다 — 다음 작업.** S2P2·외부5모델·MLP16 등과 같은 split/같은 seed의 전체·큰 수량 RMSE/MAE/TimeNLL을 비교해 수량 장점과 시간 손해를 함께 판단합니다. 이번 완료 receipt 자체는 외부 우월성 증거가 아닙니다.
2. **Taxi 시간 과신을 억제하는 대조 방법을 확정한다 — 다음 작업.** 이번 진단은 head 재적합에서 퍼짐 축소와 Validation 손해가 집중되는 이력 조건을 좁혔으며, 원래 공동 학습의 근본 원인이나 시간 손해의 불가피성을 증명하지 않습니다. Intermittent 시간 재적합·S2P2 동일 재적합은 아직 수행하지 않았습니다.
3. **동결 Test3seed와 독립 평가로 논문 주장을 검증한다 — 다음 작업.** 이번CNN+GRU54 Test3seed·CPU binary encoder 재추론 감사·새 독립평가는 별도 미완료입니다. 기존Test 노출계보를 보존해야 하며 독립 미접근 평가라고 표현하지 않습니다. 결합모델3seed만으로CNN/GRU 독립기여·synergy나 논문 효율 측정을 확정하지 않습니다.

상세 파일 경로·SHA·관측 명령 수는 [최종 증적](receipt.json)을 참조합니다.

자동화 삭제 확인 기록: 2026-10-06 13:53:09 KST. 종료한 것은 이 통합 heartbeat이며 Test3seed·CPU binary 재추론·독립평가는 별도 미완료입니다.
