# CNN+GRU54·시간 출력부 통합 시간별 관측

회차: 10/06 12:39:40 KST. 실제 서버 관측시각은 아래 표와 같습니다. 종료 cache는 원래 실제 시각을 유지했습니다.

원래 신규6조건: 완료5, 실패0, 진행1, 대기0, 미확정0. 재사용seed42 3조건을 포함하면 8/9 완료입니다. bias2/full130은 각각 별도6조건이며 원래9조건에 중복 합산하지 않습니다.

| 대상 | 실제 KST 관측 | 완료/실패/진행/대기/미확정 | 현재 조건 | 저장/best | 조건부 종료 |
|---|---|---|---|---|---|
| 원래 5080 | 10/06 04:37:49 | 4/0/0/0/0 | Taxi·RAF CNN+GRU54 seed52/62 | 각 조건 아래 참조 | 종료 검증·cache 재사용 |
| 원래 5090 | 10/06 12:42:33 | 1/0/1/0/0 | Intermittent CNN+GRU54 seed62 | 47/13 | 10/06 13:30:59 KST |
| 5080 bias2 | 10/06 09:57:54 | 6/0/0/0/0 | Taxi·RAF seed42/52/62 | 별도 head 단계 | 종료·원본 SHA 회수 확인 |
| 5080 full130 | 10/06 10:22:28 | 6/0/0/0/0 | Taxi·RAF seed42/52/62 | 별도 head 단계 | 종료·원본 SHA 회수 확인 |

**원래 신규 학습의 저장 상태와 같은 선택 epoch의 full Validation — 진행 중**

| 서버 | 데이터·seed | 상태 | 저장/best | 전체 RMSE | 전체 MAE | 전체 TimeNLL |
|---|---|---|---|---:|---:|---:|
| 5080 | Taxi·52 | complete | 197/157 | 76.846316 | 24.595004 | 2.163854 |
| 5080 | Taxi·62 | complete | 94/54 | 82.022244 | 26.012447 | 0.875252 |
| 5080 | RAF·52 | complete | 45/5 | 34.440158 | 9.136572 | 3.479681 |
| 5080 | RAF·62 | complete | 60/20 | 33.983366 | 9.197878 | 3.611548 |
| 5090 | Intermittent·52 | complete | 51/11 | 1.676304 | 0.688762 | 0.287827 |
| 5090 | Intermittent·62 | running | 47/13 | 1.744248 | 0.712499 | 0.322681 |

5090의 Intermittent seed62: 해당 fit 최근 완료10epoch 중앙값 484.369초. best 미갱신·현재 단독worker 속도 유지 가정으로 E53까지 6epoch, 48.44분입니다. 300epoch 시나리오는 34.04시간(10/07 22:44:58KST). best 갱신/속도 변화 시 ETA가 변하며 완료 보장이나 통계적 구간이 아닙니다. endpoint replay·원본 회수·CPU 감사는 별도입니다.

**두 시간 출력부 단계의 원본·수량 보존·종료 — 완료**

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

큰 수량은 원래 Train 기준 Taxi3449·Intermittent187·RAF200 초과입니다. head6조건의 full Train/Validation 수량 예측 byteSHA와 동결 weights/buffers SHA가 유지됐습니다. 각 stage는 119개 원본/archive SHA 회수 및 controller/worker/GPU PID 부재·exit0 확인 완료입니다. attempt1 준비 경로 실패는 보존하며 현재 bias2 attempt2의 수치적 학습 실패로 중복 집계하지 않습니다.

세 단계 비교는 Taxi 평균 TimeNLL1.468361564 불변(E0 선택), RAF3.546413112→full1303.509917798(1.0291% 감소)입니다. RAF52는 bias2 선택값이 full130보다 좋았다는 결과를 보존합니다. Train fitting/Validation 개발 선택이며 독립 calibration/Test나 외부 비교 우월성의 증거가 아닙니다. bias2 README의 130개 미실행 문장은 당시 계획이고, 현재 full130 완료 receipt가 최신 증거입니다.

**원래 원본 회수와 Intermittent 최종 연결 — 외부 작업 대기**

원래 finalize_once는 pending_verified_terminals이며 원격 회수·평가를 하지 않았습니다. 로컬 진단 refresh는 pending_original_terminal_and_retrieval입니다. 원래 seed62 학습과 실제 서버 종료·새 archive SHA 회수가 미완료라 자동화 삭제 gate를 통과하지 않았습니다. 관측은 승인된9명령을 각각1번 실행했고 읽기 전용 SSH는 live5090 1회, 종료5080/bias2/full130은0회였습니다. 소유 worker2983365와 GPU UUID/PID가 일치하며 supervisor2861215 및 timeout 부모 관계도 보존됩니다. 현재 관측 실패나 학습 실패는 없습니다.

**남은 작업 순서**

1. **5090 Intermittent seed62 종료 증거와 원본을 연결한다 — 진행 중.** 기존 학습과 매시간 관측을 유지하고 scientific terminal·selected/last full Validation·SHA·PID 부재·exit0를 확인한 후 승인된 회수 gate를 적용합니다.
2. **세 데이터의 최종 Validation과 Taxi 진단을 연결한다 — 다음 작업.** 원래9조건 회수 receipt가 확보되면 승인된 로컬 연결에서 원본 endpoint/history 계보를 대조합니다. seed42 이종 GPU의 기존 수치 차이는 계약대로 분리합니다.
3. **세 stage 최종 보고 후 이 자동화만 종료한다 — 다음 작업.** 원래9조건 및 bias2/full130 종료·원본SHA·진단 연결이 모두 검증돼야 합니다. Test3seed·CPU binary 재추론 감사·새 독립 평가는 별도 미완료입니다. 다른 서버 작업·Pod·자동화·채팅 archive는 변경하지 않습니다.

원본 관측 파일과 SHA, stage receipt 경로는 [회차 증적](receipt.json)을 참조합니다. 삭제된 A100과 사용자 중단 GRU 대조33조건은 재관측·재시작하지 않았습니다.
