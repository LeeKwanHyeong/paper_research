# 승인된54조건의 시간별 관측

대상은 `search_artifacts/titantpp_architecture_contribution_dual_20261006_v1`만이다.
현재 포인터·실행 계약·source/operation closure·GPU UUID가 모두 일치하는지 먼저 확인한다.
이전 CNN+GRU, 시간 출력부 재적합, A100 및 중단된 GRU 대조를 다시 관측하지 않는다.

사용자가 요청한 시간별 GPU 보고를 유지한다. 각 회차에 다음 두 명령을 독립적으로
각각 한 번 실행하며 한쪽 조회 실패가 다른 쪽을 막지 않도록 한다.

```bash
/usr/local/bin/python3 search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/operation/monitor.py --bundle search_artifacts/titantpp_architecture_contribution_dual_20261006_v1 --host 5080
/usr/local/bin/python3 search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/operation/monitor.py --bundle search_artifacts/titantpp_architecture_contribution_dual_20261006_v1 --host 5090
```

실제 KST 관측시각·완료/실패/진행/대기/미확정 수와 현재 데이터·구조·seed,
실제 저장/best epoch를 짧은 표로 보고한다. 실제 supervisor/worker/timeout argv·PID·PPID,
GPU UUID·PID와 저장 receipt/history/timing/원본 SHA를 대조한다. 저장 전이면 첫 checkpoint가
아직 없음을 구분한다. SSH 실패는 조회 실패이며 학습 실패로 집계하지 않는다.

평가는 Validation만 읽는다. 최초 최소 raw수량RMSE epoch의 RMSE·MAE·TimeNLL을 함께
보고하고, Train큰수량경계 Taxi3449·Intermittent187·RAF200초과를 유지한다.
미완료3seed를 최종 비교 결과로 표현하지 않는다. 시간 손해와 불리한 결과도 보존한다.

해당 fit 자체 최근 최대10개 완료 epoch의 중앙값으로 현재 best가 갱신되지 않고 속도가
유지될 때 `min(300,max(40,best+40))`까지의 조건부 ETA와300epoch 시나리오를 나눈다.
자체 실측 없는 미시작 fit의 속도를 복사하지 않고 전체 큐 ETA는 미확정으로 둔다.
실제 lease 마감은 ETA가 아니며 endpoint replay·회수·CPU감사 시간은 별도다.

terminal scientific_success·selected/last fullValidation·소형 원본 SHA·실제 소유 프로세스
부재와 supervisor exit0가 확인된 조건만 완료다. 종료 서버의 cache는 원 관측시각을
보존하며 반복 SSH를 끝낸다. 모든54fit 검증 이후 binary 원본 manifest/archive SHA 회수는
별도 승인된 원본보존 작업으로 수행한다. 완료 전에 학습·qualification·retry·resume·선택
변경·프로세스 종료·claim 삭제·공용 Runtime/인증 변경·Test 접근을 추가하지 않는다.

전체54조건 terminal·두 서버 종료·새원본 SHA 회수와 최종 Validation 보고까지 검증한 뒤
이번 자동화만 종료한다. CPU binary 재추론 감사·새 독립 평가·Test는 별도 미완료다.
다른 자동화·PAUSED monitor·Pod·서버 작업·chat archive는 변경하지 않는다.
