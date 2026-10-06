# 승인된54조건의 시간별 관측

활성 heartbeat는 `titantpp-54`이며 생성 증거는 `scheduler_receipt.json`에 남겼다.

대상은 `search_artifacts/titantpp_architecture_contribution_dual_20261006_v1`만이다.
현재 포인터·실행 계약·source/operation closure·GPU UUID가 모두 일치하는지 먼저 확인한다.
이전 CNN+GRU, 시간 출력부 재적합, A100 및 중단된 GRU 대조를 다시 관측하지 않는다.

실행 계약SHA는 `19b2670327f1654923476a34c70c6eae5a065d4399098877148fafddb9bf2966`이며
현재 두 remote root의 끝은 각각 `_5080_attempt2`, `_5090_attempt2`다.
Source127 closure `2b1d69786e5988ae7c302d90c2485438c6472e845b16a129821acd92d99e9ec0`,
operation closure `f300da390b8072374705eea433106812a3cc78a5913a50028eba4fcf6c2feede`를 유지한다.
실제 과학 source revision은 `a699448f2991465b4fc17c75fafcc1b85845d1d6`이다.
기존168시간 lease 마감2026-10-13 15:20:32 KST는 준비 복구로 연장하지 않았다.

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

두 관측 뒤 아래 로컬 finalizer를 정확히 한 번 실행한다.

```bash
/usr/local/bin/python3 search_artifacts/titantpp_architecture_contribution_dual_20261006_v1/control/finalize_once.py
```

`current.json`의 `control_integrity_receipt_sha256`와 `control/integrity_receipt.json`의 두
control 코드 SHA를 먼저 확인한다. 양쪽54terminal과 실제 종료가 아직 없으면 로컬 pending만
기록하며 SSH·평가·학습은0회다. 모두 검증되면 소유 root의 manifest/archive SHA를 회수하고
동일 선택 epoch의 전체·큰 수량6지표를3seed 평균·표본SD로 집계한다. 회수 lock·claim·실패를
보존하고 자동 재시도하지 않는다. Standalone summary는 회수 원본 SHA를 재검증하는 로컬
처리이며 새 Test·GPU 재추론·checkpoint 선택을 하지 않는다.

수량 기여의 고정 기준은 두 동일예산 대조가 **같은2개 이상 데이터의 모든3seed**에서
수량 RMSE를 개선하는지다. 시간 손해를 함께 보고하며 이 기준을 통계적 유의성이나
외부모델·독립Test 우월성 증거로 바꾸지 않는다.

전체54조건 terminal·두 서버 종료·새원본 SHA 회수와 최종 Validation 보고까지 검증한 뒤
이번 자동화만 종료한다. CPU binary 재추론 감사·새 독립 평가·Test는 별도 미완료다.
다른 자동화·PAUSED monitor·Pod·서버 작업·chat archive는 변경하지 않는다.
