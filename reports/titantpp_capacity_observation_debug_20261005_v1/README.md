# 5080의 13/14 표시 원인과 복구

사용자가 2026-10-05 요청한 “왜 13/14로 되어 있는지 원인 분석해서 해결”에 대한 기록이다. 폭8·12 캠페인은 Taxi·RAF·Intermittent × 2개 폭 × seed42/52/62의 **18개 고유 조건**이다. Instacart와 A100 CNN·GRU 캠페인은 이 집계에 포함하지 않는다.

## 원인 — 확인 완료

- 5080의 마지막 정상 관측은 10/4 22:30:05 KST였다. 당시 원래 Taxi·RAF 12개와 이관한 Intermittent seed62 폭8은 완료됐고, 폭12는 저장 epoch23/best12였다. 이 보존 증거로 표시한 값이 13/14였다.
- 이후 SSH는 종료 코드255, stdout 0byte, `Operation timed out`으로 실패했다. 원격 관측기까지 도달하지 못했다. [SSH 진단](20261004T225522588908Z/diagnosis.json)
- 10/5 08:01:58 KST의 Tailscale 진단은 `peer's node key has expired`를 반환했다. 5080 Tailscale 노드 인증 키 만료가 연결 장애의 직접 원인이다. [진단 원문](tailscale_ping_receipt.json)
- 기존 관측기가 SSH 오류 내용을 보고서에 충분히 보존하지 않아, 오래된 완료 수와 현재 조회 장애를 구분하기 어려웠다. 조회 실패를 학습 실패로 집계하지 않는다.

## 완료 상태 복구 — 검증 완료

실시간 Tailscale peer 기록의 내부망 주소로 같은 5080에 한 번 접속했다. 기존 `100.118.62.81` SSH 호스트 키를 `HostKeyAlias`로 사용하고 `StrictHostKeyChecking=yes`를 적용했다. 서버 GPU UUID, 운영 계약, 동결 소스 및 원격 root를 기존 관측기 기준으로 확인했다. SSH 설정·인증·학습·선택 기준은 변경하지 않았다. [접속 증거](lan_probe/receipt.json), [관측 원본](lan_probe/snapshot.json)

실제 관측 시각은 **10/5 08:02:35 KST**이며, 서버의 두 조건 완료 기록 시각은 **10/4 23:30:09 KST**이다. 원래 관측 시각을 새 시각으로 바꾸지 않는다.

| 완료 범위 | 검증된 완료 | 증거 |
|---|---:|---|
| 5080 원래 Taxi·RAF | 12/12 | 보존된 terminal snapshot 및 SHA |
| 5080 이관 Intermittent seed62 | 2/2 | 새 관측, terminal SHA, selected/last 전체 Validation replay |
| 5090 소유 Intermittent seed42/52 | 4/4 | 보존된 terminal 및 승인된 이관 경계 |
| 고유 조건 합계 | **18/18** | 중복·누락 없는 finalizer gate |

마지막 두 조건의 선택 값은 같은 selected checkpoint의 전체 Validation 지표이다.

| Intermittent seed62 | 저장/선택 epoch | RMSE | MAE | Time NLL |
|---|---:|---:|---:|---:|
| 폭8 | 45/5 | 1.660997 | 0.724478 | 0.384338 |
| 폭12 | 52/12 | 1.727200 | 0.686524 | 0.397907 |

[18조건 완료 gate](../../search_artifacts/titantpp_history_capacity_handoff_20261004_v1/test_finalizer/gate_receipt.json)

## 관측기 수정 — 완료

`paper/scripts/observe_titantpp_capacity_handoff_hourly.py`가 SSH 종료 코드와 허용된 오류 종류를 `observation_error.json`, analysis, receipt, report에 보존하도록 수정했다. stdout 및 원문 stderr 대신 고정된 오류 표현만 기록한다. 단일 SSH, terminal SHA·replay 검증 및 원래 관측 시각 보존 규칙을 유지한다.

- 새 회귀 테스트 15개와 관련 테스트 총51개 통과.
- `git diff --check` 통과.
- 복구 후 공식 관측기를 다시 실행해 **SSH 호출0회**로 완료 증거를 재검증했다. 다음 정기 관측도 완료 캐시를 사용하므로 이 캠페인의 5080에 다시 접속할 필요가 없다.
- Tailscale 인증 자체는 갱신하지 않았다. 이후 외부망에서 5080에 새로 접속하려면 노드 재인증이 필요하다. 이번 완료 확인과 승인된 후속 평가는 동일 peer의 내부망 경로를 사용했다.

## 승인된 Intermittent 평가 연결 — 완료

기존 [사용자 승인](../titantpp_capacity_test_handoff_20261004_v1/approval.json)에 따라 18조건 완료 gate 통과 후 5090 소유4개와 5080 소유2개의 immutable 원본 파일을 회수하고 manifest의 SHA를 검증했다. 전용 5080 평가 폴더에 배치·사전 검증·실행 요청을 완료했다. [실행·전송 receipt](test_lan_execution.json)

10/5 08:06:41 KST에 **전체 Validation 6개 gate 통과, Test 첫 조건 완료 및 다음 조건 진행**을 실제로 확인했다. [실제 실행 관측](test_lan_collection.json)

이후 [완료 확인](test_lan_completion_check.json)에서 **6조건 × Validation/Test의 12개 전체 모집단 평가**가 모두 완료됐으며, Validation 선행 gate·각 receipt의 SHA·전체 완료 receipt를 검증했다. [finalizer 완료 기록](../../search_artifacts/titantpp_history_capacity_handoff_20261004_v1/test_finalizer/completion.json)에 결과 경로를 보존했다. 완료 후 기존 finalizer를 다시 실행해 `reused_completed_evidence=true`, `remote_calls=0`을 확인했다. 다음 정기 회차에도 평가를 재실행하거나 만료된 SSH 경로에 접속하지 않는다.

동결한 Validation 선택 checkpoint를 재사용하며 재학습·checkpoint 재선택·자동 retry는 하지 않는다. 기존 Test 분할 재평가이며, 새 독립 미접근 평가라고 표현하지 않는다. 이 기록은 학습 완료·파일 SHA 회수와 CPU 재추론 감사를 구분하며, 전체 연구 감사 완료를 주장하지 않는다.

## 남은 순서

1. **A100 캠페인의 종료 증거를 확인한다 — 외부 작업 대기**: CNN·GRU 9조건의 terminal, 원본 회수·비용·Pod 정리를 확인한 후 통합 모니터 종료 여부를 판단한다. 폭8·12 학습18조건과 Intermittent 후속 Test6은 완료된 현재 기준선이다.
2. **후속 원격 실험의 접속 환경을 복구한다 — 외부 작업 대기**: 이후 외부망에서 새 5080 작업이 필요하면 Tailscale 노드를 재인증한다. 이번 작업에서 인증정보를 변경하지 않았다.

현재 원고, A100 계약, 다른 서버 작업, 인증정보 및 Git 커밋·Push는 변경하지 않았다.
