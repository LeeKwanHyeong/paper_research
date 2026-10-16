# v4 GPU 종료 경계 판정 독립 QA

상태: 최종 로컬 코드 PASS. bundle 및 실제 native 검증 대기. QA 직접 SSH/GPU 실행은 0이며, 5090은 별도 사용자 명령 전 모든 원격 작업 금지다.

## 현재 기준선

v2는 필수 design metadata 누락으로 qualification exit1이었다. v3는 해당 누락을 복원했고 CPU 실입력 경로까지 검증했으나 실제 concurrency admission 중 `Foreign GPU worker command`로 qualification exit1이었다. 이를 수치 학습 실패나 완료로 바꾸지 않는다. 두 root와 claims/failures, 최초 5080 lease를 보존한다.

v3 로컬 snapshot:
`search_artifacts/titantpp_cnn_gru_components_20261012_v3/control/status/5080/1e152547c4b94e3aa322e6fe9577051b.json`
raw SHA `272a308482ab284bf5244c7bec3f2537a01f5e64685633af06319da1b12c372e`.

관측 당시 GPU PID는 0, checkpoint는 없다. 실제 campaign PID39070/PPID39054의 qualify exit는 1이다. traceback은 capacity admission의 begin 이후 관측에서 campaign v3 `verify_gpu_membership`의 argv 판정 실패를 가리킨다. 종료 경계의 빈 argv가 정확한 원인이었는지는 당시 원시 프로세스 자료가 없어 확정할 수 없고, 코드상 재현과 추가 종료 원문을 별도로 확인한다.

## 확인할 조건

- [x] 소유 GPU 프로세스의 정확 argv/PPID/process group/start ticks 검증 유지
- [x] 안정된 외부 프로세스·PID 재사용·권한 또는 조회 오류를 정상 종료로 면제하지 않음
- [x] 빈 cmdline/좀비/소멸 경계는 제한된 재관측과 실제 GPU 부재 증거로만 처리
- [x] 3개 실제 GPU owner witness와 모든 실제 child exit0 조건 유지
- [x] 초기 학습 상태 조회에 같은 오류 패턴이 없는지 검증
- [x] source131 및 학습 과학 불변, 새 v4 root와 절대 lease 보존
- [x] 5090 무승인 모든 remote mode 호출0
- [x] immutable bundle SHA와 5080 승인 확인
- [ ] 실제 native 종료 원문과 첫 학습 PID/GPU/저장 상태 확인

변경 파일은 이 검토 보고서뿐이다. 원고·source·bundle·.agents 변경 0.

## 추가 원문 확인

v3 `NATIVE_FAILURE_RAW_SNAPSHOT.json` raw SHA는 `35a2c42736b58f3cbc212174ee4533c0d587656eb77f5d3c9e76d4dd7c70ddec`이며 embedded18개 SHA가 모두 일치했다. native child guard39085/parent39070은 실제 exit0이고 native receipt는 passed다. CNN-only capacity PID39181/guard39178은 passed receipt 및 실제 guard exit0를 남겼다. FULL/GRU guard39177/39180은 실패 정리 중 -15였다. 전체 admission은 통과하지 않았으며 실제 scientific fit은 0이다.

## 초기 candidate 검토 이력: WARNING → 수정 및 재검증으로 해결

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- `paper/scripts/run_titantpp_cnn_gru_component_campaign_v4.py:508` — 초기 candidate는 사라진 PID를 과거 cache에 있다는 이유만으로 소유 종료로 면제한다. 실제 독립 mock에서 `active={}`, 오래된 `observed_unix=1`의 PID777 witness, `/proc` 부재, NVML `{777}→{}`로 PASS가 재현됐다. 현재 active guard/정확 argv 및 최근 관측에 witness를 결합하고, active가 없거나 witness가 오래된 경우 실패해야 한다. 권고 형태: `require(exact_current_owner(witness, active) and 0 <= now - witness['observed_unix'] <= bound, 'Stale or foreign witness')`. worker에 전달했으며 원문 코드 수정은 QA가 하지 않았다.

### LOW
- 초기 operational reason이 정확한 원인이 확인되지 않은 종료 race를 단정했다. root가 관측 사실과 원 argv 미기록을 구분하는 표현으로 수정했다. 해결됨.

## 최종 로컬 판정: PASS

MEDIUM 문제는 현재 active owner의 guard PID/전체 command와 정확 argv, 유한한 0~60초 관측 age에 결합해 해결했다. QA의 동일 mock은 수정 후 ValueError가 됐고 전용 회귀에서도 stale/no-active/wrong-command가 거부됐다. 현재 미해결 CRITICAL/HIGH/MEDIUM/LOW 없음. 독립 총53개(기존42+새 경계11) PASS이며 native/GPU 통과로 확대하지 않는다.

초기 학습 상태 조회는 같은 순간적 불일치가 발생할 수 있지만 `training_identity_unconfirmed`로 보고하며 학습 실패나 재실행으로 바꾸지 않는다. 실제 원격 관측의 안정성은 아직 확인 전이다.
