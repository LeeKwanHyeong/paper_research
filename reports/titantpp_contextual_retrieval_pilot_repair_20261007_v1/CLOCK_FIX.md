# 5080 qualification 종료 시각 검증 수정

상태: 로컬 수정·회귀검증 완료. 원격 활성화·학습 실행은 이 수정 작업에서 수행하지 않았다.

## 원인과 재현

서버 관측 UTC `2026-10-07T05:02:54.157994`가 로컬 회수 시각
`2026-10-07T05:02:52.178051`보다 약 1.979943초 앞섰다.
기존 `0 <= time.time() - observed_unix < 600` 검사는 이를 오래된 관측으로
판정했다. qualification PASS와 실제 종료 확인 이후, training permit·launch intent
생성 전에 차단된 경우다. 이 판단은 모델 성능이나 수치 발산에 관한 것이 아니다.

## 수정 범위

- `paper/scripts/deploy_titantpp_contextual_retrieval.py`의 로컬 collect/launch 검증만 수정했다.
- 새 collect는 로컬 수신 시각 `local_received_unix_ns`를 원본 readback JSON에 저장한다.
  immutable 파일명의 nanosecond timestamp와 이 값이 같아야 한다.
- latest가 가리키는 파일은 해당 host의 qualification 디렉터리 아래 timestamp JSON이어야
  하며 symbolic link는 허용하지 않는다. latest raw SHA, host, 정상 종료·GPU/PID 부재를 검증한다.
- 원격 관측 시각과 로컬 수신 시각의 차이는 절댓값 10초 이하로 제한한다.
  로컬 수신 시각 기준 age는 기존과 동일하게 `0 <= age < 600`초여야 한다.
- readback의 qualification 원본 text·parsed 내용·raw SHA를 보존된 receipt와 다시 대조한다.
- 기존 readback에는 로컬 시각 binding이 없으므로 launch에 재사용할 수 없다.
  승인된 fresh collect 이후에만 launch할 수 있다.
- 모든 검사는 training permit·launch intent를 소비하기 전에 이루어진다.

실패 claim, 동결 source126·operation·계약/current, 서버 시각, 공용 Runtime과 인증은
변경하지 않았다. 자동 retry나 qualification 재실행을 추가하지 않았다.

## 검증

명령:
`/usr/local/bin/python3 -m unittest paper.tests.test_titantpp_contextual_retrieval_deployment paper.tests.test_titantpp_contextual_retrieval_observer`

결과: **24 tests PASS**, 로컬 fabricated 파일만 사용, 원격/GPU 호출 0회.

- 약 2초 ahead·behind 및 10초 이내 시각 차이 허용.
- 600초 이상 오래된 기록, 미래의 로컬 수신 시각, 11초 이상의 서버 시각 차이, NaN 거부.
- foreign 경로·host, readback SHA 변경, qualification 원본 변경, timestamp 불일치,
  로컬 시각이 없는 legacy 기록 거부.
- stale launch가 permit·intent를 만들거나 SSH를 호출하지 않음 확인.
- 같은 deploy/observer 파일의 유사 freshness 패턴을 검색했으며 해당 직접 wallclock
  차이 비교는 이 수정 지점에만 있었다.

## 원본과 변경 SHA

수정 전 helper 원본은 `/private/tmp/pilot_deploy_before_clockfix_20261007.py`에 보존했다.

- 이전 helper raw SHA: `4fa05adde865971ecb2659964d5e4bb16adfbf84bcc707ecafb69fa9732f48ae`
- 변경 helper raw SHA: `2b6baa504c334e074d7dabbe0660dd04e5a304795bebd7ca55aeb2588b89a1fb`
- 변경 deployment test raw SHA: `ddfc4bd8ddf05a0b74e3c4a27f36dec1cd024d4caa3cfa738608a97098d3fcdf`

## 수용 기준

- [x] 작은 시각 차이와 stale/future 오류를 분리한다.
- [x] fresh 관측의 경로·timestamp·원본 receipt SHA를 함께 검증한다.
- [x] 기존 일회 실행·실패 보존 조건과 source 경계를 유지한다.
- [x] 의미 있는 local-only 회귀 테스트를 통과한다.
- [ ] 상위 작업에서 수정 helper binding을 기록하고 승인된 fresh 관측·launch를 수행한다.

실제 원격 재개 결과는 상위 작업의 별도 실행 증거로 확인해야 한다.
