# RTX 인계 관측·진행·원본 보존·최종 집계 QA

## Review Result: PASS — 발견 사항 수정 및 통합 회귀 검증 완료

상태: 로컬 QA 완료. 미해결 CRITICAL/HIGH/MEDIUM 없음. 실제 monitor_integrity 봉인 생성 및 운영 실행은 부모 세션의 다음 단계다. 이 판정은 로컬 코드 검토이며 실제 RTX 학습 완료 판정이 아니다. 소스 변경·원격 SSH·Pod API 실행은 하지 않았다. 변경 파일은 이 보고서 하나다. 프로젝트 QA 설정은 Astra/high이며 실행 환경의 모델 자체 식별은 별도로 확인하지 않았다.

검토 대상: `paper/scripts/observe_titantpp_history_rtx_handoff.py`, `advance_titantpp_history_rtx_handoff.py`, `finalize_titantpp_history_rtx_handoff.py`와 각 focused tests. 기존 배포·control QA는 `RTX_PRELAUNCH_QA.md`에 있다.

### CRITICAL

확인된 사항 없음.

### HIGH

미해결 사항 없음. 아래 두 초기 발견 사항은 담당자가 수정했고 재검증했다.

- `paper/scripts/observe_titantpp_history_rtx_handoff.py:262` — scientific terminal와 supervisor exit0만으로 terminal cache를 고정하고 pipeline 완료·최종 heartbeat·worker 실제 exit0·supervisor raw SHA 연결을 요구하지 않았다. `:286`의 캐시 영구 재사용과 finalizer `:138`의 추가 조건이 결합하면 회수가 영구 pending이 된다. 담당자 수정 완료. `classify()`는 finalizer의 `verify_terminal_snapshot()`과 같은 조건을 호출하며3종 raw hash와 완료 heartbeat 시간 순서까지 확인한다. 이전 불충분 cache는 원문·pointer bytes를 별도 보존한 뒤1회 갱신하고 유효한 terminal cache만 SSH0으로 재사용한다. 해결 방향: cache 생성 전에 `terminal = scientific_terminal and pipeline_complete_bound and final_heartbeat_complete and all_worker_exit0`를 적용하고, 증적이 불충분한 이전 cache는 완료로 재사용하지 않는다.

- `paper/scripts/finalize_titantpp_history_rtx_handoff.py:208` — 로컬 launch receipt raw SHA를 원격 launch receipt에 직접 비교했다. 배포 도구는 원격 `json.dump(r, f)`와 로컬 `indent=2`를 사용해 객체가 같아도 bytes가 다르다. 실제5090 로컬 raw SHA는 `11b2f5a79bdfe6312bb4e6416b3ad21d51b5ae52c15a4106849ef091f983863f`, 동일 객체를 원격 writer 방식으로 직렬화한 raw SHA는 `522901bf96031c801fd8d77589e2c1e8ac973c36280c290dde49a7fc11961505`로 불일치했다. 이는 첫 회수에서 실패·claim 보존으로 이어진다. 담당자 수정 및 회귀2개 완료. 적용한 해결: launch receipt는 `canonical(read(remote)) == canonical(local_object)`로 의미적 연결을 검증하고 원격 원문 raw SHA를 archive에 보존한다. SCP로 동일 bytes를 복사한 나머지 승인 파일은 기존 raw SHA 연결을 유지한다.

### MEDIUM

미해결 사항 없음. 아래 초기 발견 사항은 수정·검증 완료했다.

- `paper/scripts/advance_titantpp_history_rtx_handoff.py:29` — monitor seal 부재를 `not_yet_sealed`로 반환한 뒤 원격 단계로 진행할 수 있다. 개발 단계에서 아직 봉인하지 않은 상태였지만 자동 관측 활성 전에는 `require(path.exists(), "Monitor seal missing; remote0")`로 차단해야 한다. 현재 `path.is_file()`을 필수로 요구하고, seal이 없으면 activation·observer·finalizer 실행 전에 차단됨을 mock 회귀로 확인했다.

### LOW

확인된 사항 없음.

## 독립 검증

- observer10 + advance10 =20 focused tests PASS.
- finalizer 최초15 focused tests PASS. 실제 로컬 A100 원본15와 재개 E23/E46의 history·timing 연속성 검증 포함.
- 현재 finalizer `validate_local()`의 계약·source/input/op·local_tools 원본 SHA PASS.
- 생성된5090 remote export 코드는 Python `compile()`만 수행해 문법 PASS. 실행하지 않았다.
- 최종 observer12 + advance11 + finalizer19 = **42/42 통합 focused tests PASS**. earliest strict finite RMSE tie 선택, 선택 epoch와 다른 TimeNLL 거부, launch formatter 차이 허용·내용 변조 거부를 포함한다.
- native qualification 비용 집계는 실제 부모 원본과 합성 RTX qualification fixture로72행/4물리 GPU 소유 구분을 확인했다. 실제 RTX qualification 결과가 확인됐다는 의미는 아니다. Q1에 사후 비용 gate를 추가하지 않는다.
- 최종 수정 후 현재 기존 local_tools/source/input/op seals 및 새 generated export `compile()` 재검증 PASS. 원격 실행0.
- A10015 epoch 표시도 SHA 검증된 로컬 archived endpoints를 우선 사용하며 과거 checkpoint 관측을 최종 epoch로 오인하지 않는다.

## 수락 조건

- [x] scientific terminal3/9, actual dispatcher exit0와 pipeline OS 직접 exit 미관측의 구분
- [x] pending이면 원격 회수0, exclusive claim, 실패 원본 보존, 자동 retry 없음
- [x] archive 경로 탈출·중복·비정규 파일 거부와 모든 raw SHA 검증
- [x] 삭제된 A100 Pod 접근 없이 로컬 완료15 + RTX12 = 원래 실행27 중복 없는 집계
- [x] 동일 선택 epoch 전체·큰 수량 RMSE/MAE/TimeNLL, 3seed 평균·표본 SD, 엄격한 Q1 조건
- [x] RAF42 이관 한계와 물리 GPU별 시간 분리, CPU binary 재추론·Test·새 독립 평가 미완료 표시
- [x] 종료 캐시와 finalizer 조건 통합, 봉인 누락 차단 수정 검증
- [ ] 새 monitor_integrity 봉인 생성 및 자동화의 고정 raw SHA 연결 — 부모 세션의 운영 준비 단계
- [ ] 실제 RTX 모든 종료·원본 회수·최종 집계 — 운영 단계, 이 QA의 실행 범위 아님

Node/UI 변경이 없으므로 npm audit·타입 검사·접근성 검사는 적용 대상이 아니다. Python 단위 테스트와 stdlib 원본 해시 검증을 사용했다. `.agents/`는 수정하지 않는다.

## 남은 순서

**관측 도구 봉인을 고정한다 — 다음 작업**
- 부모 세션에서 검토된 observer·advance·finalizer와 tests를 monitor_integrity에 연결하고 stdlib SHA로 검증한다. 봉인 전 advance는 원격0으로 차단된다.

**이미 승인된 RTX 학습과 종료 보존을 확인한다 — 진행 중**
- 각 호스트 native qualification·3worker admission·실제 fit 상태를 승인된 도우미로 확인한다. 모든 종료 증적과 원본 SHA가 갖춰질 때만 최종 Validation 집계를 완료한다. QA PASS를 학습 완료로 표시하지 않는다.
