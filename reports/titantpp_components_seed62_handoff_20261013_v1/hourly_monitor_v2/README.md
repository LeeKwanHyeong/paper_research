# 5090 관측 도구 수정과 실제 종료 확인

**기존 36개 실험의 종료 상태를 확인한다 — 완료**

- Intermittent seed42·52의 8개는 이미 승인된 36개 구성요소 대조에 포함된 학습이다. 새 실험이나 재학습을 추가하지 않았다.
- 2026-10-14 08:03:03 KST에 5090의 8개 scientific worker 실제 exit0와 terminal 원격 파일 104개의 SHA를 검증했다.
- dispatcher는 05:28:27 KST에 seed62 중복 실행을 막는 기존 exclusive claim에서 `FileExistsError`, 실제 exit1로 종료했다. claim·failure는 원문 그대로 보존했다. 부모의 원래 12개 전체 완료나 수치 학습 실패로 해석하지 않는다.
- dispatcher·소유 wrapper·종료 worker의 PID 10개를 명시적으로 조사해 부재를 확인했다. 고정 GPU UUID가 존재하고 compute PID도 없다. Pipeline wrapper 자체 OS exit는 직접 관측하지 않았다.

**관측 오류와 종료 판정을 수정한다 — 완료**

- 이전 status collector는 행정 종료 상태의 `active_jobs=[]`에 `.values()`를 호출했다. 같은 오류를 로컬에서 재현했으며 실제 새 snapshot에서도 해당 상태를 확인했다. 과거 실패의 stderr는 저장되지 않아 과거 traceback 자체는 복구할 수 없다.
- 새 status 전용 transport는 stdout/stderr를 decode 전에 exclusive 파일로 보존한다. 전체·보존 SHA, byte 수, truncation, 실제 SSH exit와 timeout을 구분한다. 이번 실제 조회는 양쪽 exit0, stderr 0bytes다.
- 처음 `/proc/stat`이 없는 경우와 조회 중 소멸·오류·PID 재사용을 구분한다. 진행 판정에는 안정된 live probe, 종료 판정에는 명시적인 absent probe가 필요하다. 실제 소유 PID를 조사하지 않은 빈 process 목록으로 종료를 확정하지 않는다.
- 원본 source131·input·operation·local40·monitor 봉인을 유지했다. 기존 증적 382개의 전후 SHA도 일치한다. 새 로컬 운영 파일만 추가했고 기존 학습·claim·failure·Runtime·driver·인증·원고를 변경하지 않았다.

**합성 회귀와 원격 연결을 검증한다 — 완료**

- 통합 테스트 98개와 독립 코드·원문 검토가 통과했다. 근거는 `LOCAL_TESTS_v2.log`, `LOCAL_QA.json`, `FINAL_QA.json`이다.
- 잘못된 상태/PID, 중간 소멸, start-ticks race, stderr 보존, timeout, JSON 오류, 손상 캐시의 원격 조회 전 차단, 부재 증적 누락 시 캐시 생성 차단, 정상 캐시의 SSH0 재사용을 검증했다.
- 새 부재·live gate를 테스트 프로세스에서만 제거하면 해당 음성 회귀가 실패한다. 원본 코드 파일은 변경하지 않았다.
- 첫 통합 검증에서 새 live gate와 합성 fixture가 맞지 않아 실패한 로그 `LOCAL_TESTS.log`는 보존했다. Fixture를 보완한 최종 98개 검증은 통과했다. 실제 Test 데이터로 디버깅하거나 평가하지 않았다.
- 수정 후 실제 조회는 원5090·새5080 각각 SSH1회, 원5080 종료 캐시는 SSH0이었다. 원격 쓰기는 0이다.

**남은 Intermittent62 FULL을 관측한다 — 진행 중**

현재 고유36개는 **완료35·진행1·대기0·미확정0**이다. 5080의 Intermittent62 FULL은 저장39/best13, 최종 selected binary는 아직 생성 전이다.

자체 최근10 epoch 중앙값 537.314초 기준으로 best 미갱신·현재 속도 유지 시 2시간5분22초가 남는다(10:08:26 KST). 300epoch 시나리오는 38시간57분19초가 남으며 조건 기한 잔여30시간5분41초를 초과한다. 실제 기한 만료는 아니다. 전체 큐 종료시각은 미확정이다.

기존 자동화 `cnn-gru36-checkpoint-eta` 하나를 v2 모듈과 봉인에 연결했다. 한시간 주기·동일 채팅·ACTIVE 상태를 저장 후 확인했다. 다른 자동화는 변경하지 않았다. 원5080 캐시의 원관측2026-10-13 08:27:20 KST와 새5090 행정 종료 캐시의 원관측2026-10-14 08:03:03 KST는 유지한다. 앞으로 두 캐시는 SSH0, 진행 중인 새5080만 status SSH1회로 관측한다.

Intermittent42/52는 5090/cuDNN92000, seed62는 5080/cuDNN91900이므로 seed 간 환경 차이를 명시하고 시간은 host별로 분리한다.

**전체 원본과 최종 Validation을 확정한다 — 다음 작업**

- 남은 FULL62와 새5080 dispatcher의 실제 정상 종료·소유 PID 부재·모든 terminal binary SHA가 확인되면 최종 관측 보고 후 해당 자동화 하나만 삭제한다.
- 로컬 immutable archive/manifest/binary 원본 회수와 최종 Validation 집계는 학습 종료와 별도 작업이다.
- 이번 수정에서 새 학습·재시작·retry/resume·claim/failure 삭제·원본 회수·Test 평가는 모두 0이다. 기존 평가 계약과 준비 도구는 기준선으로 유지한다.

회차 원문은 `rounds/db608ee704c2477c9037c2bbfae92535.json`, 읽기용 표는 같은 이름의 `.md`에 있다. `FINAL_QA.json`에 회차·latest·종료 캐시·검증 로그 SHA를 기록했다.
