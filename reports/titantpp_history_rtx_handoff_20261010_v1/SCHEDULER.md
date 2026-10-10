# 매시간 진행·관측 범위

대상 자동화는 기존 `titantpp-instacart-63` 하나다. 2026-10-10 사용자는 미시작12조건의 RTX 분담을 승인했다. 관측만 유지하던 기존 A100 지시를 새 소유권 분담에 맞추며, 다른 자동화는 변경하지 않는다.

**현재 기준선과 봉인부터 확인한다 — 완료 / 매회 재검증**

- `TEST_SESSION_PROTOCOL.md` 0·1·9·12와 이 보고서의 README/QA/논문 작업 순서를 읽는다.
- 새 bundle의 current/contract/approval, source129/input7/op, control 과학 봉인, host별 control 및 local_tools 봉인, ownership_release/source를 stdlib SHA로 대조한다. 불일치면 원격0·덮어쓰기0·사용자 조치 필요다.
- 관측 도구 별도 봉인 `control/monitor_integrity.json` raw SHA는 `a2d761e756fa907262fd890738c55b85921c2844fd859c5631f29f38fe943394`다. 봉인 파일이 없거나 project_files/files SHA가 다르면 activation과 원격 관측을 모두 차단한다. 원래 과학·원격 operation 봉인은 유지한다.
- Test/held-out 예측·성과와 Validation/Test 혼합 파일을 열지 않는다. 데이터·loss·optimizer·selector·기존 Runtime과 원래 execution ID는 유지한다.

**양 RTX의 승인된 pipeline을 관측한다 — 진행 중**

- 승인 도우미 `paper/scripts/advance_titantpp_history_rtx_handoff.py`를 회차당 한 번 실행한다. 이는 새로운 과학 설정을 만드는 도우미가 아니다.
- 아직 배포하지 않은 5080만 읽기 전용 건강 점검을 한 번 수행한다. kernel/library mismatch·reboot-required·GPU 점유·Runtime 불일치면 배포와 학습은0이다. 재부팅·업데이트·인증·공용 Runtime 변경을 하지 않는다.
- 건강 상태·고정 UUID·기존 Runtime·유휴 상태가 확인되면 승인된 RAF62 세 조건의 새 격리 root 배포/pipeline을 한 번만 시작한다. 이전 deploy/launch intent 또는 failure가 있으면 이를 보존하고 자동 retry하지 않는다.
- 5090은 Intermittent42/52/62 세 대조군씩 실행한다. qualification과 native 동시 세 worker admission이 실제 exit0/idle를 통과한 뒤만 학습한다. 이미 시작한 root는 다시 배포·qualification·launch하지 않는다.
- 5080은 사용자 재부팅 뒤 건강·유휴·기존 Runtime을 확인하고 12:40:04 KST에 pipeline을 시작했다. 기존 launch receipt를 유지하며 이후 deploy/launch와 별도 건강 SSH는0이다. `unattended-upgrade-shutdown --wait-for-signal` 대기 서비스와 실제 패키지 업데이트/잠금은 구분한다.
- 실행한 live RTX는 관측기에서 host별 읽기 SSH1회, 미launch는 로컬 waiting/관측SSH0, 검증된 terminal cache는SSH0/원관측 시각을 유지한다.
- 학습은 min40/max300/patience40/B128, 조건36시간·host168시간 lease, fresh fit, 자동 retry/resume0이다. GPU별 동시 최대3, 같은 dataset/seed의 모든 arm을 한 GPU/Runtime에 둔다.

**A100 인계 원본과 Instacart는 로컬 증거로만 확인한다 — 완료 / 매회 로컬 재검증**

- A100은 12:17:18 KST의 실제15terminal/worker·dispatcher·pipeline/GPU PID부재와 해제 원문을 유지한다. 의도된 exclusive claim 경계의 FileExistsError·실제exit1·failure/claim을 보존하며 수치 실패로 세지 않는다. A100의 다음12조건은 시작하지 않았다.
- A100 manager/terminal_service가 기존 누적50달러 상한 안에서 immutable partial 원본을 회수했고, 12:25:29 KST에 전용 Pod `z278euk8d74nqt`와 volume `pkjlrapcbm` 삭제를 확인했다. 종료된 서비스 heartbeat에 새 live 상태를 요구하지 않는다. cleanup/partial receipt와 원본 SHA만 로컬 재검증하며 추가 SSH·API 재조회·회수 retry·새 Pod·예산 연장은0이다.
- 삭제·정지된 Pod에 다시 접속하지 않는다. 삭제된 부모 `qq2uk2dpfi2rum` 원본393개와 SHA는 로컬 검증만 한다.
- Instacart completion raw SHA `459ae9f15f6f4247584a8d47bd8e3901ab6e7ba91d6a98df3f64b09dde95cea6`와614원본·analysis/archive SHA를 재검증하며 기존 RTX observer/SSH/재학습은0이다.

**종료한 RTX 원본을 회수하고27조건을 한 번씩 집계한다 — 다음 작업 / 종료 증거 이후**

- 도우미 내부 finalizer는 host별3/9 scientific terminal, 실제 supervised dispatcher/worker exit0와 owner/GPU PID부재, qualification/admission bridge, 선택/최종 binary manifest를 확인한 뒤만 immutable 원본을 한 번 회수한다. exclusive retrieval claim/failure를 보존한다.
- A10015 + RTX12를 원래27execution ID로 집계하고 최초 strict finite 최소 full Validation raw 수량RMSE epoch의 전체/큰 수량 RMSE·MAE·TimeNLL과3seed 평균/표본SD를 보존한다. 사후 선택이나 불리한 조건 제외는 없다.
- Q1은 같은 두 개 이상 dataset 각각 모든3seed에서 full의 전체 RMSE가 두 대조군보다 엄격히 낮을 때 개발 단계 지지다. 수량 개선과 큰 수량·시간 손해를 같이 보고한다. RAF42 물리 GPU 이관과 cross-seed GPU/Runtime 차이를 표시하며 GPU 시간을 혼합하지 않는다.
- 전체 큐 ETA는 미확정이다. 각 fit 자체 최근 최대10 완료 epoch 시간 중앙값으로 best 미갱신시 종료와300epoch 시나리오를 구분한다. 미시작 arm의 속도를 복사하지 않는다. deadline은 ETA가 아니며 원본 회수·CPU 감사는 별도다.

**실험 종료 후 논문 본문을 반영한다 — 다음 작업 /27조건 원본 확인 후**

- [PAPER_WORK_ORDER.md](PAPER_WORK_ORDER.md)의 순서로 원본 검증 → Q1/여섯 지표 분석 → CNN+GRU54 중심 Results/Methods/주장 일치 → 검색 포함 여부 → 미노출 평가 계약 → 인용·부록 → 컴파일·투고 점검을 진행한다.
- 고정 검색은 대기다. A100 full을 RTX 같은 GPU mean으로 혼합하거나 빈 GPU 때문에 검색27/36개를 자동 실행하지 않는다.
- CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다.
- Instacart9/history27 실제 종료·모든 원본 SHA·최종 Validation 보고·A100 Pod/volume부재가 확인되면 최종 관측 보고 후 이 자동화만 삭제한다. partial/failed/미확정이면 유지하며 필요한 조치를 알린다.
