# A100 → RTX 미시작 12조건 인계 검토

## Review Result: PASS

- 상태: 로컬 코드·테스트·봉인 접속 증적 검토 완료. 실제 원격 reservation/release/RTX 실행은 이 검토에서 수행하지 않았다.
- 검토 범위: `paper/scripts/titantpp_history_rtx_handoff.py`, `paper/tests/test_titantpp_history_rtx_handoff.py`; 동작 대조용으로 봉인된 A100 `operation/campaign.py`, `control/pipeline.py`, `control/host.py`, 할당·무결성 JSON을 읽었다.
- 변경 파일: 이 보고서만 작성. 소스, 계약, `.agents/`, 원격 파일 및 API 변경 0.
- 역할 설정: `.codex/agents/qa-reviewer.toml:6–7`에 `gpt-6-astra`, reasoning `high` 지정. 이는 프로젝트 역할 설정 확인이며 런타임 모델 식별 API로 재확인한 것은 아니다.

### CRITICAL

없음.

### HIGH

없음.

### MEDIUM

없음.

### LOW

없음.

## 확인 근거

1. **진행 중 fit을 종료하지 않는 예약** — `paper/scripts/titantpp_history_rtx_handoff.py:118`은 봉인 계약, 실제 GPU, 정확한 dispatcher argv/PID, 90초 미만 running 상태와 RAF52 활성 집합을 확인한다. 아직 시도되지 않은 12조건의 run/log/lease/worker exit/progress/claim 부재를 검사한다. 이 모듈에는 프로세스 신호·학습 호출·원격 호출이 없다.
2. **중복 실행 방지** — 같은 파일 `:99`의 예약은 임시 파일 fsync 후 `os.link`로 완전한 바이트를 배타적으로 게시한다. dispatcher가 먼저 claim을 만들면 EEXIST로 중단하며 그 claim을 덮어쓰지 않는다. 게시 후 다시 미시작 여부를 검사한다. 예약 상태는 `reserved_only_not_released`로 명시된다.
3. **자연종료 경계** — 봉인된 `search_artifacts/titantpp_contribution_history_recovery_20261010_v1/operation/campaign.py:593`의 반복문은 현재 배치의 모든 실제 child 종료와 terminal 검증을 기다린다. `:607–608`에서 GPU 및 paired exposure를 감사한 다음 배치로 넘어간다. 다음 배치의 `:582–583` exclusive claim 쓰기는 `:589` Popen보다 앞서 있으므로 RAF62 첫 claim의 예약에서 FileExistsError가 난다. 이 시점 active={}여서 `:611–620` 실패 처리의 worker stop 대상은 없다. 기존 failure와 exit1은 그대로 남는다.
4. **실행권 해제 조건** — 새 모듈 `:137`은 예약 raw SHA, 정확한 계약·Pod·GPU·이관 집합, 미시작12 부재, 실제 owner/GPU 부재, supervisor 실제 exit1, 예상 FileExistsError, active/stopped=[], pipeline dispatch failure를 확인한다. 기존10 및 복구·진행 완료5의 terminal manifest와 해당 원본 SHA를 확인해야만15완료를 반환한다. parent scientific completion을 조작하지 않는다.
5. **과학 코드와 원본 보존** — `:46–67`은 고정 canonical/control raw SHA, source/input/operation/control 파일과 부모 archive/manifest/보존 파일을 검증한다. 보정 모델·optimizer·selector를 편집하거나 다시 학습하지 않는다. RTX 새 실행의 실제 과학 동등성·native qualification은 별도 후속 검토 대상이다.

## 자동 검증과 한계

- `/usr/local/bin/python3 -m unittest paper.tests.test_titantpp_history_rtx_handoff -v`: **9/9 PASS**.
- `git diff --check -- paper/scripts/titantpp_history_rtx_handoff.py paper/tests/test_titantpp_history_rtx_handoff.py`: 오류 없음.
- stdlib SHA256으로 A100 `control/local_tools_integrity.json`의 **15개 파일 전부 일치** 확인.
- 테스트 `test_actual_sealed_dispatch_finishes_current_then_blocks_next`는 원본 dispatcher의 claim 쓰기 한 문장을 AST로 추출하여 실행한다. 전체 dispatcher와 실제 GPU worker의 자연종료 통합 테스트는 아니다. 자연종료 선후관계는 위 봉인 코드의 수동 검토로 확인했다.
- 실제 remote reserve/confirm, PID/PPID/GPU 관측 및 원본 회수는 root의 승인된 실행 단계에서 별도 증적이 필요하다. 여기서 통과를 실제 이관 완료로 해석하지 않는다.
- Node/UI 변경이 없으므로 npm audit, UI 접근성 도구는 대상 아님. 검사 대상은 stdlib 운영 Python이며 신규 패키지 설치는 하지 않았다.
- 요구된 `.agents/skills/_shared/runtime/execution-protocols/codex.md`와 사용자 skill 루트의 동일 경로는 없었다. 존재하는 oma-qa 실행 지침·self-check를 읽었고, `.agents/` 수정 금지에 따라 이 보고서만 남긴다.

## 기존 승인 Pod와 SSH endpoint의 연결 근거

`search_artifacts/titantpp_contribution_history_recovery_20261010_v1/control/allocation.json`:

- `:2–4`: recovery bundle/root 및 Pod **z278euk8d74nqt**.
- `:32–35`: IP **154.54.102.26**, SSH port **10192**, 기존 전용 SSH key 경로.
- `:37–40`: 생성 시각, RUNNING readback 기록, 사용자의 복구 승인 문구.
- 파일 raw SHA **1f97e70ae6b1c117475fc799a3b51ec356ca9dd54d18f99e1fc228fe860c9e6f**가 `control/local_tools_integrity.json`의 `files["control/allocation.json"]`과 일치한다.
- `control/local_tools_integrity.json` 자체 raw SHA: **aa2f947a39288a4d5effb059c08e1c91bdbb7656c5f1f6133db6de2ce657f96f**.
- 같은 bundle `control/host.py:33–42`가 할당 봉인과 root/예산을 검증하고, `:49–62`가 그 할당의 IP·port·key를 **root@IP** SSH/SCP 인자로 사용한다.
- `reports/titantpp_history_recovery_20261010_v1/known_hosts:1`에 `[154.54.102.26]:10192` 호스트 키가 기록되어 있다.

따라서 이 IP는 임의의 새 서버가 아니라 승인된 recovery Pod에 연결된 기존 endpoint라는 로컬 봉인 증적이 있다. 이번 검토는 provider API를 호출하지 않았으므로 현재 provider endpoint의 새 readback을 수행했다고 주장하지 않는다. 자동 승인 검토의 허용 여부는 이 보고서가 대신 결정하지 않는다.

## 수락 조건

- [x] 검토 범위 두 파일 전체 읽기
- [x] 미시작12의 범위와 기존 execution ID 보존 확인
- [x] 예약의 배타적 게시·기존 claim 무변경·경쟁 조건 테스트
- [x] 현재 배치 자연종료 후 다음 배치 차단 순서 확인
- [x] 예상 exit1/failure 보존과 owner/GPU 부재 해제 조건 확인
- [x] 과학 코드·부모 원본 SHA 검증 경로 확인
- [x] 기존 Pod → SSH endpoint 봉인 연결 근거 확인
- [ ] 실제 remote reservation/release 및 immutable 원본 회수 — 후속 실행 증적 필요
- [ ] RTX5080/5090 qualification·parallel3 admission·실제 학습 진입 — 별도 범위

검토 시 소스 SHA: `2cb00eb383d8b0f625dea2954c59cc65f1c512510414d7f36206d077cc71311b`.
테스트 SHA: `4502a5f1cbba3c92d68f4990ea1729ecb518ea1291e7da8674ee45524e7ff69b`.
