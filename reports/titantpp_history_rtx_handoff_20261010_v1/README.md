# CNN+GRU54 이력 보정: 미시작 12조건 RTX 인계

2026-10-10 사용자는 이력 보정 27조건 중 아직 시작하지 않은 조건을 5080/5090에 분담하고, 완료 후 논문 작업 순서를 재정렬하도록 승인했다. 과학 설정과 원래 execution ID는 그대로 유지한다.

## 현재 상태

| 주제/서버 | 상태 | 작업 범위 |
|---|---|---|
| Instacart | 완료 | 구조 3개 × seed42/52/62, 원본·Validation 집계 9개 재사용 |
| 이력 보정 A100 | 15개 완료 / 원본 검증·자원 정리 완료 | 부모 완료10 + 복구RAF42 두 개 + RAF52 세 개 |
| 이력 보정 5090 | 진행 3 / 대기 6 | Intermittent seed42/52/62 × full/disabled/current_only 9개 |
| 이력 보정 5080 | 진행 3 | RAF seed62 세 개. 사용자 재부팅 후 driver595.99.02·GPU 유휴·기존 Runtime 확인, 12:40:04 KST에 시작 |
| 고정 검색 | 대기 | 공통 비교 환경과 논문 포함 여부 확정 전 자동 실행하지 않음 |

5090 pipeline 시작은 2026-10-10 12:21:12 KST다. native qualification 및 동시 세 worker admission 모두 PASS/actual exit0/idle를 통과했다. 12:22:25 KST의 첫 관측은 checkpoint 저장 전이었다. 이후 [12:43 KST 관측](STATUS_20261010_124331KST.md)에서 양 GPU 각각 실제3fit 동시 실행과 checkpoint 저장을 확인했다. 5080 full/보정 제거/current_only 저장·best는8/8·17/8·17/8, 5090은2/1·4/4·4/4다. 전체27 중 완료15·진행6·대기6, 수치 실패·미확정0이며 qualification은 학습으로 세지 않는다.

사용자는 이후 5080 재부팅 완료를 알리고 계속 실행하도록 승인했다. 12:38:33 KST에 kernel7.0.0-38/driver595.99.02·기존 Runtime·고정 UUID·compute PID0을 확인했다. `unattended-upgrade-shutdown --wait-for-signal`은 PPID1의 대기 서비스이며, dpkg/apt 네 개 잠금은 모두0이었다. 실제 패키지 작업과 구분한 근거는 [5080_POSTREBOOT_MAINTENANCE.json](5080_POSTREBOOT_MAINTENANCE.json)에 보존했다. 공용 driver/Runtime 변경은 수행하지 않았다.

## 인계와 원본 보존

- 승인 A100 Pod `z278euk8d74nqt`, GPU `GPU-17e020b0-7bbe-d835-af19-881e511f9a2a`의 기존 RAF52 worker는 자연종료했다. GPU/소유 프로세스 부재와 15개 terminal의 원본 SHA를 12:17:18 KST에 확인했다.
- 미시작 RAF62 첫 claim 위치에 새로운 **행정 인계 예약** 하나를 exclusive 방식으로 생성했다. 기존 dispatcher는 현재 batch의 종료·원본 감사 후 다음 claim에서 `FileExistsError`로 멈췄다. 실제 exit1·failure·claim을 보존한다. 이를 새로운 수치 실패나 부모27개 완료로 바꾸지 않는다. 진행 중 worker 종료 신호는 0이다.
- 이관 12개는 A100에 run/log/lease/worker-exit/progress 원본이 없음을 확인했다. 5080 RAF62와 5090 Intermittent42/52/62는 원래 execution ID를 한 번씩만 사용한다.
- `a100_reservation_stdout.json` 원본 SHA는 `99aee91896df7c35996a7ca6ece08595e53780b40e1418aee3a11a4f1f536b7a`, 해제 원문 `a100_release_stdout.json` SHA는 `e7d6fbaf59b1caa7763d2662ac1ed29d65f003f6846f8bc3aaff96fa6d3a8ea1`이다. 원격 claim SHA는 `607cd6721d3c6bbfca7817d63e5eb0b329e1caf9b25e66add1718e8728ed41b8`이다.
- 기존 A100 manager/terminal_service가 immutable partial 원본을 회수하고 12:25:29 KST에 전용 Pod `z278euk8d74nqt`와 volume `pkjlrapcbm` 삭제를 확인했다. 삭제된 두 Pod에는 다시 접근하지 않는다. 부모27개 전체 범위의 partial receipt는 그대로 보존한다.
- [A100_ORIGINALS_VERIFIED.json](A100_ORIGINALS_VERIFIED.json)에 로컬 manifest/archive·모든 원본과 15개 선택/최종 binary의 SHA 및 실제 저장/best epoch를 검증해 남겼다. 누적 비용은 저장 포함 추정 $32.3624이며 최종 청구액은 미확인이다. 현재 A100 자원 비용은 종료됐다.

## 새 계약과 실행 환경

새 bundle: `search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1`.

- canonical: `518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b`
- source129 closure: `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`
- frozen science revision: `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`
- operation closure: `8a7ff350488fe8b03289ea6e6683a41ace4c95c2bdf0d24f2c06f3d78f5c83da`
- 최초 과학 봉인 raw SHA: `1fce3395f77f05af3b79fe54e39fddf593d555d18145bea5c237c27deae5caa9`
- 최종 local tools 봉인 raw SHA: `9349554f61f5861324dd32f1d1ea6a447eb7f0fc5ea3c26e3dc02eaebc8f1c51`
- 5090 remote control 봉인 raw SHA: `a2821c35778c29dea84044ad77316a5555dc834739f0e6831a8b22beea67fcca`
- 5080 remote control 봉인 raw SHA: `b6a302042d67bbfd9e621a8e11bab074f4ce6c0d19aa4485246059661ceabdf0`

원격 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_<host>`다. 5080 UUID는 `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, 5090은 `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`다. 기존 ai_env/5090 전용 resume runtime_env를 유지한다. 5090 cuDNN92000과 5080 cuDNN91900을 구분한다. 공용 Runtime·driver·인증을 변경하지 않는다.

source129/input7와 qualification·과학 함수는 원본 bytes를 유지한다. 변경한 부분은 host 배정, 격리 경로, 동시 세 worker 운영과 인계 확인이다. 활성 보정 파라미터 full/current_only25308, 현재 hidden의 prefix 이력과 계산 경로·깊이·활성화 차이의 한계를 유지한다.

min40/max300/patience40/B128, 최초 strict finite 최소 full Validation 전체 raw 수량 RMSE epoch 선택을 유지한다. host168시간/조건36시간 lease다. 자동 retry/resume·예산 연장·새 Pod는 없다. 5080도 고정 UUID·기존 Runtime·유휴 상태 확인 후 qualification pipeline을 한 번만 시작했다.

## 검증과 해석

배포 전 QA에서 실행권 해제 proof의 identity/hash 검증을 보강했다. 초기 봉인은 `control/preserved_prelaunch`에 bytes 그대로 보존하고, 실제 배포 전 local tool SHA만 명시 재봉인했다. remote control/package/operation 원본은 바뀌지 않았다. [A100 인계 QA](HANDOFF_QA.md), [RTX 배포 전 QA](RTX_PRELAUNCH_QA.md)를 참조한다.

관측/진행/회수 도우미와 회귀 테스트를 별도 `control/monitor_integrity.json`에 봉인했고 raw SHA는 `a2d761e756fa907262fd890738c55b85921c2844fd859c5631f29f38fe943394`다. 원래 배포 과학/운영 봉인은 변경하지 않았다. [종료·회수 QA](FINALIZER_QA.md)의 두 종료 연결 문제를 수정 검증했고 전체80개 테스트가 통과했다. 기존 `titantpp-instacart-63`만 양 RTX 관측으로 갱신·저장 재확인했다. 현재 finalizer는 양host pending/remote0이며 과학 완료를 앞당겨 표시하지 않는다.

Q1은 같은 두 개 이상 dataset 각각 모든 3seed에서 full이 두 대조군보다 전체 수량 RMSE를 엄격히 개선할 때 개발 단계 지지다. 같은 선택 epoch의 전체/큰 수량 RMSE·MAE·TimeNLL을 함께 보고한다. 불리한 조건과 시간 손해도 보존한다. 물리 GPU·Runtime별 시간을 분리하며 RAF42의 A100 이관 예외를 표시한다. CPU 재추론 감사·Test·새 독립 평가는 별도 미완료다.

논문 작업 순서는 [PAPER_WORK_ORDER.md](PAPER_WORK_ORDER.md)에 정리했다. 모든 27개가 끝난 뒤 원본 검증 → 보정 판정 → CNN+GRU54 중심 본문 반영을 먼저 진행한다.
