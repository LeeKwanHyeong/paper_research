# 현재 연구 작업의 develop 반영과 재현 패키지 정리

사용자 요청: 진행 중인 실험을 유지하면서 현재 작업을 `develop`에 올리고, 논문 재현을 위해 코드 정리를 어떤 순서로 할지 구분한다. 모델·실험 결과·원고를 수정하는 작업은 아니다.

## 연구 작업과 실행 증거를 보존한다 — 완료

- 저장소: `paper_research`, 작업 브랜치: `codex/contextual-retrieval-pilot-20261007`.
- 시작 HEAD: `017e51f39b55284336827a1ef18486ced533102c`.
- `git fetch origin` 후 원격 wildcard 추적 ref에 `develop`이 없음을 확인했다. 기존 `master`의 `90f3343775c754d4e416270fccee795e2a15efb2`는 작업 HEAD의 조상이며, 작업 브랜치는156커밋 앞·뒤처짐0이다. `origin/master`는 변경 대상이 아니다.
- 시작 미커밋은246파일이다. tracked 진척 포인터3개와 untracked 보고서·로그243개이며, 모델 코드 변경·symlink·신규 checkpoint/archive는 없다. 정확한 초기 bytes·SHA·Git blob은 [INITIAL_INVENTORY.json](INITIAL_INVENTORY.json)에 기록했다.
- 이 중 `recovery_v2/advance.lock` 하나는 디스크에 보존하고 Git에서 제외한다. lock은 실행 증거가 아닌 로컬 동시 실행 조정 파일이다. `.gitignore`에 해당 파일명 규칙만 추가하며 실패·상태·로그를 숨기는 규칙은 추가하지 않는다.
- 현재 extra42의 봉인 source987, 입력·계약·archive, 회수 제어·과거 실패를 포함한1,036경로 SHA를 [ACTIVE_EXECUTION_BINDINGS.json](ACTIVE_EXECUTION_BINDINGS.json)에 결속했다. 학습·회수 helper, SSH, 모델 로드, 평가를 실행하지 않았다.
- 최초 보호 검사에서 bundle 아래에 source 폴더가 있다고 잘못 가정한 로컬 경로 오류가 있었다. 실제 계약의 저장소 상대경로로 바로잡아 모두 확인했다. 이는 실험 실패가 아니며 최초 기록도 별도 보존했다.

## 현재 연구 내용을 새 develop에 반영한다 — 진행 중

1. 기존 보고서·실패·운영 진척 기록을 현재 연구 브랜치에 명시적 파일 목록으로 커밋한다. 가설 문서의 보존 커밋이 해당 가설의 채택·학습 승인은 아니다.
2. `paper_research`의 `develop`을 확인한 연구 commit에서 만든다. 현재 checkout과 파일을 교체하지 않으며 기존 연구 브랜치·master를 삭제하거나 이력을 다시 쓰지 않는다.
3. 새 `develop`을 `origin/develop`으로 push하고 원격 SHA와 Git blob 보존을 확인한다. 원격에 같은 이름이 새로 생겨 충돌하면 강제 push하지 않고 최신 상태를 재검토한다.

완료 commit·원격 SHA·검증은 후속 통합 receipt를 기준으로 한다. 이 문서의 계획 문장을 완료 증거로 읽지 않는다.

## 논문용 최소 실행 경로를 정리한다 — 다음 작업

[REPRODUCTION_SCOPE.md](REPRODUCTION_SCOPE.md)에 현재 구현의 의존 경로를 구체적인 파일로 정리했다. 투고 후에만 시작할 일로 미루기보다, 투고 전에도 환경·데이터·선택 규칙과 실행 경로를 명확히 해야 한다. 탐색용 파일을 대규모로 덜어내는 일은 최종 결과를 고정한 뒤 별도 release에서 수행한다.

핵심 코드와 shell 진입점 외에도 다음 내용이 필요하다.

- 최종 모델과 공통 수량·시간 head, 손실, memory, mask·padding 처리.
- 데이터 확보·전처리·split, seed, 학습 예산과 checkpoint 선택 기준.
- 고정 checkpoint의 Validation/Test 평가와 지표 집계 코드.
- 공통 조건으로 사용하는 외부 비교군 adapter와 vendor license.
- 실제 Python/PyTorch/CUDA 환경 및 의존성 고정 파일, 설정, 실행 안내.
- 코드 commit과 결과·checkpoint·데이터 manifest의 연결. Git에서 제외한 데이터나 checkpoint는 Git commit 자체로 백업되지 않는다.

현재 factory는 TPP 모듈24개를 즉시 import하며, 일부 host adapter는 기존 worker 전체 바이트 SHA에 의존한다. 미채택처럼 보이는 파일을 바로 삭제하거나 이름을 바꾸면 실행·봉인이 깨진다. 따라서 기존 연구 저장소와 봉인본을 유지하면서 **별도 재현 진입점 → 의존성 축소 → 작은 합성/계약 검사 → 정해진 재현 검증** 순으로 정리한다. 이번에는 해당 재구현이나 새 학습을 시작하지 않는다.

일반적인 재현성 참고 기준으로 [NeurIPS 공식 체크리스트](https://neurips.cc/public/guides/PaperChecklist)는 주 결과와 비교군을 재현하는 코드·데이터·정확한 명령·환경·학습 세부사항을 함께 설명하도록 안내한다. 이는 정리 범위를 판단하기 위한 참고이며 PAKDD의 별도 제출 요건을 검증했다는 의미는 아니다.

## 남은 순서

1. **develop 반영과 원격 SHA 확인 — 현재 작업:** 현재 연구 증거를 보존한 commit들을 새 `origin/develop`에 반영한다.
2. **기존 추가2seed·MAC 실험과 회수 — 외부 작업 대기:** 기존 제어와 원래 기한을 그대로 유지한다. 이 Git 작업이 중지·재시작·다른 서버 분할의 승인은 아니다.
3. **최종 논문 재현 범위 고정 — 결과 확정 후:** 어떤 모델·비교군·seed·데이터·환경을 공개 재현 대상에 포함할지 정한다.
4. **별도 release용 코드 단순화 — 후속 작업:** 위 고정 범위를 검증하는 최소 경로를 만들고, 사용하지 않는 의존성을 제거한다. 과거 실패·중간 결과·source snapshot은 연구 기록으로 보존한다.
