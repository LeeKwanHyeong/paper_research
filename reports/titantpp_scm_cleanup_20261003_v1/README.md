# 연구 브랜치의 미커밋 코드·재현 소스 정리

2026-10-03 사용자 요청: 분리된 branch의 미커밋 코드와 기록을 정리하고 이후
master에 통합할 수 있게 한다. 폭4·8·12·16 비교와 별도 A100 CNN＋GRU 계획을 구분한다.
이번 작업은 로컬 보존·계약 연결·검증이다. 원격 GPU 실행·데이터 전송·재임대·
학습 중단·새 모델 구현·성능 재분석·원고 개정은 하지 않았다.

## 현재 기준선 — 완료

- 시작 branch: `codex/hard-lmm-causal-qkv`, HEAD `15cf7eee43bf2631857dba32607a29d6eb9be80b`.
- 로컬 master 및 origin/master: `567196e510e14211e0ee400ea312872bb3ec2703`.
- 시작 시 master 대비 94커밋 앞, 뒤처짐0. 누적 825파일, 409,772삽입/76삭제.
- 원격 fetch/push는 하지 않았다. 위 원격 추적 ref는 로컬 캐시 기준이다.
- 시작 dirty: tracked 수정8개, untracked14,535개. 총14,543파일 약4.12GB의
  경로·크기·SHA를 `initial_inventory.json`에 저장했다. 결과 수치를 열어 해석하지 않았다.

## 과학 코드와 의존 관계 고정 — 완료

`778957a` — `feat(research): preserve observed-time and history-width controls`

- 기존 모델/vendor25개, runner91개, 계약50개, 테스트80개와 공유 코드7개를
  하나의 의존 단위로 고정했다. Factory의 eager import에 필요한 소스를 모두 포함한다.
- 253개 파일 모두 초기 SHA와 같아 이 정리로 과학 코드의 동작은 바뀌지 않았다.
- native NB Deep Renewal과 History MLP가 동일 loss라고 합치지 않는다.
  보존 커밋은 원고 채택 여부와 별개다.
- 원래 파일의 EOF 빈 줄 4건은 바이트 보존을 위해 유지했다. 그 외
  `git diff --check` 오류는 없었다.

## 동결 소스와 큰 원본 분리 — 완료

`3b68fb3` — `chore(repro): archive frozen campaigns with verifiable source restore`

- [재현성 인덱스](../../paper/reproducibility/README.md)와
  [캠페인 목록](../../paper/reproducibility/campaigns.json)을 추가했다.
- width16 seed42, seed52·62 및 A100 routing의 과학 소스336개를 원계약 SHA로
  검증했다. 코드·계약·운영 스크립트 등404경로를 원본 바이트로 보존했다.
- Git에서 제외한 중복 폴더의 Python·license4,511경로는143개 고유 object로 보존했다.
- 최초 캡처에서 `LICENCE` 철자가 필터에서 빠진 문제를 독립 검토가 발견했다.
  과학 manifest 전체를 먼저 선택하도록 고쳤고 336개 전부 보존됨을 재검증했다.
- 원본 데이터·checkpoint·예측·archive는 삭제하거나 이동하지 않았다. `artifact_disposition.json`
  및 `verification/original_preservation.json`이 초기 파일 보존 여부를 기록한다.
- `.gitignore`는 원본 바이너리·회수 사본·source 복제 폴더·빌드 캐시만 제외한다.
  기존 tracked 파일에 소급 삭제는 없다. aggregate 연구 기록과 논문 소스는 별도로 보존한다.
- 초기14,543파일의 SHA를 다시 확인했으며 변경0·누락0이었다.
- 원본 hash 목록과 source export는 원본 데이터 백업이 아니다. 원본의 별도 보관과
  접근권한 없이는 전체 GPU 실험을 재현할 수 없다. source 복원이 과학 결과 재현 성공은 아니다.
- 일부 native qualification receipt는 원래 bundle에만 있다. 실행 계약의 환경 기대값과
  원본 receipt를 함께 사용해야 한다. 최신 실행 상태는 이 고정 캡처가 아닌 기존 observer가 관리한다.

## 클린 checkout 문제 수정 — 완료

`9b47787` — `fix(repro): retain the root sentinel without tracking research data`

- 기존 workspace에서205개 CPU 테스트는 통과했으나, `git archive`로 만든 새 checkout은
  테스트 수집에서6오류가 났다. `simple_lab_test/common/pathing.py`가 `sample_data`
  디렉터리를 repo root 표시로 요구하는데 이 폴더가 전부 Git ignore였기 때문이다.
- 경로 탐지·모델 소스를 바꾸지 않고 `sample_data/.gitkeep`만 추적했다.
  실제 연구 데이터는 여전히 제외된다. 첫 실패 로그를 지우지 않고 수정 후 검증과 구분한다.
- 수정 commit `9b477872`의 Git 추적 파일3,148개만으로 만든 클린 checkout에서
  기존205＋archive36＝241테스트가 통과했다. 미추적 파일이나 임시 marker를 보충하지 않았다.

## 검증 증거

- `verification/source_tests.*`: 합성/mock 입력205개 통과; Python3.12/PyTorch2.14 CPU.
- `verification/archive_tests.*`: 손상·경로 충돌·symlink·복원36개 통과.
- `verification/capture_tests.*`: LICENCE·확장자·계약 검증 관련 추가6개 회귀 사례 통과.
- `verification/roundtrip.json`: 네 snapshot의4,915경로를 새 임시 폴더에 복원해 SHA 전부 확인.
- `verification/frozen_capture_audit.json`: 독립 source·계약·license·비밀정보 패턴 검사.
- `verification/clean_checkout*`: Git 파일만으로 한 클린 검증의 최초 실패와 수정 후 재검증.
- `verification/source_preservation.json`, `verification/original_preservation.json`: 기존 바이트 보존.

검증 범위는 해당 CPU 계약과 코드 복원이다. 전체 과거 스크립트·모든 GPU runtime·
원본 checkpoint CPU 감사·전체 재학습을 완료했다고 주장하지 않는다. Mac CPU 초기화와
native x86 초기화는 같다고 가정하지 않는다. secret 검사는 알려진 패턴 기준이며,
이를 공개 배포 적격성 전체 감사로 확대 해석하지 않는다.

## 후속 작업 순서

아래는 이 로컬 정리 완료 당시의 순서다. 이후 사용자가 승인한 master 통합·Push와
codex branch 정리는 [후속 통합 완료 보고서](../titantpp_master_integration_20261003_v1/README.md)에서
확인한다. 현재 남은 작업은 그 보고서의 폭 비교 계약과 별도 CNN＋GRU 설계다.

1. **master 통합 범위 확정 — 다음 작업:** 현재 `paper_research/codex/hard-lmm-causal-qkv`
   보존 커밋을 기준으로 원격 최신 ref와 별도 branch 고유12커밋을 검토한다.
   `codex/b1-prior-prefix-read`9개, `codex/raw-rmse-baseline-completion`1개,
   `codex/hard-lmm-dual-timescale`2개는 그대로 남겨두었다.
2. **master 통합 검증 — 위 확인 이후:** 깨끗한 통합 checkout에서 필요한 이력만 연결하고
   `paper_research/master` 통합 및 origin 반영 대상을 확정한다. 이번에는 master·origin을 바꾸지 않았다.
3. **폭4·8·12·16 공통 계약 — 다음 실험 준비:** 폭8·12는 미구현이다. 현재 폭16 seed42와
   seed52·62 큐의 동결본은 보존하고, 폭별 identity·동일 선택 기준·초기화·파라미터 수를
   검증한 뒤 새 계약으로 실험한다. 현재16의 결과를 보고4/8/12 계약을 소급 변경하지 않는다.
4. **A100 Encoder1 CNN＋중간 GRU — 별도 후속 설계:** 현재 routing/placement와 구분해
   CNN 참조 범위·GRU 폭·삽입 위치·상태 초기화·파라미터 예산·분리 대조를 정한다.
   이번 SCM 작업은 해당 GPU 학습의 실행 승인을 포함하지 않는다.
