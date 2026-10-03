# master 통합 범위와 연구 브랜치 복원 기준

2026-10-03 사용자가 master 통합 및 codex 브랜치 정리를 요청했다.
원격 fetch 이후 master는 `567196e5`, 활성 연구 정리본은 `533f45bc`다.
활성 정리본의98커밋을 fast-forward 대상으로 확정했다. 모델·손실·데이터·선택
기준·서버의 동결 실행본은 이 SCM 작업으로 변경하지 않는다.

## 활성 master에 포함하는 범위 — 통합 대상 확정

- `533f45bc`의 현재 개발 소스, 계약, synthetic 테스트, 연구 기록 및 논문 소스.
- 폭4/16과 width16 seed42·52·62 동결본, A100 routing/placement source archive.
- 폭8·12와 Encoder1 CNN＋중간 GRU는 계획으로 표시한다. 아직 구현된 후보로 등록하지 않는다.
- v0.7의 기존 기준선과 수치·과거 실험의 source SHA는 그대로 보존한다.

## 별도12커밋의 보존 방식 — 범위 확정

| 원래 branch | 고유 커밋 | 활성 코드에 채택하지 않는 근거 | 복원 tag |
|---|---:|---|---|
| codex/b1-prior-prefix-read | 9 | legacy clamped/cap300 및 joint selector를 쓰는 독립 MAC prior-prefix 후보 | archive/20261003/b1-prior-prefix-read |
| codex/raw-rmse-baseline-completion | 1 | 과거4조건 완료 runner. 동결9파일 중 현재6파일 SHA가 달라 현재 tree에서 그대로 실행 불가 | archive/20261003/raw-rmse-baseline-completion |
| codex/hard-lmm-dual-timescale | 2 | legacy clamped head용 recent8/older-prefix 전이 메모리 대안. 구현·수정이 하나의 이력 | archive/20261003/hard-lmm-dual-timescale |

고유 파일의 과학 계약을 현재 조건에 맞춰 소급 수정하지 않는다. 이12커밋을 현재
모델에 병합해 채택한 것으로 표현하지 않는다. 원래 commit과 전체 부모 이력을
복원 tag로 유지한다. `branch_inventory.json`이 모든12개 로컬 branch의 tip, 고유
커밋 목록 및 tag를 연결한다. 통합된 나머지9개 branch도 정리 전 tip을 tag로 남긴다.

## 복원 방법

master와 태그를 가져온 저장소에서 별도 작업 폴더로 복원할 수 있다.
태그에 붙은 과거 GPU 승인·permit은 그 당시 기록이며 재실행 승인이 아니다.

```sh
git fetch origin --tags
git worktree add --detach /private/tmp/titantpp-prior-prefix-restored archive/20261003/b1-prior-prefix-read
git worktree add --detach /private/tmp/titantpp-dual-timescale-restored archive/20261003/hard-lmm-dual-timescale
```

과거 전체 소스와 기록은 tag에서, 현재실험의 source 파일별 복원은
[재현성 인덱스](../../paper/reproducibility/README.md)에서 확인한다. 데이터·checkpoint
원본은 해당 계약에 기록된 별도 보관 위치와 접근권한이 필요하다.

## 통합·정리 절차

1. **로컬 복원 tag 생성 — 진행 예정:** 12개 tip과 master 이전 기준선을 기록하고 SHA를 검증한다.
2. **master 갱신·검증 — 진행 예정:** 현재 활성 소스를 fast-forward한다. 커밋 전후 과학
   소스 바이트와 source archive SHA를 대조하고 Git 추적 파일만으로 CPU 테스트를 실행한다.
3. **원격 반영·tag 검증 — 진행 예정:** master와 명시한 tag만 Push한 뒤 원격 SHA를 읽어 확인한다.
4. **codex branch 정리 — 진행 예정:** 원격2개·로컬12개를 정리한다. 원격은 정리 전 tip과
   같은 경우에만 삭제한다. 실제 경로가 없는 stale worktree2개의 Git 등록 정보만 정리한다.

stale checkout 부재는 과거 미커밋 파일을 회수했다는 증거가 아니다. 이번 작업이
보존하는 대상은 Git에 존재하는 commit 및 기존 로컬 산출물이다. 학습·Pod·GPU
프로세스는 이 정리 대상에 포함되지 않는다.

실제 완료 상태와 원격 SHA 검증은 후속 completion receipt로 기록한다.
