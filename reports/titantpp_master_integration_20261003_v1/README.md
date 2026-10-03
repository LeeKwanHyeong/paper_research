# master 통합 완료와 연구 브랜치 복원 기준

2026-10-03 사용자가 master 통합 및 codex 브랜치 정리를 요청했다.
원격 fetch 이후 master는 `567196e5`, 활성 연구 정리본은 `533f45bc`다.
활성 정리본의98커밋에 통합 기록과 테스트 fixture 수정을 더해 `master`를
fast-forward했고, 검증된 코드 commit `b05c26e29ecad9d28cb5335338334d8e6c5fe6fc`를
원격 `origin/master`에 반영했다. 이후 완료 증적과 안내 문서만 추가한다.
모델·손실·데이터·선택 기준·서버의 동결 실행본은 이 SCM 작업으로 변경하지 않았다.

## 활성 master에 포함하는 범위 — 완료

- `533f45bc`의 현재 개발 소스, 계약, synthetic 테스트, 연구 기록 및 논문 소스.
- 폭4/16과 width16 seed42·52·62 동결본, A100 routing/placement source archive.
- 폭8·12와 Encoder1 CNN＋중간 GRU는 계획으로 표시한다. 아직 구현된 후보로 등록하지 않는다.
- v0.7의 기존 기준선과 수치·과거 실험의 source SHA는 그대로 보존한다.

## 별도12커밋의 보존 방식 — 완료

| 원래 branch | 고유 커밋 | 활성 코드에 채택하지 않는 근거 | 복원 tag |
|---|---:|---|---|
| codex/b1-prior-prefix-read | 9 | legacy clamped/cap300 및 joint selector를 쓰는 독립 MAC prior-prefix 후보 | archive/20261003/b1-prior-prefix-read |
| codex/raw-rmse-baseline-completion | 1 | 과거4조건 완료 runner. 동결9파일 중 현재6파일 SHA가 달라 현재 tree에서 그대로 실행 불가 | archive/20261003/raw-rmse-baseline-completion |
| codex/hard-lmm-dual-timescale | 2 | legacy clamped head용 recent8/older-prefix 전이 메모리 대안. 구현·수정이 하나의 이력 | archive/20261003/hard-lmm-dual-timescale |

고유 파일의 과학 계약을 현재 조건에 맞춰 소급 수정하지 않는다. 이12커밋을 현재
모델에 병합해 채택한 것으로 표현하지 않는다. 원래 commit과 전체 부모 이력을
복원 tag로 유지한다. `branch_inventory.json`이 정리 전12개 로컬 branch의 tip,
고유 커밋 목록 및 tag를 연결한다. 통합된 나머지9개 branch도 정리 전 tip을 tag로
남겼다. 총13개 tag(12개 branch 및 master 이전 기준선)의 원격 SHA를 검증했다.

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

## 검증과 브랜치 정리 — 완료

1. **로컬 복원 tag 생성 — 완료:** 12개 tip과 master 이전 기준선을 기록하고 SHA를 검증했다.
2. **master 갱신·검증 — 완료:** 현재 활성 소스를 fast-forward했다. 활성 정리본 대비
   production source 차이는0이다. 보존된 source archive의4,915경로 복원 검증을 재사용한다.
3. **원격 반영·tag 검증 — 완료:** `master`와 명시한13개 tag를 한 atomic Push로
   반영하고 원격 SHA를 읽어 검증했다. master 강제 Push는 하지 않았다.
4. **codex branch 정리 — 완료:** 원격2개는 예상 tip을 조건으로 삭제했다. 로컬12개는
   worktree 미사용과 원격 tag를 확인한 후 예상 SHA를 조건으로 한 원자적 ref 삭제를
   수행했다. 다른 세션이 tip을 바꾸면 삭제가 실패하도록 했다. 실제 경로가 없는
   stale worktree2개의 Git 등록 정보와 삭제한 두 branch의 추적 설정만 정리했다.

Git 파일만으로 만든 첫 clean checkout의 전체254개 CPU 테스트 중7개는 미추적
`search_artifacts/titantpp_core_ablation_20260928_v1/draft_contract.json` 의존으로
실패했다. 해당 계약 원본55,159바이트를 tracked test fixture로 보존하고 테스트의
읽는 경로만 바꿨다. 원본과 fixture SHA는 `fixture_preservation.json`에 기록했다.
수정 commit `b05c26e2`의 새 `git archive`에서 **254개 전부 통과**했다.
실패·오류·skip0이며, 원래 미추적 파일을 새 checkout에 보충하지 않았다.
최초 실패와 중간 focused247개 통과 기록도 지우지 않았다. 최종 검증 기준은
`verification/clean_master_afterfix.json` 및 해당 로그·JUnit XML이다.
소스·문서의 `git diff --check`는 통과했다. 원래 pytest 실패 로그의 traceback에
있는21개 후행 공백은 출력 원본 바이트를 보존하기 위해 그대로 남겼다.

이 검증은 CPU 합성/mock 계약 검증이며 전체 과거 테스트·GPU 재현·원본 checkpoint
CPU 감사 완료를 의미하지 않는다. 실행 중인 폭16·A100 캠페인의 동결본과 프로세스는
변경하지 않았다.

stale checkout 부재는 과거 미커밋 파일을 회수했다는 증거가 아니다. 이번 작업이
보존하는 대상은 Git에 존재하는 commit 및 기존 로컬 산출물이다. 학습·Pod·GPU
프로세스는 이 정리 대상에 포함되지 않는다.

실제 완료 상태는 [완료 receipt](completion_receipt.json), 원격 SHA는
[Push receipt](publish_receipt.json), 삭제 검증은 `remote_cleanup_receipt.json`과
`local_cleanup_receipt.json`에 기록했다. 삭제 후 로컬·원격 branch는 `master`만 남았다.

## 남은 작업 순서

1. **폭4·8·12·16 비교 기준을 확정한다 — 다음 작업:** 대상은 `paper_research/master`의
   모델·실험 계약이다. 폭8·12는 아직 미구현이다. 기존 폭4/16의 분기 가용성·고정/8·
   초기화·출력부·손실·checkpoint 선택 기준을 기준선으로 삼고, 폭별 identity와
   파라미터 수를 명시한다. 실행 중인 폭16 seed42·52·62 계약은 소급 변경하지 않는다.
2. **Encoder1 CNN＋중간 GRU를 별도 후보로 설계한다 — 병렬 가능:** 대상은 별도 A100
   후속 설계다. 현재 routing/placement6조건과 구분한다. CNN의 과거 범위·GRU 폭·
   삽입 위치·상태 초기화·파라미터 예산과 CNN/GRU 분리 대조를 확정한다.
   이 SCM 통합은 신규 GPU 실험의 실행 승인이 아니다.
