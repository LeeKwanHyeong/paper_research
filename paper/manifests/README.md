# Experiment artifact manifests

현재 기준은
[TitanTPP v0.7 validation-freeze manifest](titantpp_v0_7_validation_freeze_manifest.md)다.
이 manifest는 세 데이터셋, RMTPP·THP·TitanTPP, seeds 42·52·62의 27개 validation
row와 source/generated artifact SHA-256을 고정한다. v0.7 manuscript와 그 루트
README mirror는 `publication_documents`로 별도 기록한다. Held-out test는 포함하지 않는다.

```bash
python paper/scripts/build_v0_7_paper_artifacts.py
python paper/scripts/verify_v0_7_paper_artifacts.py
```

첫 명령은 manifest와 T1–T5·F2를 생성하고 루트 README를 manuscript에서
동기화한다. 두 번째 명령은 27개 validation seed 행, 집계·paired·수량 구간 결과,
데이터 통계, 그림 형식, 원고 수치·링크, README 동일성과 모든 manifest hash를
원천 파일에서 다시 계산한다. 공유 가능한 동결 조건은
[`titantpp_v0_7_validation_freeze_verification.json`](titantpp_v0_7_validation_freeze_verification.json)의
`status`가 `PASS`이고 실패 검사가 0인 상태다.

`final_fair_artifact_manifest.*`는 v0.6의 39-run 계약 provenance다. 다음 명령은
그 과거 manifest만 재생성한다.

```bash
python paper/scripts/build_artifact_manifest.py \
  --project-root . \
  --active-run-root search_artifacts/final_fair_matched_rmtpp_thp_e300_20260805
```
