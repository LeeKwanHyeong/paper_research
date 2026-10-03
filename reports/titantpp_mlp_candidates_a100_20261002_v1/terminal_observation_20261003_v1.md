# A100 후보 6조건 종료 관측 — 2026-10-03

실제 종료 관측 2026-10-03T04:47:38+09:00의 보존 snapshot을 사용했다. 05:02 회차의 새 원격 관측이 아니다.

완료6 / 실패 terminal0 / 진행0 / 대기·미시작0. 최초 두 Intermittent 비용 gate 중단과 승인된 재개는 같은 두 조건의 이력으로 보존한다.

| 데이터 | 후보 | 저장/best | MAE | RMSE | 시간 NLL | MLP 대비 RMSE |
|---|---|---:|---:|---:|---:|---:|
| Taxi | cross_product | 151/111 | 26.654178 | 82.435172 | 1.121980 | -0.059% |
| Taxi | recent4_mean | 135/95 | 27.646771 | 87.371109 | 1.115452 | +5.925% |
| RAF | cross_product | 93/53 | 9.362425 | 34.200336 | 3.640114 | +0.888% |
| RAF | recent4_mean | 56/16 | 9.151168 | 34.479366 | 3.520884 | +1.711% |
| Intermittent | cross_product | 41/1 | 0.885260 | 1.993651 | 0.507090 | +26.301% |
| Intermittent | recent4_mean | 108/68 | 0.677873 | 1.634571 | 0.524067 | +3.553% |

모든 지표는 최초 최소 raw 수량 RMSE로 선택된 동일 epoch의 validation replay다. selected/last replay, terminal manifest와 소형 기록 SHA를 확인했다. 기존 MLP를 재학습하지 않았다.

두 후보가 기존 MLP를 일관되게 대체한다는 근거는 없다. Taxi cross-product는 RMSE가 약0.059% 낮지만 MAE·시간 NLL은 높고 나머지5조건은 RMSE가 높다. 단일 seed·이종 GPU 탐색이며3seed 결론이나 효율 비교로 사용하지 않는다.

원본449파일 및 archive SHA를 로컬에서 다시 확인했다. 2026-10-03T04:48:04+09:00 Pod삭제·목록부재 receipt가 있다. 준비 시도를 합친 저장료 포함 기록 요율 추정액은 $15.0481, 승인상한 $150 이내다. 이는 개별 청구서가 아니다.

Taxi·RAF4조건의 기존 CPU감사는 재사용한다. 새 Intermittent2조건 checkpoint CPU감사는 별도 미완료다. 전체 연구 감사 완료로 표시하지 않는다.

원래 provider stopAfter 연장 적용은 미확인이었으나 이번 소유 Pod삭제가 확인됐다. 삭제된 Pod를 다시 조회하지 않았다.

원본: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_mlp_candidates_a100_20261002_v1_retry1/retrieved/original`
회수: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_mlp_candidates_a100_20261002_v1_retry1/retrieved/retrieval_receipt.json`
삭제: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_mlp_candidates_a100_20261002_v1_retry1/control/cleanup_receipt.json`
기준선 원본 연결: `/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed22_audit_20261002_v1/comparison.json`
