# Count-aware Tail-aware Auxiliary Seed-42 e300 Start Record

- 상태: **진행 중**
- 재시작 시각: `2026-08-16 09:40:47 KST`
- 실행 서버: `5090`
- tmux: `inter_tail_aux_e300_5090_0816`
- Artifact: `search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816_5090_rerun`
- Source revision: `7de638a5c9290f79dae02a40fd22839aba9802e7`
- Frozen `lambda_tail`: `0.09111380335463036`
- Variant 순서: T0-logMSE → T1-tail-shared → T2-tail-head-only
- 실행 범위: Intermittent fixed split, seed 42, 최대 300 epoch, validation-only
- Held-out test: 미사용

5080의 최초 실행은 `2026-08-16 06:51:10 KST`에 시작했으며 T0 epoch 100까지
finite하게 진행된 뒤, 서버 변경 요청에 따라 중단했다. 기존 partial artifact
`search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816`은 삭제하지 않았다.

5090에서는 5080 checkpoint를 이어받지 않고 T0부터 fresh rerun한다. Source checksum,
Python compile, shell syntax, frozen dataset과 split manifest, CUDA process 진입을
확인했다. 이후에는 1시간 heartbeat에서 단회 상태만 확인한다.
