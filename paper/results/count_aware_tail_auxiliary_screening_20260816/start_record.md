# Count-aware Tail-aware Auxiliary Seed-42 e300 Start Record

- 상태: **진행 중**
- 시작 시각: `2026-08-16 06:51:10 KST`
- 실행 서버: `5080`
- tmux: `inter_tail_aux_e300_0816`
- Artifact: `search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816`
- Source revision: `2e7e99dd4a85c88599b7dcf70b529493c48af12e`
- Frozen `lambda_tail`: `0.09111380335463036`
- Variant 순서: T0-logMSE → T1-tail-shared → T2-tail-head-only
- 실행 범위: Intermittent fixed split, seed 42, 최대 300 epoch, validation-only
- Held-out test: 미사용

Fresh T0 epoch 1은 train joint `1.486691`, validation joint `0.828299`, time NLL
`0.822056`, quantity MAE `0.843643`으로 finite하게 완료됐다. 이후에는 요청 시 단회
확인한다.
