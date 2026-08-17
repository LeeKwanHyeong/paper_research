# Count-aware Tail-aware Auxiliary 5080 Fresh Rerun Start Record

- 상태: **완료**
- 시작 시각: `2026-08-16 19:05:12 KST`
- 실행 서버: `5080`
- tmux: `inter_tail_aux_fresh_e300_5080_0816`
- Artifact: `search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816_5080_fresh_rerun`
- Source revision: `7de638a5c9290f79dae02a40fd22839aba9802e7`
- Frozen `lambda_tail`: `0.09111380335463036`
- Variant 순서: T0-logMSE → T1-tail-shared → T2-tail-head-only
- 실행 범위: Intermittent fixed split, seed 42, 최대 300 epoch, validation-only
- Held-out test: 미사용

이번 실행은 strict matched 비교를 위해 T0, T1, T2를 모두 checkpoint 없이 처음부터
학습한다. 5080의 기존 epoch-100 checkpoint와 5090 partial run은 보존하되 최종
comparator 입력에는 포함하지 않는다. Source checksum, Python compile, shell syntax,
frozen dataset과 split manifest, fresh artifact, tmux, CUDA process 진입을 확인했다.

Fresh T0 epoch 1은 train joint `1.486691`, validation joint `0.828299`, time NLL
`0.822056`, quantity MAE `0.843643`으로 finite하게 완료됐다.

학습은 `2026-08-17 18:21:35 KST`에 완료됐으며 결과는 `result.md`에 기록했다.
