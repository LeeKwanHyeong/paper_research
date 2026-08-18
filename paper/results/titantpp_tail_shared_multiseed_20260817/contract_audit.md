# Intermittent T0/T1 Three-seed Contract Audit

## 판정

- 상태: **통과**
- 재실행 필요 모델: 없음
- 평가 범위: validation-only
- held-out test: 미사용
- 주의: 세 artifact의 source revision은 동일하지 않지만, source diff에서 T0의 입력, encoder, time head, log-MSE quantity head와 checkpoint selection 경로가 유지됨을 확인했다.

## Artifact

| 역할 | 경로 | Source revision | Seeds |
| --- | --- | --- | --- |
| T0 backbone control | `paper/results/count_aware_tpp_backbone_control_20260812/source_5080` | `044add1f3de768d804d9f0269fd0013bd9658a35` | 42, 52, 62 |
| TitanTPP-T1 seed 42 | `search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816_5080_fresh_rerun` | `7de638a5c9290f79dae02a40fd22839aba9802e7` | 42 |
| TitanTPP-T1 extension | `search_artifacts/count_aware_tail_shared_multiseed_extension_e300_20260817` | `d7fb2da367fa7efbe2de232a3394c2af50c36bcf` | 52, 62 |

## 고정 조건

| 항목 | 공통 값 | 판정 |
| --- | --- | --- |
| Dataset SHA-256 | `85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f` | 일치 |
| Split manifest SHA-256 | `393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04` | 일치 |
| Split rows | train 398,824 / validation 86,285 / test 88,019 | 일치 |
| Maximum epochs | 300 | 일치 |
| Batch size | 128 | 일치 |
| Learning rate | 0.001 | 일치 |
| Lookback | 520 weeks | 일치 |
| Maximum sequence length | 256 | 일치 |
| Hidden dimension | 64 | 일치 |
| Minimum epochs | 40 | 일치 |
| Early-stopping patience | 40 | 일치 |
| Selection | minimum validation joint objective | 일치 |
| Restore | best validation checkpoint | 일치 |
| Evaluation | validation-only | 일치 |
| Held-out test | not evaluated | 일치 |

T0는 `time NLL + log1p quantity MSE`, T1은 여기에 train-only로 고정한 tail auxiliary loss를 추가한다. 이 차이는 의도한 loss ablation이며 계약 불일치가 아니다.

## Epoch 및 Checkpoint

| Model | Seed | Best epoch | Completed epochs | Early stopped |
| --- | ---: | ---: | ---: | --- |
| Adapted RMTPP-T0 | 42 | 207 | 247 | yes |
| Adapted RMTPP-T0 | 52 | 249 | 289 | yes |
| Adapted RMTPP-T0 | 62 | 202 | 242 | yes |
| Adapted THP-T0 | 42 | 198 | 238 | yes |
| Adapted THP-T0 | 52 | 228 | 268 | yes |
| Adapted THP-T0 | 62 | 243 | 283 | yes |
| TitanTPP-T0 | 42 | 200 | 240 | yes |
| TitanTPP-T0 | 52 | 202 | 242 | yes |
| TitanTPP-T0 | 62 | 291 | 300 | no |
| TitanTPP-T1 | 42 | 300 | 300 | no |
| TitanTPP-T1 | 52 | 180 | 220 | yes |
| TitanTPP-T1 | 62 | 259 | 299 | yes |

모든 run은 `status=success`, finite metric, checkpoint SHA-256, history와 completed epoch 일치를 충족했다.

## Source Compatibility

- `044add1` 이후에는 Log-Normal 및 tail-aware quantity variant와 추가 metric logging이 도입됐다.
- T0 branch는 기존과 동일하게 `log1p(quantity)` MSE를 사용하고 model parameter 구성과 point prediction을 유지한다.
- RMTPP, THP, TitanTPP encoder 구성과 공통 RMTPP형 time density head는 변경되지 않았다.
- `7de638a`와 `d7fb2da` 사이의 count-aware runner source diff는 없다.
- 따라서 strict same-revision 실험은 아니지만 model/data/training/selection 계약은 matched이며 기존 T0 run을 재실행하지 않는다.
