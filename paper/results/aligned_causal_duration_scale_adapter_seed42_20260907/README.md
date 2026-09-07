# Objective-aligned causal duration scale adapter 결과

## 결론

공통 causal time adapter는 **Taxi 선행 게이트에서 기각**했다. Frozen aligned-B의 수량 경로, duration location과 time median을 고정하고 과거 `log1p(duration)`만 읽는 GRU8 scale adapter를 normalized positive-integer likelihood로 학습했지만, validation selector는 candidate와 두 control 모두 epoch 0을 선택했다.

| 경로 | Taxi validation normalized NLL | 선택 epoch |
|---|---:|---:|
| aligned-A | 0.6506978539 | 고정 기준선 |
| aligned-B | 0.6956474254 | 고정 기준선 |
| causal duration candidate | 0.6956474254 | 0 |
| same-capacity length-only control | 0.6956474254 | 0 |
| global-scale control | 0.6956474254 | 0 |

Candidate의 aligned-B 대비 개선은 `0`, 두 control 대비 개선도 `0`이다. aligned-A와의 격차는 `+0.0449495715`로 허용치 `+0.01`을 넘었다. Candidate는 epoch 1부터 8까지 모두 epoch 0보다 높은 validation NLL을 기록했고 early stopping됐다. 따라서 과거 duration 값에 따른 scale 조정으로 Taxi의 aligned-B 시간 성능을 회복한다는 가설은 지지되지 않는다.

## 실행 범위와 계약 확인

- Source commit: `a15513d918e47d2ce1272d2544269bb787bbcb58`
- Contract SHA-256: `b5fe1300fb2bfe48498644c070b289f71f14c36b821f8c38c5f643571a72aeec`
- Runtime: NVIDIA GeForce RTX 5080, PyTorch `2.11.0+cu130`
- Remote artifact: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/aligned_causal_a15513d_5080_20260907`
- Remote decision SHA-256: `369f7e54b9c47a8f5dae90c6061d01118fe0394a5992ea32c14906b1d1c1bddf`

Intermittent, Taxi와 Instacart의 full-data e1은 모두 성공했다. 각 e1은 전체 train/validation 표본, CUDA gradient, finite loss, checkpoint 저장·재생, source state 불변성, target 제외, 수량 prediction과 time median의 exact identity를 확인했다. Held-out test는 읽거나 평가하지 않았다.

Taxi full fit에서도 quantity prediction digest는 aligned-B의 `98d3128c638ee53ec9c7fba8dfe8bfff1f11ea6074daa94dd08540a8ab9acc57`와 같았다. 수량 MAE `28.6740202512`, RMSE `88.1949966535`, body MAE `18.9974781212`, `>p99` MAE `332.7697444867`가 그대로 보존됐다. 시간 median과 aligned-B model state도 변하지 않았다.

계약에 따라 Taxi 실패 뒤 Intermittent와 Instacart full fit, 추가 seed와 held-out test를 실행하지 않았다. 이 결과는 aligned-B의 Taxi 시간 격차가 단순한 scale calibration이나 causal duration-prefix scale residual로 해결되지 않음을 뜻한다. 다음 구조 후보를 검토한다면, 이미 사전 순위에서 정한 **B-anchored first-bin hurdle과 conditional tail**이 대상이다. 이는 `D=1` 질량과 오른쪽 tail을 분리하므로 현재 K=1 log-normal scale-only 표현이 다루지 못한 관측 분포 형태를 직접 검증한다.

## 실패 이력

최초 원격 실행은 source archive에 `sample_data` mount가 없어 프로젝트 루트 탐지 단계에서 학습 전에 종료됐다. 해당 로그는 remote artifact의 `e1/intermittent_frozen_5000/bootstrap_failed.log`로 보존했다. 데이터 경로를 기존 checksum 고정 source에 read-only로 연결하고 import를 재검증한 뒤 동일 commit으로 재실행했다. 재실행 결과 로그에는 traceback, runtime error 또는 CUDA out-of-memory가 없으며 최종 exit code는 `0`이다.
