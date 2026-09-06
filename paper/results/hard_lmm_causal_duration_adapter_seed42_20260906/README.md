# Frozen-B causal duration scale adapter: Taxi seed-42 screening

## 판정

**단일 log-normal의 scale만 이력에 따라 바꾸는 causal adapter는 공통 Time NLL 개선안으로 채택하지 않는다.** Taxi validation에서 후보의 continuous-density NLL은 Frozen-B보다 `0.001769` 낮았지만, 이력 입력이 없는 1-parameter global-scale control보다 `0.000278` 높았고 matched A의 허용 상한도 `0.698345` 초과했다. 이 두 개의 유효한 사전 gate가 실패했으므로 Intermittent와 Instacart 실행 및 추가 seed를 중단했다.

실행 당시 함께 기록한 centered-bin 점수도 Frozen-B보다 `0.007010` 악화했다. 다만 사후 감사에서 첫 bin `(0.5,1.5]` 아래의 `F(0.5)`를 어느 정수에도 배정하지 않아 전체 확률 합이 `S(0.5)`라는 문제가 확인됐다. 따라서 이 값은 **legacy 민감도 점수**로만 남기며 최종 기각 근거나 proper Time NLL로 사용하지 않는다. 이 교정은 독립적인 continuous gate 두 개에 따른 기각을 바꾸지 않는다.

정규화된 양의 정수 관측 likelihood를 사후 재생한 결과에서는 후보가 global control보다 `0.006821` 낮았다. 다만 두 경로는 continuous-density NLL로 서로 다른 epoch에서 선택됐고 둘 다 Frozen-B보다 악화했다. 같은 용량의 length-only control도 실행하지 않았으므로 이 차이를 raw duration 값의 조건부 신호로 해석하지 않는다. 이번 실험이 배제한 가설은 더 좁다. **Frozen-B의 location을 고정한 단일 log-normal scale adapter가 채택 가능한 공통 Time NLL 개선을 만든다**는 가설이 지지되지 않았다.

## 실험한 후보

- Source는 raw-RMSE로 선택된 seed-42 Frozen-B checkpoint다.
- B의 encoder, memory, quantity head, log-normal location `mu_B`, 시간 중앙값을 고정했다.
- 후보는 active context의 관측 `log1p(delta_t)`만 읽는 1-layer GRU(`hidden_size=8`)와 scale residual projection으로 구성했다.
- 다음 target, padding, mark, quantity, B hidden state는 adapter 입력에서 제외했다.
- projection을 0으로 초기화해 epoch 0의 전체 duration distribution을 B와 일치시켰다.
- 후보는 273개 파라미터를 학습했다. Global control은 같은 B scale에 하나의 공통 residual만 적용했다.
- 학습은 train split, 선택은 validation continuous-density NLL의 가장 이른 strict minimum으로 제한했다.

계약은 [`hard_lmm_causal_duration_adapter_v1.json`](../../contracts/hard_lmm_causal_duration_adapter_v1.json)에 있다.

## Taxi validation 결과

낮을수록 좋다. 마지막 열은 실행 당시의 subnormalized centered-bin telemetry이며 proper NLL이 아니다. Continuous-density NLL은 관측된 정수 bucket을 잠재 연속시간의 한 점으로 평가한 값이다. 서로 다른 열 사이의 절댓값은 직접 비교하지 않는다.

| 경로 | 선택 epoch | 완료 epoch | Continuous-density NLL | Legacy centered-bin score |
|---|---:|---:|---:|---:|
| A, matched duration head | - | - | -0.443555791 | 0.824442863 |
| Frozen-B | 0 | 0 | 0.266558144 | 0.899918847 |
| Global-scale control | 5 | 13 | 0.264511399 | 0.911263844 |
| Causal scale adapter | 1 | 9 | 0.264789579 | 0.906928920 |

| 사전 gate | 관측 차이 또는 상태 | 판정 |
|---|---:|:---:|
| Candidate continuous NLL `<= B` | `-0.001768565` | 통과 |
| Candidate continuous NLL `<= A + 0.01` | 허용 상한보다 `+0.698345370` | **실패** |
| Candidate legacy centered-bin score `<= B` | `+0.007010074` | 실패했으나 최종 판정에서 제외 |
| `global control - candidate >= 0.005` | `-0.000278180` | **실패** |
| B quantity prediction bitwise identity | 일치 | 통과 |
| B time median bitwise identity | 일치 | 통과 |
| B base location bitwise identity | 일치 | 통과 |
| Source model state identity | 일치 | 통과 |

후보가 B보다 낮춘 continuous NLL `0.001768565`는 history attribution gate를 통과하지 못했다. 더 단순한 global control의 continuous NLL이 더 낮았기 때문이다.

### 정규화된 양의 정수 likelihood 사후 재생

Taxi target은 연속 interarrival가 아니라 occupied-hour bucket 사이의 양의 정수 gap이다. 두 가지 정규화 convention으로 **같은 선택 checkpoint**를 validation에서 재평가했다. 학습이나 checkpoint 재선택은 하지 않았다.

| 경로 | `D=max(1, round(T))` NLL | `D=round(T) \mid D>=1` NLL |
|---|---:|---:|
| A, matched duration head | 0.811792 | 0.811328 |
| Frozen-B | 0.853267 | 0.862033 |
| Global-scale control | 0.871290 | 0.878591 |
| Causal scale adapter | 0.864470 | 0.872286 |
| RMTPP, matched duration head | 1.022646 | 1.023399 |
| THP, matched duration head | 0.857599 | 0.858261 |

두 convention 모두 후보는 global control보다 각각 `0.006821`, `0.006304` 낮았다. 반면 Frozen-B보다 각각 `0.011202`, `0.010254` 높고 A보다 `0.052677`, `0.060959` 높았다. 두 경로를 normalized likelihood로 선택하지 않았고 선택 epoch도 후보 1, control 5로 다르므로, 이 차이는 **후보가 해당 operating point에서 global control보다 덜 악화했다**는 사후 관측으로만 남긴다. Length-only control이 없으므로 prefix 길이, duration 값, recurrent capacity 및 평균 scale 이동의 효과는 분리되지 않았다.

이 표는 continuous-density NLL로 선택된 checkpoint의 사후 민감도 분석이다. A와 후보를 normalized likelihood로 다시 선택한 matched-selector 비교가 아니므로 다음 후보의 최종 기준선으로 그대로 사용하지 않는다. 재현 파일은 [`taxi_normalized_integer_likelihood_replay_20260906.json`](taxi_normalized_integer_likelihood_replay_20260906.json)과 [`replay_normalized_integer_likelihood.py`](replay_normalized_integer_likelihood.py)다.

동일한 네 기준 모델을 관측 정의만 바꿔 재생한 감사에서 B와 A의 **각 점수 안에서의 수치 격차**는 continuous-density NLL 기준 `0.710114`, legacy centered-bin 기준 `0.075476`이었다. 후자가 전자보다 `89.37%` 작지만, subnormalized 점수이므로 이 비율은 설명적인 민감도 기록일 뿐이다. Taxi validation 8,268건 중 `delta_t=1`이 6,790건(`82.12%`)이라는 사실과 함께 보면, 다음 실험에서 이력 표현과 decoder family 및 정규화된 관측 likelihood의 영향을 분리할 필요가 있다.

### A/B location-scale 출력 분해

A와 B가 같은 Taxi validation 표본에 출력한 `mu`와 `sigma`를 교차해 재생했다. 이는 새로운 모델이나 학습 결과가 아니라, 관측된 A-B 격차가 어느 출력 파라미터 축과 더 강하게 연결되는지 보는 진단이다.

| Location / scale 조합 | Continuous-density NLL | Fold-to-one NLL | Time median MAE |
|---|---:|---:|---:|
| A `mu` / A `sigma` | -0.443556 | 0.811792 | 0.747084 |
| B `mu` / B `sigma` | 0.266558 | 0.853267 | 0.749579 |
| A `mu` / B `sigma` | 0.242048 | 0.839292 | 0.747084 |
| B `mu` / A `sigma` | -0.370101 | 0.815147 | 0.749579 |

B의 location을 유지하고 A의 scale만 넣으면 B-A 격차의 `89.66%`를 continuous-density NLL에서, `91.91%`를 fold-to-one NLL에서 회복했다. 반대 조합인 A location/B scale의 회복률은 각각 `3.45%`, `33.69%`였다. 따라서 Taxi의 관측된 격차는 location보다 **표본별 scale 출력과 더 강하게 연결된다.** B의 location과 time median을 고정했던 선택 자체가 핵심 장애였다는 근거는 없다.

이 교차 재생은 A scale을 만들어 낸 입력 정보나 학습 경로를 알려주지 않으며 causal attribution도 아니다. 현재 GRU adapter가 A와 같은 scale 구조를 학습하지 못했다는 사실과 함께 사용해, 다음 실험에서는 scale objective·selector와 scale 표현을 한 번에 하나씩 검증한다. 재현 파일은 [`taxi_location_scale_hybrid_replay_20260906.json`](taxi_location_scale_hybrid_replay_20260906.json)과 [`replay_location_scale_hybrids.py`](replay_location_scale_hybrids.py)다.

## 수량 성능 보존

Frozen-B의 수량 예측 digest는 학습 전후에 일치했다. 따라서 다음 값은 B와 계산상 동일하다.

| Taxi quantity metric | 값 |
|---|---:|
| MAE | 28.674020 |
| RMSE | 88.194997 |
| Body MAE | 18.997478 |
| `> p99` MAE | 332.769744 |

수량 경로의 보존은 성공했다. Time NLL gate 실패를 수량 경로의 drift로 설명할 수 없다.

## 실패가 말해 주는 것

선택된 후보의 bounded log-scale residual은 평균 `-0.031293`, 표준편차 `0.002338`, 범위 `[-0.033002, -0.020328]`이었다. 표본별 변동은 평균 이동보다 매우 작았으며, adapter가 사실상 거의 일정한 scale 축소를 학습했다.

학습 NLL은 epoch 1의 `-0.005748`에서 epoch 9의 `-0.264180`으로 계속 낮아졌다. 같은 구간에서 validation continuous NLL은 `0.264790`에서 `0.582294`로, legacy centered-bin score는 `0.906929`에서 `1.273153`으로 높아졌다. 가장 이른 validation optimum을 복원했다. 이 발산은 과적합과 train/validation 분포 이동을 포함한 일반화 실패를 나타내며, 현재 증적만으로 두 원인을 분리하지 않는다.

Continuous density는 관측 격자의 한 점에서 density를 평가한다. 실행 당시 centered-bin 계산은 정수 `t`를 `max(0.5, t-0.5) < T <= t+0.5`로 두었지만 양의 정수 support에서 정규화되지 않았다. 후보가 continuous NLL만 소폭 낮춘 사실은 확인되지만, 이 legacy 점수로 실제 관측 확률질량의 변화를 주장하지 않는다. Continuous 기준의 history attribution은 실패했고, normalized 사후 재생만으로 attribution을 되살릴 수도 없다. 두 결과를 함께 보면 이력 신호의 부재보다 **목적함수·관측 모델·location 고정 제약**을 다음에 분리해 검증해야 한다.

이 결과는 단일 unimodal log-normal의 고정 location과 하나의 scale이라는 출력 제약을 **다음 검토 가설**로 좁힌다. 이력 자체의 가치와 decoder family의 유연성을 분리하려면 같은 분포군의 history-conditioned 후보, history-free control, same-capacity length-only control을 함께 비교해야 한다.

## 실행 및 증적 경계

- 로컬 관련 계약·회귀 테스트: 45 passed
- CUDA 계약 테스트: 32 passed, RTX 5090 forward/backward sentinel 통과
- Full Taxi seed-42 runtime: `59.361 s`
- Peak CUDA memory: allocated `250,416,128 B`, reserved `383,778,816 B`
- Train/validation target: `38,393 / 8,268`
- Source revision: `17b215a5c74453940a627dee049a5ee91796f005`
- Source manifest SHA-256: `9683e34c036fe9b3df262e8d025d5f5e5d98ae3c8aa7364e61eeb4c6d49053f3`
- Contract SHA-256: `2a456ae64eae180f2c983a3c169c982e15b0f38877564f7613884e1ba63a4cbb`
- Local synchronized artifact: `search_artifacts/hard_lmm_causal_duration_adapter_seed42_5090_20260906_17b215a`
- Remote artifact: `/home/leekwanhyeong/artifacts/hard_lmm_causal_duration_adapter_seed42_5090_20260906_17b215a`

재현 가능한 요약과 원본 감사 정보는 다음 파일에 고정했다.

- [`metrics.csv`](metrics.csv): A·B·RMTPP·THP와 후보·control의 Taxi validation 지표
- [`validation_audit.json`](validation_audit.json): 실행 범위, gate, 보존 조건, 학습 궤적과 checksum
- [`taxi_duration_likelihood_fourway_audit.json`](taxi_duration_likelihood_fourway_audit.json): 네 모델의 continuous/legacy centered-bin 계산을 보존한 원본 감사
- [`taxi_normalized_integer_likelihood_replay_20260906.json`](taxi_normalized_integer_likelihood_replay_20260906.json): 여섯 경로의 normalized positive-integer validation 재생과 입력 hash
- [`replay_normalized_integer_likelihood.py`](replay_normalized_integer_likelihood.py): normalized likelihood 재생 스크립트
- [`taxi_location_scale_hybrid_replay_20260906.json`](taxi_location_scale_hybrid_replay_20260906.json): A/B location-scale 교차 재생 결과와 입력 hash
- [`replay_location_scale_hybrids.py`](replay_location_scale_hybrids.py): 교차 재생 스크립트
- [`duration_observation_semantics_audit.md`](duration_observation_semantics_audit.md): 정수 gap 생성 과정과 likelihood 정규화 교정
- [`artifact_sha256.txt`](artifact_sha256.txt): 동기화한 24개 5090 artifact의 파일별 checksum
- [`build_result.py`](build_result.py): 동기화된 원본 artifact와 저장된 likelihood audit의 checksum을 검사하고 요약을 재생성하는 builder

실험 runner와 재생 스크립트는 held-out test를 불러오지 않았고 test prediction·loss도 계산하지 않았다. 다만 최종 문서 감사 중 별도의 read-only shell check가 Intermittent parquet를 split별로 요약하면서 test `delta_t`의 count·min·max·정수 여부·상위 5개 빈도를 출력했다. 이는 후보 판정 뒤에 발생했고 모델·threshold·방법 선택에는 사용하지 않았다. 따라서 Intermittent test는 **model-unevaluated**이지만 더 이상 analyst-blind라고 쓰지 않는다. Taxi와 Instacart test에는 이 확인을 수행하지 않았다.

Seeds 52·62는 실행하지 않았고, Taxi-first 중단 규칙에 따라 Intermittent와 Instacart의 causal scale-adapter 학습도 실행하지 않았다. 이 결과로 세 데이터셋의 공통 성능이나 test 성능을 주장할 수 없다.

## 다음 후보의 상태

다음 단일 후보는 **같은 K=1 causal scale adapter의 objective·selector 정렬**이다. 먼저 Frozen A/B duration head를 같은 normalized positive-integer likelihood로 맞춘 뒤, 기존 273-parameter adapter를 그 likelihood로 학습·선택한다. 이렇게 해야 현재 실패에서 architecture와 관측 목적을 한꺼번에 바꾸지 않는다. 이 후보가 실패하면 B-anchored first-bin hurdle, 그 뒤에 `K=2` shared-`mu_B` scale mixture를 검토한다. 제안과 계약 초안만 [`time_nll_methodology_review.md`](time_nll_methodology_review.md)에 기록하며, 세 후보 모두 아직 구현하거나 학습하지 않았다.
