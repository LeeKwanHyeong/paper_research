# Taxi Quantity Representation 통제 실험 결과

## 1. 실험 목적

이 실험은 RMTPP history encoder를 고정하고 quantity 출력 방식만 변경하여, exponent + residual 표현의 효과를 분리해서 확인하기 위해 수행했다. 모든 결과는 동일한 Taxi fixed split과 seed 42, 52, 62를 사용한 validation 결과다. Held-out test는 평가하지 않았다.

비교한 quantity interface는 다음과 같다.

- Uniform categorical binning
- Quantile categorical binning
- Raw-scale MSE regression: 음수 출력이 발생하는 기존 진단용 결과
- Min-max scaling + sigmoid regression: train 범위 내 출력 보장
- Log-scale regression: `log1p(q)` 목표, softplus 출력, `expm1` 복원
- Exponent + residual: magnitude mark와 within-mark residual 결합

Min-max와 log-scale regression은 train 데이터에서만 변환 통계를 계산하며, 복원된 quantity가 음수가 되지 않도록 구성했다. 따라서 기존 raw MSE에서 발생했던 음수 출력 문제를 제거한 공정한 regression baseline이다.

## 2. 전체 및 구간별 결과

아래 값은 세 seed의 validation MAE 평균이다. Quantity 구간 경계는 train 데이터에서 계산했다.

| Quantity 구간 | Log regression | Exponent + residual | 비교 |
|---|---:|---:|---:|
| 전체 | **34.399** | 65.858 | Log regression 47.8% 감소 |
| `q <= 7` | 1.668 | **1.624** | Exponent + residual 2.6% 감소 |
| `7 < q <= 686` | **26.583** | 54.162 | Log regression 50.9% 감소 |
| `686 < q <= 1,562` | **179.596** | 566.856 | Log regression 68.3% 감소 |
| `1,562 < q <= 3,449` | **275.291** | 388.326 | Log regression 29.1% 감소 |
| `q > 3,449` | 525.710 | **402.214** | Exponent + residual 23.5% 감소 |

전체 RMSE는 log regression이 113.319, exponent + residual이 185.161로, log regression이 38.8% 낮았다.

## 3. 해석

Log regression은 전체 MAE와 RMSE뿐 아니라 p50-p90, p90-p95, p95-p99 구간에서도 exponent + residual보다 낮은 오차를 보였다. 따라서 Taxi 결과로는 exponent + residual이 일반적인 log-scale regression보다 long-tail quantity를 전반적으로 더 정확하게 예측한다는 주장을 뒷받침할 수 없다.

Exponent + residual의 장점은 `q > p99`인 최상위 1% 구간에서만 나타났다. 이 결과는 극단 수요에서의 국소적인 이점 가능성을 보여주지만, 전체 long-tail 개선으로 일반화할 수는 없다. Seed별 일관성과 다른 데이터셋에서 동일한 현상이 나타나는지 추가 검증이 필요하다.

Raw MSE는 전체 MAE 35.491, RMSE 90.982로 수치상 강했지만, clipping 전 음수 예측이 발생했다. 따라서 이 결과는 regression 자체의 한계를 증명하는 근거가 아니라, positivity constraint가 없는 출력층의 문제를 보여주는 진단 결과로만 사용한다.

## 4. 과거 TitanTPP log-quantity 실험 확인

과거 artifact에서 `direct_log_qty`를 사용한 TitanTPP 실행은 두 건 확인됐다.

| Dataset | Epoch/seed | Best validation NLL | Quantity MAE | 자격 판정 |
|---|---|---:|---:|---|
| 구형 Intermittent | e50, seed 42 | 5.574 | 2.761 | 참고만 가능 |
| Instacart smoke | e1, seed 42 | 3.253 | 5.216 | CUDA 동작 확인용 |

구형 Intermittent 실행은 max sequence length 16, lookback 52, 단일 seed이며 이전 데이터와 모델 계약을 사용했다. 또한 당시 artifact에는 held-out test 평가가 포함되어 있다. 현재의 frozen 5,000-series Intermittent 계약이나 long-sequence 주장에는 사용할 수 없다.

Instacart 실행은 20개 series와 1 epoch만 사용한 smoke test이므로 성능 근거가 아니다.

과거 TitanTPP의 `direct_log_qty` 구현은 train log2 quantity를 전역 정규화하고 `exp2`로 복원한다. 이번 Taxi 통제 실험은 `log1p + softplus + expm1`을 사용한다. 두 구현은 모두 양수 quantity를 생성하지만 목표 변환과 출력 제약이 다르므로 직접 비교하거나 기존 checkpoint를 이어서 학습해서는 안 된다.

## 5. 논문과 후속 실험에 미치는 영향

현재 quantity contribution은 다음과 같이 수정해야 한다.

- 제외할 주장: exponent + residual이 log regression보다 전반적인 long-tail 예측에서 우수하다.
- 유지 가능한 관찰: exponent + residual은 Taxi의 p99 초과 구간에서 오차가 낮았지만, 전체 및 p90-p99에서는 log regression이 우수했다.
- 새 검증 목표: quantity head를 동일한 log regression으로 고정했을 때 Titan backbone이 RMTPP와 THP보다 긴 history에서 유의미하게 개선되는지 확인한다.

## 6. 다음 실행 순서

1. 세 backbone이 공유할 log regression 계약을 먼저 고정한다. 이번에 검증한 `log1p + softplus + expm1`을 기준으로 target, loss, history quantity input, checkpoint selection을 동일하게 맞춘다.
2. RMTPP, THP, Titan에 동일한 head가 적용되는지 로컬 계약 테스트와 CUDA smoke로 확인한다.
3. 먼저 seed 42 screening을 실행한다. Taxi에서는 THP-log와 Titan-log를, 최신 Intermittent에서는 RMTPP-log, THP-log, Titan-log를 비교한다.
4. Titan이 긴 sequence 구간에서 개선되는 신호가 있을 때만 seed 52와 62를 추가한다.
5. 최종 비교는 전체 평균과 함께 sequence length 구간별 MAE, RMSE, NLL을 제시한다. 긴 sequence 구간은 현재의 단순 `9+`보다 세분화하여 정의한다.
6. Titan의 개선이 확인되면 backbone contribution 중심으로 원고를 수정한다. 개선이 확인되지 않으면 현재 Titan backbone 주장을 유지하지 않고 논문 범위와 제출 일정을 다시 결정한다.

