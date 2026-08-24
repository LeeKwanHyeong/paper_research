# Online Retail II train-only time-scale 및 gradient clipping 감사 계약 v1

- 동결일: 2026-08-24
- 실행 서버: 5080 only
- 평가 범위: train-only
- validation / held-out test: 사용하지 않음

## 목적

Online Retail II의 hourly delta-time과 `legacy_clamped_rmtpp` time head 사이의
단위 부적합이 큰 Time NLL과 100% gradient clipping의 원인인지 확인한다. 모델,
quantity objective, optimizer와 context 계약은 바꾸지 않고 모델 입력 직전의
delta-time 단위만 비교한다.

## Variant 계약

| Variant | Delta-time divisor | 역할 |
| --- | ---: | --- |
| S0 raw hour | 1 | 기존 hourly 입력 음성 대조군 |
| S1 calendar day | 24 | 해석 가능한 hour-to-day 변환 |
| S2 train target median | train target p50 | train-only robust scale |
| S3 train target mean | train target mean | train-only mean scale |
| S4 train target p95 | train target p95 | time-gradient 상한 진단군 |

`seq`와 lookback은 원래 hourly 좌표를 유지한다. Loader가 만든 batch의 `delta_t`만
divisor로 나누므로 context window와 target 구성은 모든 Variant에서 동일하다.

## 고정 조건

- dataset: Online Retail II official fixed-split parquet의 train rows만 로드
- model: TitanTPP-T0 Hard-LMM
- quantity objective: direct `log1p(quantity)` MSE
- time head: `legacy_clamped_rmtpp`
- seed / epochs / batch size / learning rate: 42 / 3 / 128 / 0.001
- lookback / max sequence length: 8,760 hours / 256 events
- gradient clipping threshold: 1.0
- 최대 train batch: Variant·epoch당 16
- validation과 held-out test artifact를 생성하지 않음

## 측정 항목

- train target delta-time p50, p95, p99, maximum
- scaled-coordinate Time NLL과 Jacobian-corrected hourly Time NLL
- time-only, quantity-only, joint pre-clipping gradient norm
- joint gradient clipping count와 비율
- time-only와 quantity-only gradient가 clipping threshold를 넘는 비율
- initial `w * scaled_delta_t >= 10` 포화 표본 비율

## Train-only 안정성 판정

각 Variant는 다음 조건을 모두 충족해야 안정성 후보가 된다.

- 모든 loss, gradient와 parameter가 finite
- 최대 epoch joint objective 100 이하
- 최대 per-event Time NLL 10,000 이하
- time-only gradient의 clipping threshold 초과 비율 0.25 이하

Joint clipping 비율은 반드시 함께 기록하지만 hard gate로 사용하지 않는다. 축소 실제
데이터 실행에서 quantity-only gradient만으로도 clipping threshold를 넘을 수 있음을
확인했기 때문에, 이를 time-scale 실패로 잘못 분류하지 않기 위함이다.

이 감사는 validation 성능을 선택하지 않는다. 안정성 후보가 생기면 Online Retail II를
제외하지 않고 별도의 matched validation 계약에서 단위 변환을 재검증한다. 모든
후보가 실패하면 legacy head를 유지한 Online Retail II 본실험은 중단한다.
