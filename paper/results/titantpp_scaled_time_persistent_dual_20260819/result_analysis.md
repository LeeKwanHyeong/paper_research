# TitanTPP Scaled-Time Persistent/Dual Memory 결과 분석

## 판정

- 실험 상태: 완료
- Scope: Intermittent seed 42, validation only
- Held-out test: 사용하지 않음
- Memory gate: 실패
- 유지 모델: M1 Hard-LMM
- M2/M3a/M3b multi-seed: 진행하지 않음

실행 당시 source revision은 `6bf27cbca219c010c245a6293f27abf14fc6ccb6`이다.
Source manifest의 파일 checksum은 로컬 코드와 `14/14` 일치했고, 원격과 로컬에서
생성한 comparator 결과 세 파일의 SHA-256도 모두 일치했다.

## Matched 계약

다섯 Variant는 Intermittent frozen-5000 fixed split, seed 42, 최대 300 epoch,
minimum epoch 40, patience 40, batch 128, learning rate `1e-3`, lookback 520주,
maximum sequence length 256, hidden dimension 64를 동일하게 사용했다. 모든 모델은
persistent token 16개, mark-free log-MSE quantity head, train-only scale 3의 scaled
exact RMTPP time head를 사용했다. Checkpoint는 minimum validation joint objective로
선택했다.

## Best Validation 결과

| Variant | Best epoch | MAE | RMSE | <=p95 MAE | Time NLL | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M0 Persistent-only | 14 | 0.659124 | 1.683520 | 0.473992 | 0.821819 | 진단군 |
| M1 Hard-LMM | 22 | 0.645580 | 1.733665 | 0.464500 | 0.820917 | 유지 |
| M2 Persistent + Surprise | 29 | 0.783318 | 2.533716 | 0.454729 | 0.812943 | 실패 |
| M3a Dual shared | 22 | 0.693757 | 1.777577 | 0.513959 | 0.823367 | 실패 |
| M3b Dual adapter-only | 22 | 1.008081 | 2.869824 | 0.659239 | 0.826144 | 실패 |

M1은 M0보다 MAE와 <=p95 MAE가 각각 약 2% 낮고 Time NLL도 0.000902 낮았지만,
RMSE는 약 2.98% 높았다. 따라서 Hard-LMM의 기여는 작고 지표별 방향도 일관되지
않았다.

## Candidate Gate

| Candidate | MAE 개선률 | RMSE 개선률 | <=p95 MAE 변화 | Time NLL 변화 | 결과 |
| --- | ---: | ---: | ---: | ---: | --- |
| M2 | -21.34% | -46.15% | -2.10% | -0.007974 | 실패 |
| M3a | -7.46% | -2.53% | +10.65% | +0.002450 | 실패 |
| M3b | -56.15% | -65.54% | +41.92% | +0.005227 | 실패 |

개선률이 음수이면 M1보다 오차가 커졌다는 뜻이다. M2는 Time NLL과 <=p95 MAE를
개선했지만 p95-p99 MAE가 `2.8520 -> 3.9188`, p99 초과 MAE가
`7.0332 -> 14.9659`로 커져 전체 MAE와 RMSE가 악화됐다. Surprise memory의 이득은
time과 low/mid quantity 일부에 한정됐고 tail calibration을 잃었다.

M3a는 quantity `<=2` MAE를 `0.07767 -> 0.05976`, p99 초과 MAE를
`7.0332 -> 6.4038`로 줄였지만, `(2,31]`부터 p95까지의 오차가 커졌다. 서로 다른
구간의 이득을 합쳐도 전체 MAE와 body guardrail을 만족하지 못했다. M3b는 gradient를
adapter와 quantity head로 제한했지만 중간 수량과 tail이 모두 악화돼 gradient 분리가
해결책이 되지 못했다.

## History 길이별 결과

| Variant | History <=64 MAE | History 65-128 MAE | History >128 MAE |
| --- | ---: | ---: | ---: |
| M1 Hard-LMM | 0.803287 | 1.116519 | 0.115431 |
| M2 Surprise | 0.808658 | 1.501879 | 0.139382 |
| M3a Dual shared | 0.882993 | 1.176312 | 0.129353 |
| M3b Dual adapter-only | 0.961490 | 2.055896 | 0.133101 |

M2와 M3는 세 history 구간에서 모두 M1의 MAE를 낮추지 못했다. 이번 결과는 Surprise
memory나 dual route가 긴 history 표현을 더 잘 학습한다는 근거가 되지 않는다.

## Time Head 안정성

Scaled exact RMTPP head는 Jacobian, density integration, finite gradient 계약을
통과했고 validation metric도 finite했다. 다만 train joint objective는 첫 epoch 이후
대부분의 epoch에서 `10^6`을 넘었다.

| Variant | Train joint 중앙값 | Train joint 최댓값 |
| --- | ---: | ---: |
| M0 | 1.013e7 | 4.536e9 |
| M1 | 1.122e7 | 1.098e9 |
| M2 | 1.270e7 | 2.647e10 |
| M3a | 1.426e7 | 1.301e15 |
| M3b | 2.981e7 | 2.092e9 |

현재 `time_w_max=10/3`은 train maximum delta-time에서 `w * (dt / scale)=40`까지
허용한다. Clamp를 제거한 exact cumulative hazard에서 이 범위는 finite하더라도 매우
큰 loss와 gradient를 만든다. 따라서 이번 구현은 exact density의 수학적 계약은
충족했지만 optimization 관점의 time head 정상화까지 완료했다고 보기는 어렵다.

## Artifact 범위

- Launch contract, source manifest, launcher log, run summary, quantity/history breakdown을
  확인했다.
- Validation-only 계약 때문에 test summary와 test scale artifact는 생성되지 않았다.
- Mark-free formulation이므로 confusion과 per-class marker metric은 적용 대상이 아니다.
- 이번 runner는 별도 plot artifact를 생성하지 않았다. 수치 판정은 CSV와 comparator
  결과를 기준으로 재현했다.

## 결론과 다음 조건

현재 조건에서는 M1 Hard-LMM을 유지한다. M2와 M3를 multi-seed 또는 held-out test로
확장하지 않는다. 다음 time-head 실험에서는 train-only maximum delta-time을 기준으로
`w * tau` 상한을 8-10 수준으로 낮추고, intercept 범위와 time-head learning rate도
별도로 제한해야 한다. 새 time head가 train loss spike gate를 통과한 뒤 M0와 M1만
먼저 다시 비교하고, Hard-LMM의 독립 기여가 확인될 때에만 Surprise/dual memory를
다시 연다.
