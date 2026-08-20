# TitanTPP Final Time Head + T1 통합 검증 결과

## 판정

- 범위: Intermittent validation only, seed 42
- 공통 구조: TitanTPP Hard-LMM + T1 tail-shared quantity loss
- 비교 변수: time-head family만 변경
- safety gate: 실패
- 선택: H0 scaled exact 유지, H3 log-normal duration 미채택
- held-out test: 사용하지 않음

H3는 Time NLL과 joint objective를 크게 낮췄지만 quantity body와 tail을 동시에
손상했다. 따라서 time likelihood 개선만으로 통합 모델을 교체하지 않는다.

## Best-checkpoint 비교

| Head | Best epoch | Time NLL | Joint objective | Quantity MAE | Quantity RMSE | <=p95 MAE | >p99 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 scaled exact | 22 | 0.822590 | 0.829072 | 0.699481 | 1.650459 | 0.539278 | 6.347091 |
| H3 log-normal | 22 | 0.474637 | 0.482049 | 1.116539 | 3.643457 | 0.597421 | 22.377996 |

H3의 Time NLL 변화는 `-0.347953`으로 time gate를 통과했다. 반면 quantity MAE는
`59.62%`, RMSE는 `120.75%`, `<=p95` MAE는 `10.78%`, `>p99` MAE는
`252.57%` 악화돼 각 `2%` safety limit를 모두 초과했다.

## 학습 안정성

두 run은 최대 300 epoch, minimum 40, patience 40 계약에서 모두 62 epoch에
early stopping됐고 모든 history 값은 finite였다.

| Head | Mean clipping | Final clipping | Best-epoch pre-clip norm | Maximum train joint |
| --- | ---: | ---: | ---: | ---: |
| H0 scaled exact | 21.36% | 18.82% | 83,447,048.64 | 19,092,575,803.70 |
| H3 log-normal | 99.94% | 100.00% | 3.49 | 0.7933 |

H0는 validation quantity가 상대적으로 좋지만 train time loss와 gradient가 반복적으로
폭증했다. H3는 loss spike를 제거했지만 거의 모든 batch가 gradient clipping 대상이었고,
joint-objective checkpoint가 quantity에 불리한 representation을 선택했다. H3의 낮은
Time NLL을 안정적인 통합 개선으로 해석할 수 없는 이유다.

## 결론

H3는 독립적인 time-density 후보로는 유효하지만 현재 shared Hard-LMM + T1 계약과는
결합하지 않는다. 후속 실험에서 H3 상수나 checkpoint를 validation quantity에 맞춰
조정하지 않는다. Time head를 다시 검토하려면 time과 quantity의 optimizer 또는
representation 경로를 사전에 정의한 별도 가설이 필요하다.

원격 학습은 정상 완료됐으나 runner의 마지막 comparator 호출은 프로젝트 루트가
`PYTHONPATH`에 없어 실패했다. 동일 source의 comparator를 로컬에서 재실행했고 계약
테스트 `7 passed`와 exact gate 결과를 확인했다. Runner에는 프로젝트 루트
`PYTHONPATH`를 명시해 재발을 방지했다.

후속 train-only gradient attribution에서는 H3 best/final의 모든 batch가 clipping됐고
time head가 joint squared-gradient norm의 `93.30%`/`86.53%`를 차지했다. Shared
encoder의 time gradient도 quantity gradient보다 best 기준 약 `7.06배` 컸다. 따라서
H3 실패는 지속적인 gradient 방향 충돌보다 time-gradient scale dominance와 log-domain
quantity 목적의 raw-error 불일치로 정리한다. 세부 결과는
`paper/results/titantpp_h0_h3_gradient_attribution_20260820/result_analysis.md`에 둔다.
