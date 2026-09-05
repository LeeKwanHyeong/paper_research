# Transition-error Hard-LMM train-only 진단 결과

## 결론

Prototype-conditioned transition-error 후보는 세 데이터셋 공통 구현 gate를 통과하지 못했다. 따라서 계약에 따라 모델 경로를 구현하지 않았고 RTX 5090 CUDA·e1·e300도 실행하지 않았다. 이 중단은 실행 오류가 아니라, 구현 전에 고정한 train-only 필요조건의 실패다.

과거 예측오차를 포함한 strong control은 Intermittent와 Taxi에서 frozen T0보다 낮은 OOF log-MSE를 보였다. 그러나 현재 top-4 prototype별로 오차를 모은 후보는 strong control과 fixed sham에 비해 1%에 훨씬 못 미치는 추가 효과만 냈다. 이 결과는 **과거 오차를 이용한 적응 보정 가능성**과 **Hard-LMM prototype 연결의 추가 가치**를 구분한다. 이번 음성 결과는 후자를 지지하지 않는다.

## 동결된 질문과 판정

각 합법적 target window 안에서 이미 관측된 전이에 대해서만 다음 값을 만들었다.

```text
e_t = log1p(q_t) - softplus(stop_gradient(a_(t-1)))
```

`e_t`를 이전 T0 state의 top-4 prototype bucket에 누적하고, 현재 state의 top-4 bucket 평균을 quantity preactivation 보정 feature로 사용했다. Strong control에는 T0 preactivation, 이력·전이 수, bucket count, 최근 오차와 전체 과거 평균 오차를 넣었다. Sham은 동일 window의 과거 write assignment만 순환 이동했다.

데이터셋별 최대 8,192개의 train target을 series-disjoint 2-fold로 cross-fit했다. 후보가 strong control과 sham 각각보다 pooled log1p MSE를 1% 이상 개선하고, 두 fold 방향·10,000회 paired series-bootstrap 하한·body raw MAE guard를 모두 통과해야 구현할 수 있도록 결과 열람 전에 고정했다.

## 주요 결과

| 데이터셋 | T0 log-MSE | Strong control | Prototype 후보 | 후보 vs control | 후보 vs sham | 후보 body MAE vs T0 | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Intermittent | 0.00709070 | 0.00671516 | 0.00671577 | -0.009% | +0.169% | 12.226% 개선 | 실패 |
| Taxi | 0.17406284 | 0.16103328 | 0.16088528 | +0.092% | +0.122% | 14.447% 개선 | 실패 |
| Instacart | 0.24740824 | 0.24643455 | 0.24642742 | +0.003% | +0.006% | 0.134% 악화 | 실패 |

세 데이터셋 모두 1% 최소 효과에 미달했다. Intermittent는 control 대비 pooled 방향부터 음수였다. Taxi와 Instacart도 한 fold의 control 대비 방향, Bonferroni bootstrap 하한 또는 body guard가 실패했다. 여섯 bootstrap 하한은 모두 양수가 아니었다.

## 왜 prototype 기여로 이어지지 않았는가

후보의 prototype-conditioned 요약은 prototype을 쓰지 않은 전체 과거 오차 평균과 강하게 겹쳤다.

| 데이터셋 | Prototype 요약과 전체 평균 상관 | Prototype 요약과 sham 상관 | 현재 top-4 bucket 관측 비율 |
| --- | ---: | ---: | ---: |
| Intermittent | 0.673 | 0.709 | 0.953 |
| Taxi | 0.909 | 0.901 | 0.990 |
| Instacart | 0.973 | 0.956 | 0.856 |

특히 Taxi와 Instacart에서 실제 prototype 대응을 섞은 sham도 거의 같은 요약을 만들었다. Static top-4가 과거 전이 전체에서 반복되고 여러 bucket이 함께 갱신되면서, prototype별 값이 series 수준 평균 오차에 가까워진 것으로 해석할 수 있다. 이 표는 동결 gate를 바꾸지 않는 설명용 진단이며 새로운 후보 선택 기준으로 사용하지 않았다.

## 실행 및 무결성

- Intermittent 8,192개: fold target 4,239/3,953, unique series 1,911/1,827
- Taxi 8,192개: fold target 4,023/4,169, unique series 64/67
- Instacart 8,192개: fold target 4,132/4,060, unique series 3,959/3,889
- 세 checkpoint 모두 registry-pinned model-state SHA256와 일치했고 추출 전후 model·memory state가 같았다.
- 첫 batch의 공식 T0 수량 예측과 cache 예측 최대 차이는 세 데이터셋 모두 0이었다.
- 현재 target 수량·시간과 padding 값을 교란한 뒤 모든 허용 feature의 최대 차이는 0이었다.
- Validation과 held-out test row는 materialize하지 않았고 model parameter update도 없었다.
- 인과성·mask·OOF·sham·body veto·aggregate 무결성 계약 테스트 34개가 통과했다.

자세한 수치와 hash는 `analysis.json`, `evidence_decision.json`, `execution_manifest.json`, `reader_evidence.json`에 있다. 대용량 cache와 OOF row 배열은 `search_artifacts/hard_lmm_transition_error_probe_20260906`에 보존했다.

## 해석 범위

이 분석은 frozen T0가 이미 학습한 train series 안에서 보조 correction을 cross-fit한 필요조건 진단이다. Strong control의 개선을 validation 성능이나 새로운 최종 모델 성능으로 해석할 수 없다. 반대로 이번 실패도 모든 형태의 동적 memory가 불가능하다는 뜻은 아니다. 고정한 static prototype-conditioned error read가 공통 Backbone 후보로 구현할 만큼의 추가 신호를 보이지 않았다는 결론이다.

## 후속 작업

세 데이터셋의 raw 성능을 공통으로 높이는 다음 우선순위는 **quantile-adaptive loss와 checkpoint 정렬**이다. Train quantile로 tail weighting을 데이터셋별 동일 규칙으로 정렬하고, raw RMSE 목표와 checkpoint selection metric의 불일치를 함께 제거한다.

## 독립 감사의 제한

독립 감사에서 contract·registry·source·입력·cache·분석·OOF hash, 원본 dataset 행 정렬, OOF 예측 재구성, 10,000회 bootstrap을 다시 확인했고 차단 문제는 없었다. 실제 데이터의 target/padding 교란은 각 데이터셋 첫 batch에서 수행했으며 helper의 전 위치 인과성은 합성 계약 테스트로 보완했다. 실행 manifest에는 SciPy 버전이 빠졌고 독립 감사 환경은 `1.15.3`이었다.

Cyclic sham은 최종 target 시점에는 완전히 관측된 과거 이력만 사용하므로 target 누출은 없다. 다만 중간 token 관점에서는 뒤쪽 관측 이력의 prototype assignment를 사용할 수 있어 학습 가능한 인과 모델로 재사용할 수 없다. 이 sham은 final-state 진단 대조군으로만 사용했고, 후보가 sham과 무관하게 strong control 대비 세 데이터셋 모두 1% 기준을 크게 밑돌았으므로 공통 FAIL 판단에는 영향을 주지 않는다.
